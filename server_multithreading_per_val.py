import argparse
import socket
import struct
import threading
import traceback
from enum import Enum
import utility
import torch

import config
import mlt
import mlt_per_val


class TrainingPhase(Enum):
    BEGIN = 0
    MID = 1
    FINAL = 2


class Server:
    def __init__(self, host="0.0.0.0", tcp_port=9999, num_workers=3):
        # TCP setup
        self.host = host
        self.tcp_port = tcp_port
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # training setup
        self.training_phase = TrainingPhase.BEGIN
        self.prev_avg_acc = 0.0
        self.drop_rate = 0.0

        # worker setup
        self.worker_eval_acc = []
        self.worker_epochs = []
        self.worker_threads = []
        self.has_eval_acc = {}  # Track if any worker has sent eval data

        self.num_workers = num_workers
        self.received_gradients = []
        self.aggregation_round = 0

        # multithreading setup
        self.training_phase_lock = threading.Lock()
        self.lock = threading.Lock()
        self.recv_lock = threading.Lock()
        self.is_buffer_full = threading.Condition(self.lock)
        self.aggregation_complete_event = threading.Event()
        self.averaged_gradients = {}

        # server setup
        self.running = True
        self.tcp_connections = []
        self.conn_addr_map = {}

        self.write_to_server_port()

    def write_to_server_port(self):
        print("Writing server TCP port to .server_port file...")
        with open(".server_port", "w") as f:
            f.write(str(self.tcp_port))
            f.flush()

    def start(self):
        """
        Binds the main TCP listening port and starts accepting connections.
        """
        self.tcp_server.bind((self.host, self.tcp_port))
        self.tcp_server.listen()
        print(f"Server listening on {self.host}:{self.tcp_port}")

        aggregator_thread = threading.Thread(target=self.gradient_aggregator_loop)
        aggregator_thread.daemon = True
        aggregator_thread.start()

        try:
            while self.running:
                client_tcp_sock, addr = self.tcp_server.accept()
                self.tcp_connections.append(client_tcp_sock)
                self.conn_addr_map[client_tcp_sock] = addr

                print(f"\n ------ Accepted TCP connection from {addr} ------")

                thread = threading.Thread(
                    target=self.handle_worker, args=(client_tcp_sock, addr)
                )
                thread.start()
                self.worker_threads.append(thread)
        except KeyboardInterrupt:
            print("Server shutting down.")
        finally:
            self.shutdown()

    def gradient_aggregator_loop(self):
        """A loop that waits for all gradients, aggregates them, and signals workers to send."""
        while self.running:
            with self.is_buffer_full:
                self.aggregation_complete_event.clear()  # Reset the event for the next round

                while len(self.received_gradients) < self.num_workers:
                    print(
                        f"[Aggregator] Waiting... ({len(self.received_gradients)}/{self.num_workers} gradients received)"
                    )
                    self.is_buffer_full.wait()

                print(
                    f"\n[Aggregator] Woke up. Number of gradients received: {len(self.received_gradients)}. Starting aggregation for round {self.aggregation_round}."
                )

                all_gradients = self.received_gradients

                # no need to worry about the epoch / eval metadata
                gradient_keys = [
                    k
                    for k in all_gradients[0].keys()
                    if isinstance(all_gradients[0][k], torch.Tensor)
                ]

                averaged_gradients = {}
                for key in gradient_keys:
                    stacked_tensors = torch.stack(
                        [worker_grads[key] for worker_grads in all_gradients]
                    )
                    averaged_gradients[key] = torch.mean(stacked_tensors, dim=0)

                print("[Aggregator] Averaging complete for all keys")

                # --- ROUND-TRIP LOGIC: Prepare for broadcast ---
                self.averaged_gradients = averaged_gradients
                self.received_gradients.clear()
                self.aggregation_round += 1

                # Signal to all waiting worker threads that the new gradients are ready
                self.aggregation_complete_event.set()
                print(
                    f"[Aggregator] Event set for round {self.aggregation_round}. Worker threads will now send averaged gradients back.\n"
                )

    def handle_worker(self, client_tcp_sock, tcp_addr):
        """Manages the full round-trip lifecycle for a single worker."""
        dedicated_udp_sock = None
        have_received_metadata = False
        metadata_list = []

        try:
            dedicated_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # ask OS to find an ephemeral port
            dedicated_udp_sock.bind((self.host, 0))
            udp_port = dedicated_udp_sock.getsockname()[1]
            client_tcp_sock.sendall(struct.pack("!I", udp_port))

            socks = {"tcp": client_tcp_sock, "udp": dedicated_udp_sock}
            addrs = {"tcp": tcp_addr}

            signal_counter = 0

            while self.running:
                # 0. receive metadata from the worker
                if not have_received_metadata:
                    print(f"[{tcp_addr}] Waiting to receive metadata from worker...")
                    metadata_list = utility.receive_data_tcp(client_tcp_sock)

                    have_received_metadata = True
                    print(f"[{tcp_addr}] Received metadata from worker.")

                # 1. Receive gradients from the worker
                print(f"[{tcp_addr}] Waiting to receive gradients from worker...")
                result = mlt_per_val.recv_data_mlt(socks, tcp_addr, signal_counter, metadata_list)

                if result is None:
                    break

                gradients, worker_udp_addr, counter = result
                signal_counter = counter
                
                if config.DEBUG:
                    print(f"[{tcp_addr}] Counter should be 213? {signal_counter}")

                if gradients:
                    print(
                        f"[{tcp_addr}] Received gradient bundle from {worker_udp_addr}."
                    )
                    if "epoch" in gradients and "eval_acc" in gradients:
                        print(f"[{tcp_addr}] Received epoch data: {gradients['epoch']}")
                        print(
                            f"[{tcp_addr}] Received evaluation accuracy: {gradients['eval_acc']}"
                        )
                        # ---------- logic for training phase update BEGINS ---------
                        with self.training_phase_lock:
                            self.worker_eval_acc.append(gradients["eval_acc"])
                            self.worker_epochs.append(gradients["epoch"])
                            self.has_eval_acc[worker_udp_addr] = True
                            del gradients["eval_acc"]
                            del gradients["epoch"]

                            if all(self.has_eval_acc.values()):
                                print(
                                    f"[{tcp_addr}] All workers have reported eval accuracy."
                                )
                                self._training_phase_update()

                                if self.training_phase == TrainingPhase.BEGIN:
                                    self.drop_rate = config.BEGIN_DROP
                                elif self.training_phase == TrainingPhase.MID:
                                    self.drop_rate = config.MID_DROP
                                elif self.training_phase == TrainingPhase.FINAL:
                                    self.drop_rate = config.FINAL_DROP

                                print(
                                    f"[{tcp_addr}] Updated drop rate: {self.drop_rate}"
                                )
                        # ---------- logic for training phase update ENDS -----------

                    with self.is_buffer_full:
                        self.received_gradients.append(gradients)
                        print(
                            f"[{tcp_addr}] Added gradients to buffer ({len(self.received_gradients)}/{self.num_workers})."
                        )
                        if len(self.received_gradients) == self.num_workers:
                            print(
                                f"[{tcp_addr}] Final gradient received. Notifying aggregator."
                            )
                            self.is_buffer_full.notify()

                # --- ROUND-TRIP LOGIC: Wait for aggregation and send back ---
                print(f"[{tcp_addr}] Waiting for aggregation to complete...")
                self.aggregation_complete_event.wait()  # Wait until aggregator sets the event
                # ---------- memory barrier --------------------------

                print(
                    f"[{tcp_addr}] Aggregation complete. Sending averaged gradients back."
                )

                addrs["udp"] = worker_udp_addr

                # Use a new function to SEND BACK the whole dictionary
                signal_counter = self.send_gradient_dict(socks, addrs, signal_counter)
                
                print(
                    f"[{tcp_addr}] Finished sending averaged gradients. Ready for next round."
                )
                # --- END ---

        except (ConnectionResetError, BrokenPipeError):
            print(f"[{tcp_addr}] Worker connection lost unexpectedly.")
        except Exception as e:
            print(f"[{tcp_addr}] An error occurred in the worker thread: {e}")
            traceback.print_exc()
        finally:
            print(f"[{tcp_addr}] Cleaning up resources for this worker.")
            if client_tcp_sock:
                client_tcp_sock.close()
            if dedicated_udp_sock:
                dedicated_udp_sock.close()

    def send_gradient_dict(self, socks, addrs, signal_counter):
        """Sends a complete dictionary of gradients to a worker."""
        avg_gradients = self.averaged_gradients
        tcp_sock = socks["tcp"]

        # Send 'N' for No-evaluation data
        tcp_sock.sendall(b"N")

        num_subgradients = (
            len(avg_gradients) - 2 if "epoch" in avg_gradients else len(avg_gradients)
        )
        if num_subgradients != len(avg_gradients):
            raise ValueError(
                f"num_subgradients:{num_subgradients}, len(avg_gradients):{len(avg_gradients)} mismatch.\n"
                f"This should not happen because epoch and eval_acc has been deleted by me"
            )

        for key, tensor in avg_gradients.items():
            if not isinstance(tensor, torch.Tensor):
                print(f"Warning: Item with key '{key}' is not a tensor, skipping.")
                print(f"    Value is likely eval data: {tensor}")
                continue

            _, payload_bytes = mlt.serialize_gradient_to_custom_binary(
                tcp_sock, key, tensor
            )
            if payload_bytes is None:
                raise ValueError(
                    f"Failed to serialize tensor data for key '{key}'."
                )
            success = mlt_per_val.send_data_mlt(socks, addrs, payload_bytes, signal_counter)
            signal_counter += 1  # Increment signal counter after sending
        
            if not success:
                raise ValueError(
                    f"SERVER ERROR: Failed to send tensor data for key '{key}' using MLT. Aborting."
                )
            else:
                print(
                    f"\n--- SERVER successfully sent all the tensor data for key '{key}' ---\n"
                )

        print(
            f"\nFinished sending averaged gradients back to TCP: {addrs['tcp']} and UDP: {addrs['udp']}."
        )
        return signal_counter

    def shutdown(self):
        self.running = False
        self.tcp_server.close()
        with self.is_buffer_full:
            self.is_buffer_full.notify_all()
        self.aggregation_complete_event.set()
        for t in self.worker_threads:
            t.join()
        print("Server shutdown complete.")

    # -------------- HELPER FUNCTIONS ---------------------------------------
    def _training_phase_update(self):
        if len(self.worker_eval_acc) < 3:
            print("this should not happen. ABORTING.")
            return

        curr_avg_acc = sum(self.worker_eval_acc) / len(self.worker_eval_acc)
        acc_diff = curr_avg_acc - self.prev_avg_acc

        proposed_phase = self.training_phase  # Initially set as current phase

        if any(self.worker_epochs) == 0:
            proposed_phase = TrainingPhase.BEGIN
        elif acc_diff > 0.07:
            proposed_phase = TrainingPhase.BEGIN
        elif curr_avg_acc > 0.5:
            if acc_diff < 0.03:
                proposed_phase = TrainingPhase.FINAL
            elif acc_diff <= 0.07:
                proposed_phase = TrainingPhase.MID
            else:
                proposed_phase = TrainingPhase.BEGIN
        # Ensure no phase transitions if the accuracy is not above 50%, even if acc_diff is low
        else:
            proposed_phase = TrainingPhase.BEGIN

        # Ensure one-way state transitions:
        if proposed_phase.value >= self.training_phase.value:
            self.training_phase = proposed_phase

        self.prev_avg_acc = curr_avg_acc

        print(f"All worker eval acc: {self.worker_eval_acc}")
        print(f"All worker epochs: {self.worker_epochs}")
        print(f"Current averaged accuracy: {self.prev_avg_acc}")
        print(f"Current training phase: {self.training_phase}")

        self.worker_eval_acc.clear()
        self.worker_epochs.clear()
        self.has_eval_acc.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=str, default="9999")
    args = parser.parse_args()

    server_host = str(args.host)
    server_port = int(args.port)
    print(f"Starting server at {server_host}:{server_port}...")
    server = Server(host=server_host, tcp_port=server_port)
    server.start()
