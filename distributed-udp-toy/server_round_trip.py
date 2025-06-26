# server.py
import socket
import threading
import struct
import traceback
import json
import torch
import os
import sys

# Assuming mlt.py is in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlt


class MLServer:
    def __init__(self, host="0.0.0.0", tcp_port=9999, num_workers=3):
        self.host = host
        self.tcp_port = tcp_port
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.num_workers = num_workers
        self.received_gradients = []
        self.aggregation_round = 0

        # --- ROUND-TRIP LOGIC: Synchronization Primitives ---
        # Lock for controlling access to shared gradient buffers
        self.lock = threading.Lock()
        # Condition to signal when the gradient buffer is full
        self.is_buffer_full_condition = threading.Condition(self.lock)
        # Event to signal to all worker threads that aggregation is complete
        self.aggregation_complete_event = threading.Event()
        # The averaged gradients to be sent back to workers
        self.averaged_gradients = {}
        # --- END ---

        self.running = True
        self.worker_threads = []

    def start(self):
        """Binds the main TCP listening port and starts accepting connections."""
        self.tcp_server.bind((self.host, self.tcp_port))
        self.tcp_server.listen(3)
        print(
            f"Server listening on {self.host}:{self.tcp_port} for {self.num_workers} workers."
        )

        aggregator_thread = threading.Thread(target=self.gradient_aggregator_loop)
        aggregator_thread.daemon = True
        aggregator_thread.start()

        try:
            while self.running:
                client_tcp_sock, addr = self.tcp_server.accept()
                print(f"\n ----- Accepted TCP connection from {addr} -------")

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
            with self.is_buffer_full_condition:
                while len(self.received_gradients) < self.num_workers:
                    print(
                        f"[Aggregator] Waiting... ({len(self.received_gradients)}/{self.num_workers} gradients received)"
                    )
                    self.is_buffer_full_condition.wait()

                print(
                    f"\n[Aggregator] Woke up. Number of gradients received: {len(self.received_gradients)}. Starting aggregation for round {self.aggregation_round}."
                )

                all_gradients = self.received_gradients
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

                print(f"[Aggregator] Averaging complete for keys: {gradient_keys}")

                # --- ROUND-TRIP LOGIC: Prepare for broadcast ---
                self.averaged_gradients = averaged_gradients

                json_filename = (
                    f"server_averaged_gradients_round_{self.aggregation_round}.json"
                )
                with open(json_filename, "w") as f:
                    json.dump(
                        {k: v.tolist() for k, v in averaged_gradients.items()},
                        f,
                        indent=4,
                    )
                print(f"[Aggregator] Averaged gradients saved to {json_filename}")

                self.received_gradients.clear()
                self.aggregation_round += 1

                # Signal to all waiting worker threads that the new gradients are ready
                self.aggregation_complete_event.set()
                print(
                    "[Aggregator] Event set. Worker threads will now send averaged gradients back.\n"
                )

                if self.aggregation_round == 1:
                    print("[Aggregator] Target round 1 reached. Initiating server shutdown.")
                    self.running = False
                    # We must wake up any waiting threads so they can check the self.running flag and exit.
                    self.is_buffer_full_condition.notify_all()
                    self._unblock_main_thread()
            # --- END ---
    
    def _unblock_main_thread(self):
        """Creates a dummy connection to the server to unblock the main thread's accept() call."""
        try:
            # Use 127.0.0.1 for the dummy connection host
            with socket.create_connection(('127.0.0.1', self.tcp_port), timeout=1):
                pass
        except (socket.timeout, ConnectionRefusedError):
            # This can happen if the server socket is already closed, which is fine.
            pass


    def handle_worker(self, client_tcp_sock, tcp_addr):
        """Manages the full round-trip lifecycle for a single worker."""
        dedicated_udp_sock = None
        try:
            dedicated_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            dedicated_udp_sock.bind((self.host, 0))
            udp_port = dedicated_udp_sock.getsockname()[1]
            client_tcp_sock.sendall(struct.pack("!I", udp_port))

            socks = {"tcp": client_tcp_sock, "udp": dedicated_udp_sock}
            addrs = {"tcp": tcp_addr}

            while self.running:
                # 1. Receive gradients from the worker
                print(f"[{tcp_addr}] Waiting to receive gradients from worker...")
                result = mlt.recv_data_mlt(socks)

                if result is None:
                    break

                gradients, worker_udp_addr = result

                if gradients:
                    print(
                        f"[{tcp_addr}] Received gradient bundle from {worker_udp_addr}."
                    )
                    with self.is_buffer_full_condition:
                        self.received_gradients.append(gradients)
                        print(
                            f"[{tcp_addr}] Added gradients to buffer ({len(self.received_gradients)}/{self.num_workers})."
                        )
                        if len(self.received_gradients) == self.num_workers:
                            print(
                                f"[{tcp_addr}] Final gradient received. Notifying aggregator."
                            )
                            self.aggregation_complete_event.clear()  # Clear event from previous round
                            self.is_buffer_full_condition.notify()

                # ---------- memory barrier --------------------------
                
                # --- ROUND-TRIP LOGIC: Wait for aggregation and send back ---
                print(f"[{tcp_addr}] Waiting for aggregation to complete...")
                self.aggregation_complete_event.wait()  # Wait until aggregator sets the event

                print(
                    f"[{tcp_addr}] Aggregation complete. Sending averaged gradients back."
                )
                
                addrs["udp"] = worker_udp_addr
                

                # Use a new function to send the whole dictionary
                self.send_gradient_dict(socks, addrs)
                print(
                    f"[{tcp_addr}] Finished sending averaged gradients. Ready for next round."
                )

                if self.aggregation_round == 1:
                    self.running = False
                    break
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

    def send_gradient_dict(self, socks, addrs):
        """Sends a complete dictionary of gradients to a worker."""
        avg_gradients = self.averaged_gradients
        tcp_sock = socks["tcp"]

        # Send 'N' for No-evaluation data
        tcp_sock.sendall(b"N")

        num_subgradients = len(avg_gradients) - 2 if "epoch" in avg_gradients else len(avg_gradients)
        tcp_sock.sendall(struct.pack("!I", num_subgradients))

        for key, tensor in avg_gradients.items():
            if not isinstance(tensor, torch.Tensor):
                print(f"Warning: Item with key '{key}' is not a tensor, skipping.")
                print(f"    Value is likely eval data: {tensor}")
                continue
            
            payload_bytes = mlt.serialize_gradient_to_custom_binary(
                tcp_sock, key, tensor
            )
            if payload_bytes is None:
                print(f"Failed to serialize tensor data for key '{key}'. Skipping.")
                continue
            
            success = mlt.send_data_mlt(socks, addrs, payload_bytes)
            if not success:
                raise ValueError(
                    f"SERVER ERROR: Failed to send tensor data for key '{key}' using MLT. Aborting."
                )
            else:
                print(f"\n--- SERVER successfully sent all the tensor data for key '{key}' ---\n")

        print(
            f"\nFinished sending averaged gradients back to TCP: {addrs["tcp"]} and UDP: {addrs["udp"]}."
        )

    def shutdown(self):
        self.running = False
        self.tcp_server.close()
        with self.is_buffer_full_condition:
            self.is_buffer_full_condition.notify_all()
        self.aggregation_complete_event.set()
        for t in self.worker_threads:
            t.join()
        print("Server shutdown complete.")


if __name__ == "__main__":
    server = MLServer(num_workers=3)
    server.start()
