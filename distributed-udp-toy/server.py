# server.py
import json
import os
import socket
import struct
import sys
import threading
import traceback

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlt  # Import the user's mlt module


class MLServer:
    def __init__(self, host="0.0.0.0", tcp_port=6000, num_workers=3):
        self.host = host
        self.tcp_port = tcp_port
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # --- NEW: Attributes for aggregation ---
        self.num_workers = num_workers
        self.received_gradients = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.aggregation_round = 0
        # --- END NEW ---

        self.running = True
        self.worker_threads = []

    def start(self):
        """Binds the main TCP listening port and starts accepting connections."""
        self.tcp_server.bind((self.host, self.tcp_port))
        self.tcp_server.listen(3)
        print(f"Server listening on {self.host}:{self.tcp_port} for {self.num_workers} workers.")

        # --- NEW: Start the aggregator thread ---
        aggregator_thread = threading.Thread(target=self.gradient_aggregator_loop)
        aggregator_thread.daemon = True  # Allows main program to exit if this is the only thread left
        aggregator_thread.start()
        # --- END NEW ---

        try:
            while self.running:
                client_tcp_sock, addr = self.tcp_server.accept()
                print(f"Accepted TCP connection from {addr}")

                thread = threading.Thread(target=self.handle_worker, args=(client_tcp_sock, addr))
                thread.start()
                self.worker_threads.append(thread)

        except KeyboardInterrupt:
            print("Server shutting down.")
        finally:
            self.shutdown()

    def gradient_aggregator_loop(self):
        """A loop that waits for all gradients, aggregates them, and saves the result."""
        while self.running:
            with self.condition:
                # Wait until the buffer is full
                while len(self.received_gradients) < self.num_workers:
                    print(f"[Aggregator] Waiting... ({len(self.received_gradients)}/{self.num_workers} gradients received)")
                    self.condition.wait()  # Releases the lock and waits to be notified

                print(
                    f"\n[Aggregator] Woke up. Number of gradients received: {len(self.received_gradients)}. Starting aggregation for round {self.aggregation_round}."
                )

                # --- 1. Average the gradients ---
                all_gradients = self.received_gradients
                gradient_keys = [k for k in all_gradients[0].keys() if isinstance(all_gradients[0][k], torch.Tensor)]

                averaged_gradients = {}
                for key in gradient_keys:
                    # Stack all tensors for a given key and calculate the mean along the first dimension
                    stacked_tensors = torch.stack([worker_grads[key] for worker_grads in all_gradients])
                    averaged_gradients[key] = torch.mean(stacked_tensors, dim=0)

                print(f"[Aggregator] Averaging complete for keys: {gradient_keys}")

                # --- 2. Export the averaged gradients to a JSON file ---
                gradients_for_json = {key: tensor.tolist() for key, tensor in averaged_gradients.items()}
                json_filename = f"averaged_gradients_round_{self.aggregation_round}.json"

                with open(json_filename, "w") as f:
                    json.dump(gradients_for_json, f, indent=4)
                print(f"[Aggregator] Averaged gradients exported to {json_filename}")

                # --- 3. Clean up for the next round ---
                self.received_gradients.clear()
                self.aggregation_round += 1
                print("[Aggregator] Cleared buffer. Ready for next round.\n")

    def handle_worker(self, client_tcp_sock, tcp_addr):
        """Manages the lifecycle of a single worker connection."""
        dedicated_udp_sock = None
        try:
            dedicated_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Bind it to port 0, which tells the OS to assign any available ephemeral port.
            dedicated_udp_sock.bind((self.host, 0))
            udp_port = dedicated_udp_sock.getsockname()[1]
            client_tcp_sock.sendall(struct.pack("!I", udp_port))

            socks = {"tcp": client_tcp_sock, "udp": dedicated_udp_sock}

            while self.running:
                print(f"[{tcp_addr}] Waiting to receive new gradient data from worker...")
                expected_counter = 1  # Match the counter from worker
                result = mlt.recv_data_mlt(socks, tcp_addr, expected_counter)

                if result is None:
                    print(f"[{tcp_addr}] Worker has disconnected gracefully. Closing thread.")
                    break

                gradients, worker_udp_addr = result

                if gradients:
                    print(f"[{tcp_addr}] Successfully received gradient bundle from {worker_udp_addr}.")

                    # --- NEW: Add gradients to buffer and notify aggregator ---
                    with self.condition:
                        self.received_gradients.append(gradients)
                        print(f"[{tcp_addr}] Added gradients to buffer. Buffer size is now {len(self.received_gradients)}.")
                        if len(self.received_gradients) == self.num_workers:
                            print(f"[{tcp_addr}] All {self.num_workers} gradients received. Notifying aggregator.")
                            self.condition.notify()  # Wake up the aggregator thread
                    # --- END NEW ---
                else:
                    print(f"[{tcp_addr}] Received an empty or invalid gradient bundle.")

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

    def shutdown(self):
        """Shuts down the server and cleans up resources."""
        self.running = False
        self.tcp_server.close()
        # Notify the aggregator so it doesn't wait forever if shutdown happens mid-round
        with self.condition:
            self.condition.notify_all()
        for t in self.worker_threads:
            t.join()
        print("Server shutdown complete.")


if __name__ == "__main__":
    server = MLServer(num_workers=3)
    server.start()
