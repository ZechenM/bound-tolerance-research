# worker.py
import argparse  # Import for command-line argument parsing
import json  # Import the json library for file export
import os
import socket
import struct
import sys
import time

import helper
import torch

# Assuming mlt.py is in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlt


class MLWorker:
    def __init__(self, server_host, worker_id, tcp_port=9999):
        self.server_host = server_host
        self.tcp_port = tcp_port
        self.id = worker_id  # Store the worker's ID
        self.tcp_sock = None
        self.udp_sock = None
        self.dedicated_server_udp_port = None
        self.running = True
        self.round = 0

    def connect(self):
        """Establishes connection with the server and gets the dedicated UDP port."""
        try:
            self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_sock.connect((self.server_host, self.tcp_port))
            print(f"[Worker {self.id}] Successfully connected to server at {self.server_host}:{self.tcp_port}")

            port_data = mlt._recv_all(self.tcp_sock, 4)
            if not port_data:
                print(f"[Worker {self.id}] Failed to receive dedicated UDP port from server.")
                return False

            self.dedicated_server_udp_port = struct.unpack("!I", port_data)[0]
            print(f"[Worker {self.id}] Received dedicated server UDP port: {self.dedicated_server_udp_port}")

            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return True

        except Exception as e:
            print(f"[Worker {self.id}] Failed to connect to server: {e}")
            return False

    def start_training_loop(self):
        """Runs the main training loop of sending local and receiving global gradients."""
        while self.running:
            try:
                print(f"\n[Worker {self.id}] --- Starting training round {self.round} ---")

                # 1. Create and send local gradients
                local_gradients = {
                    "layer1.weights": torch.randn(2, 2),
                    "layer1.bias": torch.randn(2),
                }

                helper.write_to_json(self.id, helper.tensors_to_lists(local_gradients))

                self.send_gradients(local_gradients)

                # --- ROUND-TRIP LOGIC: Receive averaged gradients ---
                # 2. Wait to receive the globally averaged gradients from the server
                print(f"[Worker {self.id}] Gradients sent. Waiting to receive averaged model back...")
                socks = {"tcp": self.tcp_sock, "udp": self.udp_sock}
                result = mlt.recv_data_mlt(socks)

                if result is None:
                    print(f"[Worker {self.id}] Server disconnected. Shutting down.")
                    self.running = False
                    continue

                averaged_gradients, _ = result
                print(f"[Worker {self.id}] Successfully received averaged gradients.")

                if averaged_gradients is None:
                    raise ValueError(f"[WORKER {self.id}] failed to receive averaged gradients back from the server")

                # 3. Save the received gradients to a JSON file
                json_filename = f"worker_{self.id}_received_averaged_round_{self.round}.json"
                with open(json_filename, "w") as f:
                    json.dump(
                        {k: v.tolist() for k, v in averaged_gradients.items()},
                        f,
                        indent=4,
                    )
                print(f"[Worker {self.id}] Saved averaged gradients to {json_filename}")
                # --- END ---

                self.round += 1

                if self.round == 1:
                    self.running = False
                    break
                time.sleep(2)  # Simulate training time

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"[Worker {self.id}] An error occurred in the training loop: {e}")
                self.running = False

    def send_gradients(self, gradients_dict):
        """Sends a dictionary of gradients to the server using the mlt protocol."""
        if not self.tcp_sock or not self.udp_sock:
            print(f"[Worker {self.id}] Not connected. Cannot send gradients.")
            return

        print(f"[Worker {self.id}] Starting to send {len(gradients_dict)} gradients...")
        self.tcp_sock.sendall(b"N")
        self.tcp_sock.sendall(struct.pack("!I", len(gradients_dict)))

        for key, tensor in gradients_dict.items():
            payload_bytes = mlt.serialize_gradient_to_custom_binary(self.tcp_sock, key, tensor)
            if payload_bytes is not None:
                socks = {"tcp": self.tcp_sock, "udp": self.udp_sock}
                addrs = {"udp": (self.server_host, self.dedicated_server_udp_port)}
                addrs["tcp"] = (self.server_host, self.tcp_port)

                success = mlt.send_data_mlt(socks, addrs, payload_bytes)

                if success:
                    print(f"[Worker {self.id}] Successfully completed transmission for key '{key}'.")
                else:
                    print(f"[Worker {self.id}] Failed to transmit data for key '{key}'.")
                    break

        print(f"[Worker {self.id}] Finished sending gradients.")

    def close(self):
        """Closes the worker's sockets."""
        if self.tcp_sock:
            self.tcp_sock.close()
        if self.udp_sock:
            self.udp_sock.close()
        print(f"[Worker {self.id}] Connections closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a distributed ML worker client.")
    parser.add_argument(
        "worker_id",
        type=int,
        help="The unique integer ID for this worker (e.g., 0, 1, or 2).",
    )
    args = parser.parse_args()

    worker = MLWorker(server_host="127.0.0.1", worker_id=args.worker_id)
    if worker.connect():
        worker.start_training_loop()
        worker.close()
