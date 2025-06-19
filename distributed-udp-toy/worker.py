# worker.py
import socket
import struct
import time
import torch
import argparse
import helper
import sys
import os
import json

# Assuming mlt.py is in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlt


class MLWorker:
    def __init__(self, server_host, worker_id, tcp_port=6000):
        self.server_host = server_host
        self.tcp_port = tcp_port
        self.id = worker_id  # Store the worker's ID
        self.tcp_sock = None
        self.udp_sock = None
        self.dedicated_server_udp_port = None

    def connect(self):
        """Establishes connection with the server and gets the dedicated UDP port."""
        try:
            # Connect to the main TCP listening port
            self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_sock.connect((self.server_host, self.tcp_port))
            print(
                f"[Worker {self.id}] Successfully connected to server at {self.server_host}:{self.tcp_port}"
            )

            # 1. After connecting, immediately wait to receive a 4-byte message.
            #    This contains the dedicated UDP port the server has assigned for us.
            port_data = mlt._recv_all(self.tcp_sock, 4)  # Using helper from mlt
            if not port_data:
                raise ValueError(f"[Worker {self.id}] Failed to receive dedicated UDP port from server.")

            # 2. Unpack into an integer to get the port number.
            self.dedicated_server_udp_port = struct.unpack("!I", port_data)[0]
            print(
                f"[Worker {self.id}] Received dedicated server UDP port: {self.dedicated_server_udp_port}"
            )

            # Create the worker's own UDP socket to send from
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return True

        except Exception as e:
            print(f"[Worker {self.id}] Failed to connect to server: {e}")
            return False

    def send_gradients(self, gradients_dict):
        """Sends a dictionary of gradients to the server using the mlt protocol."""
        if not self.tcp_sock or not self.udp_sock:
            print(f"[Worker {self.id}] Not connected. Cannot send gradients.")
            return

        try:
            print(f"\n[Worker {self.id}] --- Starting new gradient transmission ---")

            self.tcp_sock.sendall(b"N")

            num_gradients = len(gradients_dict)
            self.tcp_sock.sendall(struct.pack("!I", num_gradients))

            for key, tensor in gradients_dict.items():
                print(f"[Worker {self.id}] Sending gradient for key: '{key}'")

                payload_bytes = mlt.serialize_gradient_to_custom_binary(
                    self.tcp_sock, key, tensor
                )

                if payload_bytes is None:
                    print(
                        f"[Worker {self.id}] Failed to serialize gradient for key '{key}'. Aborting."
                    )
                    return

                socks = {"tcp": self.tcp_sock, "udp": self.udp_sock}
                addrs = {"udp": (self.server_host, self.dedicated_server_udp_port)}
                addrs["tcp"] = (self.server_host, self.tcp_port)

                success = mlt.send_data_mlt(socks, addrs, payload_bytes)

                if success:
                    print(
                        f"[Worker {self.id}] Successfully completed transmission for key '{key}'."
                    )
                else:
                    print(
                        f"[Worker {self.id}] Failed to transmit data for key '{key}'."
                    )
                    break

            print(f"[Worker {self.id}] --- Finished gradient transmission ---")

        except Exception as e:
            print(f"[Worker {self.id}] An error occurred while sending gradients: {e}")

    def close(self):
        """Closes the worker's sockets."""
        if self.tcp_sock:
            self.tcp_sock.close()
        if self.udp_sock:
            self.udp_sock.close()
        print(f"[Worker {self.id}] Connections closed.")


if __name__ == "__main__":
    # To run this, start server.py first.
    # Then run this worker.py script. You can run it multiple times in
    # different terminals to simulate multiple workers.
    # --- Command-line argument setup ---
    parser = argparse.ArgumentParser(description="Run a distributed ML worker client.")
    parser.add_argument(
        "worker_id",
        type=str,
        help="The unique integer ID for this worker (e.g., 0, 1, or 2).",
    )
    args = parser.parse_args()
    worker_id = int(args.worker_id)

    worker = MLWorker(server_host="127.0.0.1", worker_id=worker_id)
    if worker.connect():
        # Create a dummy dictionary of gradients to send
        dummy_gradients = {
            "layer1.weights": torch.randn(2, 2),
            "layer1.bias": torch.randn(2),
        }
        
        gradients_for_json = helper.tensors_to_lists(dummy_gradients)
        
        json_filename = f"dummy_gradient_{worker_id}.json"
        with open(json_filename, "w") as f:
            json.dump(gradients_for_json, f, indent=4)
        print(f"[Worker {worker_id}] Dummy gradients have been exported to {json_filename}")

        # Send the dummy data
        worker.send_gradients(dummy_gradients)

        # In a real application, you'd have a training loop here.
        # For the demo, we send once and then close.
        time.sleep(2)
        worker.close()
