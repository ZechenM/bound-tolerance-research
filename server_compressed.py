import socket
import pickle
import torch
from threading import Thread
import struct
from typing import Any, Dict, List, Tuple, Set
from config import *
from compression import *

print(f"Compression Method: {compression_method}")

class Server:
    def __init__(self, host="localhost", port=60000, num_workers=3):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.start_server()
        self.run_server()
        # Thread(target=self.handle_user_input, daemon=True).start()

    def start_server(self) -> None:
        # Create a socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_workers)
        self.is_listening = True
        print(f"Server listening on {self.host}:{self.port}...")

    def recv_all(self, conn, size):
        """helper function to receive all data"""
        data = b""
        while len(data) < size:
            packet = conn.recv(size - len(data))
            if not packet:
                return None
            data += packet

        return data

    def close_worker(self):
        """Close all worker connections"""
        for conn in self.connections:
            conn.close()
        self.connections = []
        self.conn_addr_map = {}
        print("Closed all worker connections.")

    def worker_connection_handler(self) -> None:
        """Wait for all workers to connect"""
        self.connections = []
        self.conn_addr_map = {}
        for _ in range(self.num_workers):
            conn, addr = self.server_socket.accept()
            self.connections.append(conn)
            self.conn_addr_map[conn] = addr
            print(f"Connected to worker at {addr}")

        print("All workers connected.")

    def recv_send(self):
        """Receive gradients from all workers and send back averaged gradients"""
        gradients = []
        for conn in self.connections:
            # Receive the size of the incoming data
            size_data = self.recv_all(conn, 4)
            if not size_data:
                print("Failed to receive data size.")
                continue
            size = struct.unpack("!I", size_data)[0]

            # Receive the actual data
            data = self.recv_all(conn, size)
            if not data:
                print("Failed to receive data.")
                continue

            # response with ACK
            conn.sendall(b'A')

            compressed_grad = pickle.loads(data)
            grad = decompress(compressed_grad)
            gradients.append(grad)
            print(f"Received gradients from worker {self.conn_addr_map[conn]}")

        # Received gradients from all workers
        print("All gradients received.")

        avg_gradients = {}
        for key in gradients[0].keys():
            avg_gradients[key] = torch.stack([grad[key].float() for grad in gradients]).mean(
                dim=0
            )

        # Compress the averaged gradients
        compressed_avg_gradients = compress(avg_gradients)
        avg_gradients_data = pickle.dumps(compressed_avg_gradients)

        # Send averaged gradients back to all workers
        for conn in self.connections:
            # Send the size of the data first
            conn.sendall(struct.pack("!I", len(avg_gradients_data)))
            # Sendall the actual data
            conn.sendall(avg_gradients_data)
            print(f"Sent averaged gradients to worker {self.conn_addr_map[conn]}")

    def run_server(self) -> None:
        while self.is_listening:
            # server accepts connections from all the workers
            self.worker_connection_handler()

            # server receives gradients from all workers and sends back averaged gradients
            self.recv_send()

            # close all worker connections
            self.close_worker()

    def print_menu_options(self) -> None:
        print("Enter 'q' to close this server.")

    # close the server
    def close_server(self) -> None:
        """Close the server"""
        self.is_listening = False
        self.server_socket.close()
        print("Server closed.")

    def handle_user_input(self) -> None:
        self.print_menu_options()
        user_input = input()
        if len(user_input) == 0:
            print("Invalid option.")
            return
        user_input = user_input[0].lower() + user_input[1:]
        # case: close the server
        if user_input[0] == "q":
            self.close_server()
            return


if __name__ == "__main__":
    server = Server()
