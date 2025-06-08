import os
import socket
import struct
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlt

# TCP socket
tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_sock.bind(("0.0.0.0", 6000))
tcp_sock.listen(1)
print("Server listening on TCP port 6000...")

# UDP socket
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_sock.bind(("0.0.0.0", 6001))
print("Server listening on UDP port 6001...")

conn, addr = tcp_sock.accept()
print(f"Connection established with {addr}")

socks = {"tcp": conn, "udp": udp_sock}
grad = mlt.recv_data_MLT(socks)

if grad is None:
    print(f"Failed to receive gradient data from worker {addr}.")

# since this toy example only has one worker
# so we are just echoing
avg_gradients = grad if isinstance(grad, dict) else {}
ip, port = addr
receiver = {"ip": ip, "port": port}

try:
    tcp_sock.sendall(b"N")
except Exception as e:
    print(f"Error sending no eval signal: {e}")

num_subgradients = len(avg_gradients)
try:
    tcp_sock.sendall(struct.pack("!I", num_subgradients))  # Send number of subgradients
except Exception as e:
    print(f"Error sending number of subgradients: {e}")

for key, tensor in avg_gradients.items():
    assert isinstance(tensor, torch.Tensor), "Expected tensor to be a torch.Tensor"
    averaged_tensor_data = mlt.serialize_gradient_to_custom_binary(tcp_sock, key, tensor)
    mlt.send_data_MLT(socks, receiver, averaged_tensor_data)
