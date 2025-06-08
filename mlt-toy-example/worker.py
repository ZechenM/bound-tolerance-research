import json
import os
import socket
import struct
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlt

with open("gradient.json", "r") as f:
    raw_dict = json.load(f)

# Convert values to tensors
tensor_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_dict.items()}

tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_sock.connect(("127.0.0.1", 6000))

udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sent_eval = False
eval_acc = 0.85  # Example evaluation accuracy
curr_epoch = 5  # Example current epoch
worker_id = 1  # Example worker ID

if sent_eval:
    try:
        tcp_sock.sendall(b"E")
    except Exception as e:
        print(f"Error sending eval signal: {e}")

    # send eval_acc and epoch which are 2 float values: self.eval_acc and self.curr_epoch
    # !f == >f for float, 4 bytes

    eval_acc_bytes = struct.pack("!f", eval_acc)
    epoch_bytes = struct.pack("!f", curr_epoch)
    try:
        tcp_sock.sendall(eval_acc_bytes)
        tcp_sock.sendall(epoch_bytes)
    except Exception as e:
        print(f"Error sending eval_acc and epoch: {e}")
    print(f"Worker {worker_id} sent eval_acc: {eval_acc}, epoch: {curr_epoch}")
else:
    try:
        tcp_sock.sendall(b"N")
    except Exception as e:
        print(f"Error sending no eval signal: {e}")

# Send the gradient data
num_subgradients = len(tensor_dict)
try:
    tcp_sock.sendall(struct.pack("!I", num_subgradients))  # Send number of subgradients
except Exception as e:
    print(f"Error sending number of subgradients: {e}")

socks = {"tcp": tcp_sock, "udp": udp_sock}
server = {"ip": "0.0.0.0", "port": 6000}

for key, tensor in tensor_dict.items():
    tensor_data: bytes = mlt.serialize_gradient_to_custom_binary(tcp_sock, key, tensor)
    mlt.send_data_MLT(socks, server, tensor_data)

avg_gradients = mlt.recv_data_MLT(socks)

# Write to a JSON file
with open("echoed_gradient.json", "w") as f:
    json.dump(avg_gradients, f, indent=4)

print("Toy example completed. Gradient data sent and echoed back.")
