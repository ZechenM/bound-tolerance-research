import os
import socket
import struct
import sys

import torch

# Make sure the mlt module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlt  # Assuming this is the name of your file from the Canvas


# --- Main Server Logic ---
def run_server():
    # TCP socket for listening
    listening_tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_tcp_sock.bind(("0.0.0.0", 6000))
    listening_tcp_sock.listen(1)
    print("Server listening on TCP port 6000...")

    # UDP socket for data transfer
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(("0.0.0.0", 6001))
    print("Server listening on UDP port 6001...")

    # --- Main Loop to Handle One Client ---
    # This server handles one client, then exits. For multiple clients,
    # you would wrap this accept() call in another loop.
    try:
        # IMPORTANT: accept() creates a NEW socket for the connection
        # tcp_addr == worker connected tcp_addr
        conn, tcp_addr = listening_tcp_sock.accept()
        print(f"Connection established with {tcp_addr}")

        with conn:  # Use 'with' statement to ensure the connection socket is closed
            socks = {"tcp": conn, "udp": udp_sock}
            addrs = {}

            # --- RECEIVE PHASE ---
            received_tuple = mlt.recv_data_mlt(socks)

            if received_tuple is None:
                raise ValueError(f"Failed to receive gradient data from worker {tcp_addr}.")

            # udp addr is a tuple of udp_host, udp_port
            received_gradients, udp_addr = received_tuple
            addrs["udp"] = udp_addr

            # In a real scenario, you'd wait for gradients from all workers here.
            # For this toy example, we just "average" by echoing.
            avg_gradients = received_gradients if isinstance(received_gradients, dict) else {}

            # --- SEND-BACK PHASE ---
            # The client MUST be waiting for this data on the same connection.

            # Use the CONNECTION socket `conn`, not the listening socket `listening_tcp_sock`
            # And no need for a separate `receiver` dict if sending to the same client.

            print("\n--- Send (Echoing) gradients back to worker ---")

            # 1. Send 'N' signal (No more evaluation results to send)
            # Use the connection socket 'conn'
            try:
                conn.sendall(b"N")
                print("SERVER: Sent 'no eval' singal 'N'.")
            except Exception as e:
                print(f"Error sending 'N' signal: {e}")

            # 2. Send the number of tensors (layers) in the dictionary
            # Use the connection socket 'conn'
            num_subgradients = len(avg_gradients) - 2 if "epoch" in avg_gradients else len(avg_gradients)
            try:
                conn.sendall(struct.pack("!I", num_subgradients))
                print(f"SERVER: Notified server to expect {num_subgradients} gradients.")
            except Exception as e:
                print(f"Error sending number of subgradients: {e}")

            # 3. Iterate and send each tensor's metadata (TCP) and data (UDP)
            for key, tensor in avg_gradients.items():
                if not isinstance(tensor, torch.Tensor):
                    print(f"Warning: Item with key '{key}' is not a tensor, skipping.")
                    print(f"    Value is likely eval data: {tensor}")
                    continue

                # This function now sends metadata on 'conn' and returns the tensor bytes
                averaged_tensor_data = mlt.serialize_gradient_to_custom_binary(conn, key, tensor)
                if averaged_tensor_data is None:
                    print(f"Failed to serialize tensor data for key '{key}'. Skipping.")
                    continue

                # This function sends the tensor bytes using the MLT UDP protocol
                # The server address here is the client's address
                addrs["tcp"] = tcp_addr
                success = mlt.send_data_mlt(socks, addrs, averaged_tensor_data)
                if not success:
                    raise ValueError(f"SERVER ERROR: Failed to send tensor data for key '{key}' using MLT. Aborting.")
                else:
                    print(f"\n--- SERVER successfully sent all the tensor data for key '{key}' ---\n")

        print(f"\nFinished echo transaction with TCP:{tcp_addr} and UDP:{udp_addr}. Closing connection.")

    except Exception as e:
        print(f"An error occurred in the main server loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Server shutting down.")
        listening_tcp_sock.close()
        udp_sock.close()


if __name__ == "__main__":
    run_server()
