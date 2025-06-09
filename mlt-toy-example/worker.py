import json
import os
import socket
import struct
import sys
import traceback

import torch

# Make sure the mlt module can be found
# Assuming mlt.py is in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mlt


def tensors_to_lists(tensor_dict: dict) -> dict:
    """Converts a dictionary of tensors to a dictionary of lists for JSON serialization."""
    if not isinstance(tensor_dict, dict):
        return {}
    # Handles tensors and also keeps non-tensor values (like epoch number) as is.
    return {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in tensor_dict.items()}


def run_worker(server_ip: str, server_port: int, gradient_file: str, send_eval_data: bool = False):
    """
    Main function to run the worker/client logic.
    - Connects to the server.
    - Sends gradient data loaded from a file.
    - Receives averaged (echoed) gradients back.
    - Saves the result.
    """
    # Create TCP and UDP sockets
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # --- 1. Connect to server and Load Data ---
        tcp_sock.connect((server_ip, server_port))
        print(f"WORKER: Connected to server at {server_ip}:{server_port}")

        try:
            with open(gradient_file, "r") as f:
                raw_dict = json.load(f)
            # Convert loaded lists back into tensors
            tensor_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw_dict.items()}
            print(f"WORKER: Successfully loaded {len(tensor_dict)} tensors from '{gradient_file}'.")
        except FileNotFoundError:
            print(f"WORKER ERROR: Gradient file not found at '{gradient_file}'. Exiting.")
            return
        except json.JSONDecodeError:
            print(f"WORKER ERROR: Could not decode JSON from '{gradient_file}'. Exiting.")
            return

        # --- 2. SEND PHASE ---
        print("\n--- Sending gradients to server ---")

        # 2.1 Send evaluation signal and data if applicable
        if send_eval_data:
            eval_acc = 0.85  # Example value
            curr_epoch = 5  # Example value
            tcp_sock.sendall(b"E")
            tcp_sock.sendall(struct.pack("!f", eval_acc))
            tcp_sock.sendall(struct.pack("!f", curr_epoch))
            print(f"WORKER: Sent eval signal 'E' with data: acc={eval_acc}, epoch={curr_epoch}")
        else:
            tcp_sock.sendall(b"N")
            print("WORKER: Sent 'no eval' signal 'N'.")

        # 2.2 Send the number of gradients/layers to expect
        num_subgradients = len(tensor_dict)
        tcp_sock.sendall(struct.pack("!I", num_subgradients))
        print(f"WORKER: Notified server to expect {num_subgradients} gradients.")

        # 2.3 Send each gradient using the MLT protocol
        socks = {"tcp": tcp_sock, "udp": udp_sock}
        # FIX: Correctly define the server address for MLT functions
        server_addr_for_mlt = (server_ip, server_port)

        for key, tensor in tensor_dict.items():
            print(f"\nWORKER: Processing '{key}' for sending...")
            # This function sends TCP metadata and returns the raw tensor bytes
            tensor_data_bytes = mlt.serialize_gradient_to_custom_binary(tcp_sock, key, tensor)

            # This function sends the tensor bytes via the MLT UDP protocol
            success = mlt.send_data_mlt(socks, server_addr_for_mlt, tensor_data_bytes)
            if not success:
                print(f"WORKER ERROR: Failed to send tensor data for key '{key}' using MLT. Aborting.")
                return

        # *** CRUCIAL: DO NOT CLOSE SOCKETS HERE. MOVE TO RECEIVE PHASE. ***

        print("\n--- Finished sending. Now waiting for averaged gradients from server ---")

        # --- 3. RECEIVE PHASE ---
        # The same 'socks' dictionary is passed to the receiver function
        avg_gradients = mlt.recv_data_mlt(socks)

        # --- 4. PROCESS AND SAVE RESULTS ---
        if avg_gradients:
            print("\n--- Successfully received averaged gradients from server ---")
            for key, tensor in avg_gradients.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  Received '{key}' with shape {tensor.shape}")
                else:
                    # This handles the eval_acc and epoch keys if they exist
                    print(f"  Received metadata: '{key}' = {tensor}")

            # FIX: Convert tensors to lists before dumping to JSON
            list_based_gradients = tensors_to_lists(avg_gradients)
            output_filename = "echoed_gradient.json"
            with open(output_filename, "w") as f:
                json.dump(list_based_gradients, f, indent=4)
            print(f"\nWORKER: Echoed data successfully saved to '{output_filename}'.")
        else:
            print("WORKER: Failed to receive averaged gradients from server.")

    except ConnectionRefusedError:
        print(f"WORKER ERROR: Connection refused. Is the server running at {server_ip}:{server_port}?")
    except Exception as e:
        print(f"An error occurred in the worker: {e}")
        traceback.print_exc()
    finally:
        print("WORKER: Shutting down.")
        tcp_sock.close()
        udp_sock.close()


if __name__ == "__main__":
    SERVER_IP = "127.0.0.1"
    SERVER_TCP_PORT = 6000
    GRADIENT_JSON_FILE = "gradient.json"  # Assume this file exists

    # Create a dummy gradient.json if it doesn't exist for testing
    if not os.path.exists(GRADIENT_JSON_FILE):
        print(f"Creating dummy '{GRADIENT_JSON_FILE}' for testing...")
        dummy_grads = {
            "conv1": torch.randn(3).tolist(),
            "fc1.weight": torch.randn(10, 5).tolist(),
            "output.bias": torch.zeros(4, dtype=torch.int32).tolist(),
        }
        with open(GRADIENT_JSON_FILE, "w") as f:
            json.dump(dummy_grads, f, indent=4)

    # Run the worker, set send_eval_data to True or False to test both paths
    run_worker(SERVER_IP, SERVER_TCP_PORT, GRADIENT_JSON_FILE, send_eval_data=False)
