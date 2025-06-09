import select
import socket
import struct
import traceback

import numpy as np
import torch

import config
from config import STR_TO_NUMPY_DTYPE, STR_TO_TORCH_DTYPE, TORCH_DTYPE_TO_STR, TORCH_TO_NUMPY_DTYPE


# --- Helper Functions ---
def _recv_all(sock: socket.socket, num_bytes: int) -> bytes | None:
    """Helper function to reliably receive exactly num_bytes from a TCP socket."""
    if num_bytes == 0:
        return b""
    data = bytearray()
    while len(data) < num_bytes:
        try:
            # For a blocking socket, recv will wait until data is available
            packet = sock.recv(num_bytes - len(data))
            if not packet:
                # This indicates the connection was closed gracefully by the peer
                if config.DEBUG:
                    print(f"_recv_all: Connection closed by peer. Expected {num_bytes}, got {len(data)}.")
                return None
            data.extend(packet)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as e:
            if config.DEBUG:
                print(f"_recv_all: Connection error: {e}")
            return None
        except socket.timeout:
            if config.DEBUG:
                print(f"_recv_all: Socket timeout. Expected {num_bytes}, got {len(data)}.")
            return None
    return bytes(data)


def _chunk_data(data_bytes: bytes) -> list[bytes]:
    """Breaks a byte string into fixed-size chunks."""
    return [data_bytes[i : i + config.CHUNK_SIZE] for i in range(0, len(data_bytes), config.CHUNK_SIZE)]


# --- Serialization Function ---
def serialize_gradient_to_custom_binary(tcp_sock: socket.socket, key: str, tensor: torch.Tensor) -> bytes:
    """
    Serializes a key (string) and a tensor (torch.Tensor) into a custom binary format.

    Binary Format Structure (all multi-byte integers are big-endian):
    1.  Key String Length: 4 bytes (unsigned int, >I)
    2.  Key String Bytes: (variable length, UTF-8 encoded)
    3.  Dtype String Length: 2 bytes (unsigned short, >H)
    4.  Dtype String Bytes: (variable length, UTF-8, e.g., "torch.float32")
    5.  Number of Dimensions: 1 byte (unsigned char, >B)
    6.  Shape Dimensions: For each dimension, 4 bytes (unsigned int, >I)
    7.  Tensor Data Length: 8 bytes (unsigned long long, >Q)
    8.  Tensor Data Bytes: (variable length, raw bytes of the tensor)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input 'tensor' must be a torch.Tensor. Got {type(tensor)}")
    if not isinstance(key, str):
        raise TypeError(f"Input 'key' must be a string. Got {type(key)}")

    # Ensure tensor is contiguous for reliable .tobytes() behavior
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # 1. Key serialization
    key_bytes = key.encode("utf-8")
    # !I is unsigned int, 4 bytes
    packed_key_len = struct.pack("!I", len(key_bytes))

    if config.DEBUG:
        print(f"Serializing key: {key} (length: {len(key_bytes)})")

    # 2. Dtype string serialization
    dtype_str = TORCH_DTYPE_TO_STR.get(tensor.dtype)
    if dtype_str is None:
        if config.DEBUG:
            print(f"Warning: Unsupported tensor dtype {tensor.dtype} for serialization. Attempting fallback.")
        # Fallback for dtypes not explicitly in TORCH_DTYPE_TO_STR (e.g. bfloat16 if user adds it)
        # This is a basic fallback; for robust handling of unlisted types, more logic is needed.
        dtype_str = str(tensor.dtype)
        # Consider adding it to STR_TO_TORCH_DTYPE dynamically or raising error if not found later
        if dtype_str not in STR_TO_TORCH_DTYPE:
            STR_TO_TORCH_DTYPE[dtype_str] = tensor.dtype  #  Attempt dynamic addition
        if tensor.dtype not in TORCH_TO_NUMPY_DTYPE:
            # This would be an issue for deserialization if no numpy equivalent is known
            raise ValueError(f"Unsupported tensor dtype for serialization: {tensor.dtype}. No NumPy equivalent mapped.")

    dtype_str_bytes = dtype_str.encode("utf-8")
    # !H is unsigned short, 2 bytes
    packed_dtype_str_len = struct.pack("!H", len(dtype_str_bytes))

    if config.DEBUG:
        print(f"Serializing dtype: {dtype_str} (length: {len(dtype_str_bytes)})")

    # 3. Shape serialization
    shape = tensor.shape
    num_dimensions = len(shape)
    # !B is unsigned char, 1 byte
    packed_num_dimensions = struct.pack("!B", num_dimensions)
    # Pack each dimension
    packed_shape_dims = b"".join(struct.pack("!I", dim) for dim in shape)

    if config.DEBUG:
        print(f"Serializing shape: {shape} (num_dimensions: {num_dimensions})")

    # 4. Tensor data serialization
    # Ensure tensor is on CPU before converting to NumPy array
    # .numpy() on a CUDA tensor raises an error. .cpu() is a no-op if already on CPU.
    # For gradients (param.grad), they usually don't require grad themselves.
    # If serializing a tensor that requires grad and is not a leaf, .detach() might be needed.
    # For gradient values, .cpu() is the primary concern.
    tensor_numpy = tensor.cpu().numpy()
    tensor_data_bytes = tensor_numpy.tobytes()
    packed_tensor_data_len = struct.pack("!Q", len(tensor_data_bytes))

    if config.DEBUG:
        print(f"Serializing tensor data: {len(tensor_data_bytes)} bytes")

    # 5. send everything but the tensor data bytes through TCP
    # tensor_data_bytes is sent separately via MLT protocol
    metadata_bytes = b"".join(
        [
            packed_key_len,
            key_bytes,
            packed_dtype_str_len,
            dtype_str_bytes,
            packed_num_dimensions,
            packed_shape_dims,
            packed_tensor_data_len,
        ]
    )
    tcp_sock.sendall(metadata_bytes)
    if config.DEBUG:
        print(f"TCP Sent: {len(metadata_bytes)} bytes")

    return tensor_data_bytes


def send_data_mlt(socks: dict, server_addr: tuple, gradient_payload_bytes: bytes) -> bool:
    """
    Sends gradient data bytes using the MLT protocol (UDP with TCP-based ACK/retransmission).
    Returns True on success, False on failure.
    """
    tcp_sock = socks["tcp"]
    udp_sock = socks["udp"]
    server_host, server_port_tcp = server_addr
    server_addr_udp = (server_host, server_port_tcp + 1)

    chunks = _chunk_data(gradient_payload_bytes)
    num_chunks = len(chunks)

    try:
        if config.DEBUG:
            print(f"SENDER MLT: Sending num_chunks = {num_chunks} via TCP.")
        tcp_sock.sendall(struct.pack("!I", num_chunks))

        if num_chunks == 0:
            if config.DEBUG:
                print("SENDER MLT: No chunks to send. Probing for server 'Stop' ack.")
            # The loop below will handle sending a probe and waiting for 'S'
            pass

        server_ack_bitmap = bytearray((num_chunks + 7) // 8)
        max_retries_no_progress = 3
        no_progress_rounds = 0

        while True:
            # Send all unacknowledged chunks
            chunks_sent_this_round = 0
            for i in range(num_chunks):
                byte_idx, bit_idx = divmod(i, 8)
                if not ((server_ack_bitmap[byte_idx] >> bit_idx) & 1):
                    chunk_payload = chunks[i]
                    header = struct.pack("!III", i, num_chunks, len(chunk_payload))
                    udp_sock.sendto(header + chunk_payload, server_addr_udp)
                    chunks_sent_this_round += 1
            if config.DEBUG:
                print(f"SENDER MLT: Sent {chunks_sent_this_round} UDP chunks.")

            # Send 'Probe' (P) and wait for 'Stop' (S) or 'Bitmap' (B)
            if config.DEBUG:
                print("SENDER MLT: Sending 'Probe' (P) via TCP.")
            tcp_sock.sendall(b"P")

            signal = _recv_all(tcp_sock, 1)
            if not signal:
                print("SENDER MLT ERROR: Connection closed by receiver while waiting for probe response.")
                return False

            if signal == b"S":
                if config.DEBUG:
                    print("SENDER MLT: Received 'Stop' (S). Transfer complete.")
                return True
            elif signal == b"B":
                if config.DEBUG:
                    print("SENDER MLT: Received 'Bitmap' (B) signal, receiving bitmap.")

                bitmap_len_to_recv = len(server_ack_bitmap)
                new_bitmap_data = _recv_all(tcp_sock, bitmap_len_to_recv)

                if not new_bitmap_data or len(new_bitmap_data) != bitmap_len_to_recv:
                    print("SENDER MLT ERROR: Failed to receive full bitmap from receiver.")
                    return False

                if bytearray(new_bitmap_data) == server_ack_bitmap and chunks_sent_this_round > 0:
                    no_progress_rounds += 1
                    if config.DEBUG:
                        print(f"SENDER MLT: No change in bitmap. Progress stalled ({no_progress_rounds}/{max_retries_no_progress}).")
                else:
                    no_progress_rounds = 0

                server_ack_bitmap = bytearray(new_bitmap_data)

                if no_progress_rounds >= max_retries_no_progress:
                    print("SENDER MLT ERROR: Max retries with no progress reached. Aborting.")
                    return False
            else:
                print(f"SENDER MLT ERROR: Unrecognized signal '{signal}' from receiver.")
                return False
    except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
        print(f"SENDER MLT ERROR: TCP Connection error: {e}")
        return False
    except Exception as e:
        print(f"SENDER MLT ERROR: An unexpected error occurred: {e}")
        traceback.print_exc()
        return False


def recv_data_mlt(socks: dict) -> dict | None:
    """
    Receives gradient data using the MLT protocol.
    Returns a dictionary of reconstructed gradients.
    """
    final_gradients_dict = {}
    tcp_sock = socks["tcp"]
    udp_sock = socks["udp"]

    # ### DRAMATIC CHANGE START ###
    # This entire block was added to create a clear protocol start.
    # The receiver first expects an 'E' (Eval) or 'N' (No Eval) signal,
    # then the number of gradients. This makes the start of the communication
    # much more robust and less ambiguous than just waiting for data.

    # First, receive the eval signal and optional data
    eval_signal = _recv_all(tcp_sock, 1)
    if eval_signal is None:
        if config.DEBUG:
            print("RECEIVER: Did not receive initial eval signal. Connection may be closed.")
        return None

    if eval_signal == b"E":
        eval_acc_bytes = _recv_all(tcp_sock, 4)
        epoch_bytes = _recv_all(tcp_sock, 4)
        if eval_acc_bytes is None or epoch_bytes is None:
            print("RECEIVER ERROR: Failed to receive eval data after 'E' signal.")
            return None
        eval_acc = struct.unpack("!f", eval_acc_bytes)[0]
        curr_epoch = struct.unpack("!f", epoch_bytes)[0]
        final_gradients_dict["eval_acc"] = eval_acc
        final_gradients_dict["epoch"] = curr_epoch
        if config.DEBUG:
            print(f"RECEIVER: Received 'E' signal with eval_acc={eval_acc}, epoch={curr_epoch}")
    elif eval_signal == b"N":
        if config.DEBUG:
            print("RECEIVER: Received 'N' signal (no eval data).")
    else:
        print(f"RECEIVER ERROR: Unrecognized initial signal '{eval_signal}'.")
        return None

    # Next, receive the number of sub-gradients (layers)
    num_subgradients_bytes = _recv_all(tcp_sock, 4)
    if num_subgradients_bytes is None:
        print("RECEIVER ERROR: Failed to receive the number of sub-gradients.")
        return final_gradients_dict if final_gradients_dict else None
    num_subgradients = struct.unpack("!I", num_subgradients_bytes)[0]
    if config.DEBUG:
        print(f"RECEIVER: Expecting to receive {num_subgradients} gradients.")
    # ### DRAMATIC CHANGE END ###

    # Loop to receive each gradient
    for grad_idx in range(num_subgradients):
        if config.DEBUG:
            print(f"\n--- RECEIVER: Starting reception for gradient {grad_idx + 1}/{num_subgradients} ---")

        # --- Receive Metadata for one gradient ---
        # This part is structurally similar to your original code, but uses the robust _recv_all
        packed_key_len_bytes = _recv_all(tcp_sock, 4)
        if packed_key_len_bytes is None:
            break
        key_len = struct.unpack("!I", packed_key_len_bytes)[0]
        key_bytes = _recv_all(tcp_sock, key_len)

        packed_dtype_str_len_bytes = _recv_all(tcp_sock, 2)
        assert packed_dtype_str_len_bytes is not None, "Failed to receive dtype string length"
        dtype_str_len = struct.unpack("!H", packed_dtype_str_len_bytes)[0]
        dtype_str_bytes = _recv_all(tcp_sock, dtype_str_len)

        packed_num_dimensions_bytes = _recv_all(tcp_sock, 1)
        assert packed_num_dimensions_bytes is not None, "Failed to receive number of dimensions"
        num_dimensions = struct.unpack("!B", packed_num_dimensions_bytes)[0]

        shape_list = []
        for _ in range(num_dimensions):
            packed_dim_size = _recv_all(tcp_sock, 4)
            if packed_dim_size is None:
                break
            shape_list.append(struct.unpack("!I", packed_dim_size)[0])
        if len(shape_list) != num_dimensions:
            break
        shape_tuple = tuple(shape_list)

        packed_tensor_data_len_bytes = _recv_all(tcp_sock, 8)
        if packed_tensor_data_len_bytes is None:
            break
        tensor_data_len_expected = struct.unpack("!Q", packed_tensor_data_len_bytes)[0]

        assert key_bytes is not None, "Failed to receive key bytes"
        assert dtype_str_bytes is not None, "Failed to receive dtype string bytes"
        key_str, dtype_str = key_bytes.decode("utf-8"), dtype_str_bytes.decode("utf-8")
        torch_dtype, numpy_dtype = (
            STR_TO_TORCH_DTYPE.get(dtype_str),
            STR_TO_NUMPY_DTYPE.get(dtype_str),
        )

        if not torch_dtype or not numpy_dtype:
            print(f"RECEIVER ERROR: Unsupported dtype '{dtype_str}' for key '{key_str}'. Cannot proceed.")
            return None

        if config.DEBUG:
            print(f"RECEIVER TCP: Metadata OK for key='{key_str}', shape={shape_tuple}, expected_data_len={tensor_data_len_expected}")

        # --- Receive Tensor Chunks via MLT ---
        num_chunks_bytes = _recv_all(tcp_sock, 4)
        if num_chunks_bytes is None:
            break
        total_chunks = struct.unpack("!I", num_chunks_bytes)[0]
        if config.DEBUG:
            print(f"RECEIVER MLT: Expecting {total_chunks} chunks for '{key_str}'.")

        received_chunks = [None] * total_chunks
        bitmap = bytearray((total_chunks + 7) // 8)

        original_udp_timeout = udp_sock.gettimeout()
        udp_sock.settimeout(2.0)

        # ### DRAMATIC CHANGE START ###
        # The logic inside this while loop is clearer. It now explicitly waits for either
        # a UDP packet or a TCP probe using select.select(), making it more responsive
        # and less prone to deadlocks. The early termination logic is also clarified.
        has_stopped = False
        while None in received_chunks:
            try:
                readable, _, _ = select.select([udp_sock, tcp_sock], [], [], 2.0)
                if not readable:
                    print(f"RECEIVER MLT: Timeout waiting for UDP chunks or TCP probe for '{key_str}'.")
                    break

                if tcp_sock in readable:
                    signal = _recv_all(tcp_sock, 1)
                    if signal == b"P":
                        if config.DEBUG:
                            print("RECEIVER MLT: Received 'Probe' (P), sending 'Bitmap' (B).")
                        tcp_sock.sendall(b"B")
                        tcp_sock.sendall(bitmap)
                    elif not signal:
                        print("RECEIVER MLT: Sender closed TCP connection.")
                        break

                if udp_sock in readable:
                    packet, _ = udp_sock.recvfrom(config.CHUNK_SIZE + 12)
                    if len(packet) < 12:
                        continue
                    seq, _, chunk_len_in_header = struct.unpack("!III", packet[:12])
                    if seq < total_chunks and received_chunks[seq] is None and len(packet[12:]) == chunk_len_in_header:
                        received_chunks[seq] = packet[12:]
                        byte_idx, bit_idx = divmod(seq, 8)
                        bitmap[byte_idx] |= 1 << bit_idx

                # Early termination logic. This is a good feature for your specific use case.
                if total_chunks > 0 and (received_chunks.count(None) / total_chunks) <= config.loss_tolerance:
                    if config.DEBUG:
                        print(f"RECEIVER MLT: Loss tolerance ({config.loss_tolerance}) met. Sending 'Stop' (S).")
                    has_stopped = True
                    tcp_sock.sendall(b"S")
                    break
            except (socket.timeout, ConnectionError):
                break

        # if you got out of the loop, you should set the STOP signal
        if not has_stopped:
            if config.DEBUG:
                print("RECEIVER MLT: Got out of chunk send/recv loop but had not sent STOP (S) signal\n", f"Sending it right now for '{key_str}'")
            tcp_sock.sendall(b"S")

        udp_sock.settimeout(original_udp_timeout)
        # ### DRAMATIC CHANGE END ###

        # ### CRITICAL BUG FIX START ###
        # Your original code zero-filled every missing chunk with CHUNK_SIZE bytes.
        # This was incorrect for the final, potentially smaller, chunk.
        # This new logic correctly calculates the size of EACH missing chunk based on
        # the total expected data length, preventing reassembly errors.
        final_data_list = []
        for i in range(total_chunks):
            chunk = received_chunks[i]
            # Calculate the expected size of this specific chunk
            expected_chunk_size = min(config.CHUNK_SIZE, tensor_data_len_expected - (i * config.CHUNK_SIZE))

            if chunk:
                # If we have the chunk, ensure it's not longer than expected
                final_data_list.append(chunk[:expected_chunk_size])
            else:
                # If chunk is missing, append zeros of the *correct* size
                if config.DEBUG:
                    print(f"RECEIVER: Zero-filling missing chunk #{i} with {expected_chunk_size} bytes.")
                final_data_list.append(b"\x00" * expected_chunk_size)

        final_tensor_data_as_bytes = b"".join(final_data_list)
        # Trim to the exact expected length as a final safeguard
        final_tensor_data_as_bytes = final_tensor_data_as_bytes[:tensor_data_len_expected]
        # ### CRITICAL BUG FIX END ###

        # --- Reconstruct the Tensor ---
        try:
            if len(final_tensor_data_as_bytes) != tensor_data_len_expected:
                raise ValueError(f"Final buffer size {len(final_tensor_data_as_bytes)} != expected {tensor_data_len_expected}")

            reconstructed_tensor = torch.zeros(shape_tuple, dtype=torch_dtype)
            if tensor_data_len_expected > 0:
                np_array = np.frombuffer(final_tensor_data_as_bytes, dtype=numpy_dtype)
                # Use .copy() to make the array writable for PyTorch, preventing warnings
                reconstructed_tensor = torch.from_numpy(np_array.copy()).reshape(shape_tuple).to(torch_dtype)

            final_gradients_dict[key_str] = reconstructed_tensor
            if config.DEBUG:
                print(f"RECONSTRUCTION: Success for key '{key_str}'.")
        except Exception as e:
            print(f"RECONSTRUCTION ERROR for '{key_str}': {e}. Storing zeros.")
            final_gradients_dict[key_str] = torch.zeros(shape_tuple, dtype=torch_dtype)

    return final_gradients_dict
