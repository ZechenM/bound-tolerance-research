import select
import socket
import struct

import numpy as np
import torch

import config
from config import DEBUG, STR_TO_NUMPY_DTYPE, STR_TO_TORCH_DTYPE, TORCH_DTYPE_TO_STR, TORCH_TO_NUMPY_DTYPE, chunk_size


# Helper function
def recv_all(conn, size):
    """helper function to receive all data"""
    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data += packet

    return data


# Helper function
def chunk_gradient(data_bytes: bytes) -> list[bytes]:
    """Serialize gradient and break into chunks"""
    return [data_bytes[i : i + chunk_size] for i in range(0, len(data_bytes), chunk_size)]


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

    # 2. Dtype string serialization
    dtype_str = TORCH_DTYPE_TO_STR.get(tensor.dtype)
    if dtype_str is None:
        if DEBUG:
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

    # 3. Shape serialization
    shape = tensor.shape
    num_dimensions = len(shape)
    # !B is unsigned char, 1 byte
    packed_num_dimensions = struct.pack("!B", num_dimensions)
    # Pack each dimension
    packed_shape_dims = b"".join(struct.pack("!I", dim) for dim in shape)

    # 4. Tensor data serialization
    # Ensure tensor is on CPU before converting to NumPy array
    # .numpy() on a CUDA tensor raises an error. .cpu() is a no-op if already on CPU.
    # For gradients (param.grad), they usually don't require grad themselves.
    # If serializing a tensor that requires grad and is not a leaf, .detach() might be needed.
    # For gradient values, .cpu() is the primary concern.
    tensor_numpy = tensor.cpu().numpy()
    tensor_data_bytes = tensor_numpy.tobytes()
    packed_tensor_data_len = struct.pack("!Q", len(tensor_data_bytes))

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

    return tensor_data_bytes


def send_data_MLT(socks: dict, server: dict, gradient_payload_bytes: bytes) -> bool:
    """
    Sends gradient data using the MLT protocol.
    Returns True on success, False on failure.
    """
    # --- Phase 0: extracting important metadata and preprocess tensor data (gradient_payload_bytes) ---
    tcp_sock = socks["tcp"]
    if not isinstance(tcp_sock, socket.socket):
        raise TypeError(f"Expected 'tcp_sock' to be a socket.socket, got {type(tcp_sock)}")
    udp_sock = socks["udp"]
    if not isinstance(udp_sock, socket.socket):
        raise TypeError(f"Expected 'udp_sock' to be a socket.socket, got {type(udp_sock)}")

    server_host = str(server["ip"])
    server_port = int(server["port"])

    chunks = chunk_gradient(gradient_payload_bytes)
    num_chunks = len(chunks)

    try:
        # Send the total number of chunks for this gradient via TCP
        if DEBUG:
            print(f"MLT: Sending num_chunks = {num_chunks} via TCP.")
        tcp_sock.sendall(struct.pack("!I", num_chunks))

        if num_chunks == 0:
            print("MLT: No chunks to send. Probing for server acknowledgement.")
            # Even with 0 chunks, we need to "probe" to get the "Stop" signal.
            # The server needs to handle num_chunks=0 gracefully.
            # (The loop below will send 'P' and expect 'S').

        # Local bitmap representing chunks acknowledged by the server.
        # Initially, all are 0 (server has not acknowledged any).
        server_ack_bitmap = bytearray((num_chunks + 7) // 8)
        max_retries_no_progress = 3  # Number of rounds with no new acks before aborting
        no_progress_rounds = 0

        while True:  # Retransmission loop
            # --- Phase 1: Opportunistic check for early "Stop" from server ---
            # This is useful if server wants to terminate this stream early.
            # Using a very short timeout for a non-blocking check.
            ready_to_read, _, _ = select.select([tcp_sock], [], [], 0.001)
            if tcp_sock in ready_to_read:
                signal = recv_all(tcp_sock, 1)  # Read just 1 byte non-blockingly (due to select)
                if not signal:  # Connection closed
                    print("MLT: TCP connection closed by server (early check).")
                    return False
                if signal == b"S":
                    print("MLT: Received early 'Stop' (S) from server. Transmission for this gradient complete.")
                    return True
                else:
                    # This is unexpected if server only sends S/B in response to P.
                    # Could log or handle as an error. For now, we might proceed,
                    # but it could indicate a de-sync.
                    print(f"MLT: Warning - Unexpected TCP data '{signal}' during early check. Proceeding with caution.")

            # --- Phase 2: Send/Resend UDP chunks based on server_ack_bitmap ---
            chunks_sent_this_round = 0
            for i in range(num_chunks):
                byte_idx, bit_idx = divmod(i, 8)
                # Check if the i-th bit in server_ack_bitmap is 0 (server hasn't acked it)
                if not ((server_ack_bitmap[byte_idx] >> bit_idx) & 1):
                    chunk_payload = chunks[i]
                    # Header: chunk_id (0-indexed), total_chunks, payload_length_of_this_chunk
                    header = struct.pack("!III", i, num_chunks, len(chunk_payload))
                    packet_to_send = header + chunk_payload
                    try:
                        # TODO: is (self.server_host, self.server_port + 1) correct?
                        udp_sock.sendto(
                            packet_to_send,
                            (server_host, server_port + 1),
                        )
                        chunks_sent_this_round += 1
                    except Exception as e:
                        # Log UDP send error but continue; rely on bitmap retransmission
                        print(f"MLT: UDP sendto error for chunk {i}: {e}")
            if DEBUG:
                print(f"MLT: Sent {chunks_sent_this_round} UDP chunks this round.")

            # --- Phase 3: Send "Probe" (P) signal via TCP ---
            if DEBUG:
                print("MLT: Sending 'Probe' (P) signal via TCP.")
            try:
                tcp_sock.sendall(b"P")
            except Exception as e:
                print(f"MLT: Failed to send 'Probe' (P) signal: {e}")
                return False

            # --- Phase 4: Receive server's response (S or B + bitmap) ---
            if DEBUG:
                print("MLT: Waiting for server response to probe...")
            # Use select with a reasonable timeout for the server to respond
            probe_response_timeout = 5.0  # seconds
            ready_to_read, _, _ = select.select([tcp_sock], [], [], probe_response_timeout)

            if not ready_to_read:
                print(f"MLT: Timeout ({probe_response_timeout}s) waiting for server response to 'Probe'.")
                no_progress_rounds += 1
                if no_progress_rounds >= max_retries_no_progress:
                    print("MLT: Max retries with no progress reached. Aborting.")
                    return False
                continue  # Retry by sending probe again after resending unacked chunks

            signal = recv_all(tcp_sock, 1)
            if not signal:  # Connection closed or recv_all_tcp failed
                print("MLT: Failed to receive signal from server or connection closed after probe.")
                return False

            if signal == b"S":  # "Stop" signal
                print("MLT: Received 'Stop' (S). Transmission for this gradient complete.")
                return True
            elif signal == b"B":  # "Bitmap" signal
                print("MLT: Received 'Bitmap' (B)")

                new_bitmap_data = recv_all(tcp_sock, len(server_ack_bitmap))
                if not new_bitmap_data:
                    print("MLT: Failed to receive bitmap data from server.")
                    return False
                # Check if bitmap indicates progress
                if bytearray(new_bitmap_data) == server_ack_bitmap and chunks_sent_this_round > 0:
                    # We sent chunks, but the bitmap didn't change.
                    no_progress_rounds += 1
                    print(f"MLT: No change in bitmap after sending chunks. Progress stalled ({no_progress_rounds}/{max_retries_no_progress}).")
                else:
                    no_progress_rounds = 0  # Progress was made

                server_ack_bitmap = bytearray(new_bitmap_data)
                if no_progress_rounds >= max_retries_no_progress:
                    print(f"MLT: Max retries ({max_retries_no_progress}) with no progress in bitmap. Aborting.")
                    return False

                # Check if all chunks are now acknowledged (optional optimization here,
                # as the loop condition and server sending 'S' is the primary completion mechanism)
                if num_chunks > 0 and all((server_ack_bitmap[i // 8] >> (i % 8)) & 1 for i in range(num_chunks)):
                    print("MLT: All chunks appear to be acknowledged by bitmap. Next probe should yield 'S'.")
                    # We still send 'P' and let server confirm with 'S'.
            else:
                print(f"MLT: Unrecognized signal '{signal}' from server.")
                return False
    except ConnectionRefusedError:
        print("MLT: TCP Connection refused by the server.")
        return False
    except BrokenPipeError:
        print("MLT: TCP Connection broken (e.g., server closed unexpectedly).")
        return False
    except Exception as e:
        print(f"MLT: An unexpected error occurred in send_data_MLT: {e}")
        return False
    finally:
        # Assuming tcp_sock was blocking and we didn't change its global state.
        # If you had tcp_sock.setblocking(False) at the start of the function,
        # you'd restore it here: tcp_sock.setblocking(True)
        pass


def recv_data_MLT(socks: dict) -> dict[str, torch.Tensor | None] | None:
    tcp_sock = socks.get("tcp")
    udp_sock = socks.get("udp")

    if not tcp_sock or not udp_sock:
        print("MLT: Missing TCP or UDP socket.")
        return None
    if not isinstance(tcp_sock, socket.socket):
        raise TypeError(f"Expected 'tcp_sock' to be a socket.socket, got {type(tcp_sock)}")
    if not isinstance(udp_sock, socket.socket):
        raise TypeError(f"Expected 'udp_sock' to be a socket.socket, got {type(udp_sock)}")

    # for each worker, all the important metadata will always be received first through TCP
    num_sub_gradients = recv_all(tcp_sock, 4)
    if not num_sub_gradients:
        return None
    # num_sub_gradients is interchangable as layers
    # we are sending gradients layer by layer, so key-val pair will be reconstructed iteration by iteration
    num_sub_gradients = struct.unpack("!I", num_sub_gradients)[0]
    final_gradients_dict: dict[str, torch.Tensor | None] = {}

    # for each layer, we will receive the key, dtype, shape, and tensor
    for _ in range(num_sub_gradients):
        # 1. key deserialization
        # 1.1. receive the length of packed value and UNPACK it
        packed_key_len = recv_all(tcp_sock, 4)
        if not packed_key_len:
            return None
        key_len = struct.unpack("!I", packed_key_len)[0]

        # 1.2. receive the actual value and DECODE it
        key_bytes = recv_all(tcp_sock, key_len)
        if not key_bytes:
            return None
        key_str = key_bytes.decode("utf-8")
        # Initialize with None
        # to be filled out during UDP transmission
        final_gradients_dict[key_str] = None

        # 2. dtype string deserialization
        # 2.1. ...
        packed_dtype_str_len = recv_all(tcp_sock, 2)
        if not packed_dtype_str_len:
            return None
        dtype_str_len = struct.unpack("!H", packed_dtype_str_len)[0]

        # 2.2. ...
        dtype_str_bytes = recv_all(tcp_sock, dtype_str_len)
        if not dtype_str_bytes:
            return None
        dtype_str = dtype_str_bytes.decode("utf-8")

        torch_dtype = STR_TO_TORCH_DTYPE.get(dtype_str, None)
        numpy_dtype = STR_TO_NUMPY_DTYPE.get(dtype_str, None)
        if not torch_dtype or not numpy_dtype:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        # 3. shape deserialization
        # 3.1
        packed_num_dimensions = recv_all(tcp_sock, 1)
        if not packed_num_dimensions:
            return None
        num_dimensions = struct.unpack("!B", packed_num_dimensions)[0]
        # 3.2
        shape_list = []
        shape_read_success = True
        for i in range(num_dimensions):
            packed_dim_size_bytes = recv_all(tcp_sock, 4)
            if not packed_dim_size_bytes:
                shape_read_success = False
                break
            dim_size = struct.unpack("!I", packed_dim_size_bytes)[0]
            shape_list.append(dim_size)
        if not shape_read_success:
            raise ValueError("Failed to read shape dimensions")
        shape_tuple = tuple(shape_list)

        print(f"Tensor Shape: {shape_list}")

        # 4. tensor data length deserialization
        packed_tensor_data_len = recv_all(tcp_sock, 8)
        if not packed_tensor_data_len:
            return None
        tensor_data_len_expected = struct.unpack("!Q", packed_tensor_data_len)[0]

        # 5. Prepare to receive the tensor data
        #  UDP WILL START SOON
        size_data = recv_all(tcp_sock, 4)
        if not size_data:
            return None
        total_chunks = struct.unpack("!I", size_data)[0]

        # Initialize storage and bitmap
        received_chunks = [None] * total_chunks
        bitmap = bytearray((total_chunks + 7) // 8)  # 1 bit per chunk
        expected_packet_size = config.chunk_size + 12  # 12-byte header
        socket_timeout = 2.0  # Adjust based on network conditions

        # Set socket timeout
        udp_sock.settimeout(socket_timeout)
        tcp_sock.setblocking(False)  # Make TCP non-blocking
        while None in received_chunks:
            try:
                readable, _, _ = select.select([udp_sock, tcp_sock], [], [], 0.001)
                if tcp_sock in readable:
                    signal = tcp_sock.recv(1)
                    if signal == b"P":
                        tcp_sock.sendall(b"B")
                        tcp_sock.sendall(bitmap)
                        if config.DEBUG:
                            print(f"bitmap: {bitmap}")
                    else:
                        print(f"recv_data_MLT: cannot recognize signal from server:{signal}")

                if udp_sock in readable:
                    # Receive packet with extra buffer space
                    packet, _ = udp_sock.recvfrom(expected_packet_size + 100)
                    if config.DEBUG:
                        print("received packets")
                    # Verify minimum packet size
                    if len(packet) < 12:
                        print(f"Packet too small: {len(packet)} bytes")
                        continue

                    # Unpack header (seq, total_chunks, chunk_size)
                    seq, chunk_count, chunk_size = struct.unpack("!III", packet[:12])

                    # Validate sequence number
                    if seq >= total_chunks:
                        print(f"Invalid sequence number: {seq}")
                        continue

                    # Verify payload size matches header
                    if len(packet[12:]) != chunk_size:
                        print(f"Payload size mismatch: expected {chunk_size}, got {len(packet[12:])}")
                        continue

                    # Store valid chunk and update bitmap
                    received_chunks[seq] = packet[12:]
                    byte_index = seq // 8
                    bit_index = seq % 8
                    bitmap[byte_index] = bitmap[byte_index] | (1 << bit_index)

                    # early termination: loss within boundary
                    missing_rate = 1 - (received_chunks.count(None) / total_chunks)
                    if missing_rate < config.loss_tolerance:
                        tcp_sock.sendall(b"S")
                        if config.DEBUG:
                            print("recv_data_MLT: early termination")
                        break

            except socket.timeout:
                print("Timeout waiting for packets")
                break
            except Exception as e:
                print(f"Error receiving packet: {e}")
                break

        # Reset socket timeout
        udp_sock.settimeout(None)
        tcp_sock.setblocking(True)

        # if chunk not received, fill with 0
        for i, chunk in enumerate(received_chunks):
            if not chunk:
                if config.DEBUG:
                    print("fill with zeros")
                received_chunks[i] = bytes(config.chunk_size)

        # DESERIALIZATION CONTINUED
        # 6. Reconstruct original data
        # this variable below is the data we created by receiving the chunks
        final_tensor_data_as_bytes = b"".join(received_chunks)
        try:
            num_elements_in_shape = np.prod(shape_tuple) if shape_tuple else 0
            bytes_expected_by_shape_dtype = num_elements_in_shape * np.dtype(numpy_dtype).itemsize if num_elements_in_shape > 0 else 0

            if tensor_data_len_expected == 0 and num_elements_in_shape == 0:
                reconstructed_tensor = torch.empty(shape_tuple, dtype=torch_dtype)
                print(f"Reconstruction: Empty tensor for '{key_str}'.")
            elif tensor_data_len_expected != bytes_expected_by_shape_dtype:
                print(
                    f"Reconstruction WARNING for '{key_str}': TCP expected_data_len ({tensor_data_len_expected}) "
                    f"mismatches bytes for shape*dtype ({bytes_expected_by_shape_dtype}). "
                    f"Attempting reconstruction with received buffer of size {len(final_tensor_data_as_bytes)}."
                )
                # If tensor_data_len_expected is not a multiple of itemsize, np.frombuffer might behave unexpectedly
                # or only read up to the last full element.
                # It's safer to ensure the buffer used matches bytes_expected_by_shape_dtype if possible,
                # or handle the discrepancy by creating zeros if reconstruction is impossible.
                if len(final_tensor_data_as_bytes) >= bytes_expected_by_shape_dtype:
                    # Use only the part of the buffer that corresponds to the shape
                    buffer_for_reconstruction = final_tensor_data_as_bytes[:bytes_expected_by_shape_dtype]
                    np_array = np.frombuffer(buffer_for_reconstruction, dtype=numpy_dtype)
                    if np_array.size == num_elements_in_shape:  # Check if number of elements is correct
                        np_array = np_array.reshape(shape_tuple)
                        reconstructed_tensor = torch.from_numpy(np_array).to(torch_dtype)
                    else:  # Should not happen if buffer_for_reconstruction was sized correctly
                        print(f"  Reconstruction ERROR for '{key_str}' (mismatch case): Element count mismatch. Creating zero tensor.")
                        reconstructed_tensor = torch.zeros(shape_tuple, dtype=torch_dtype)
                else:  # Not enough data in the buffer (even with zero-filling) for the shape
                    print(
                        f"  Reconstruction ERROR for '{key_str}' (mismatch case): Not enough data in buffer ({len(final_tensor_data_as_bytes)}) for shape ({bytes_expected_by_shape_dtype} bytes needed). Creating zero tensor."
                    )
                    reconstructed_tensor = torch.zeros(shape_tuple, dtype=torch_dtype)
            else:  # tensor_data_len_expected == bytes_expected_by_shape_dtype
                # hopefully every time we fall into this case
                np_array = np.frombuffer(final_tensor_data_as_bytes, dtype=numpy_dtype)
                np_array = np_array.reshape(shape_tuple)
                reconstructed_tensor: torch.Tensor = torch.from_numpy(np_array).to(torch_dtype)
                print(f"  Reconstruction: Tensor reconstructed for '{key_str}'.")

            final_gradients_dict[key_str] = reconstructed_tensor

        except ValueError as ve:
            print(f"  Reconstruction ERROR for '{key_str}': ValueError (likely reshape failed due to size mismatch) - {ve}. Creating zero tensor.")
            final_gradients_dict[key_str] = torch.zeros(shape_tuple, dtype=torch_dtype)
        except Exception as e:
            print(f"  Reconstruction ERROR for '{key_str}': Unexpected error - {e}. Skipping this gradient from dict.")
            # Decide if to break or continue for other gradients
            # break # Safer to break if unexpected reconstruction error occurs
            continue  # Or try to process next gradient if error is isolated

    print(f"\nReceiver: Finished processing loop. Total gradients in dictionary: {len(final_gradients_dict)}")
    return final_gradients_dict
