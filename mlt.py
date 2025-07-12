import select
import socket
import struct
import traceback

import numpy as np
import torch

import config
from config import STR_TO_NUMPY_DTYPE, STR_TO_TORCH_DTYPE, TORCH_DTYPE_TO_STR, TORCH_TO_NUMPY_DTYPE

# # --- Helper Functions ---
# def _recv_with_retry(sock: socket.socket, size: int, retries: int = 5) -> bytes | None:
#     sock.settimeout(1.0)  # Set a timeout for the receive operation
#     for attempt in range(retries):
#         try:
#             data = _recv_all(sock, size)
#             if data is not None:
#                 return data
#         except socket.timeout:
#             print(f"Socket timeout occurred on attempt {attempt}. Retrying...")
#             if attempt == retries - 1:
#                 raise TimeoutError("Max retries reached. Giving up.")
#             time.sleep(1)  # Wait before retrying
#         except Exception as e:
#             raise RuntimeError(f"Error on attempt {attempt}: {e}")
#     return None

# TODO: timeout might also be 0.001 or 0.01 if we don't wany to stall the program that much
def _flush_recv_buffer(sock: socket.socket, timeout=0.1):
    all_data = b""
    
    while True:
        ready_to_read, _, _ = select.select([sock], [], [], timeout)
        if not ready_to_read:
            break
        
        try:
            data = sock.recv(1024)
            if not data:
                print("Connection closed by the sender.")
                break
            all_data += data
        except socket.error as e:
            print(f"Error while flushing receive buffer: {e}")
            break

    return all_data



def _recv_all(conn: socket.socket, size: int, recv_lock=None) -> bytes | None:
    """Helper function to reliably receive exactly num_bytes from a TCP socket."""
    # conn.settimeout(60.0)  # Set a timeout for the receive operation
    # conn.setblocking(False)  # Ensure the socket is in blocking mode
    data = b""

    if recv_lock:
        recv_lock.acquire()
    # -------- Critical Section: Ensure thread-safe receiving --------
    while len(data) < size:
        try:
            # Use a lock to ensure thread-safe receiving
            packet = conn.recv(size - len(data))
            if not packet:
                print("Connection closed by the sender.")
                return None
        # except socket.timeout:
        #     print("Receive operation timed out.")
        #     return None
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None

        data += packet
    # -------- End of Critical Section --------
    if recv_lock:
        recv_lock.release()  # Release the lock after receiving data

    return data


def _chunk_data(data_bytes: bytes) -> list[bytes]:
    """Breaks a byte string into fixed-size chunks."""
    return [data_bytes[i : i + config.CHUNK_SIZE] for i in range(0, len(data_bytes), config.CHUNK_SIZE)]


# TODO maybe change the 0.001 to 0.01 or 0.1, that might solve the extra probing issue
def _check_if_told_to_stop(sock: socket.socket, timeout: float = 0.001) -> bool:
    """
    Polls a socket to check if it is ready for reading.
    Returns True if the socket is readable, False otherwise.
    """
    ready_to_read, _, _ = select.select([sock], [], [], timeout)
    if sock in ready_to_read:
        signal = _recv_all(sock, 1)  # Read just 1 byte non-blockingly (due to select)
        if not signal:  # Connection closed
            raise ConnectionError("SENDER MLT: Connection closed by receiver during early check.")
        if signal == b"S":
            print("MLT HELPER: Received early 'Stop' (S) from server. Transmission for this gradient complete.")
            return True
        else:
            # This is unexpected if server only sends S/B in response to P.
            # Could log or handle as an error. For now, we might proceed,
            # but it could indicate a de-sync.
            print(f"MLT: Warning - Unexpected TCP data '{signal}' during early check. Proceeding with caution.")

    return False


# --- Serialization Function ---
def serialize_gradient_to_custom_binary(tcp_sock: socket.socket, key: str, tensor: torch.Tensor) -> bytes | None:
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
    # !Q is unsigned long long, 8 bytes
    # !i is signed int, 4 bytes
    packed_key_len = struct.pack("!I", len(key_bytes))

    try:
        tcp_sock.sendall(packed_key_len)  # Send key length first
        tcp_sock.sendall(key_bytes)  # Then send the key bytes
    except Exception as e:
        raise ConnectionError(f"tcp sock connection error: {e}")

    if config.DEBUG:
        print(f"Serializing key: {key} (length: {len(key_bytes)})")
        print("SENDER TCP: Sent packed_key_len and key in bytes")

    # # 2. Dtype string serialization
    # dtype_str = TORCH_DTYPE_TO_STR.get(tensor.dtype)
    # if not dtype_str:
    #     if config.DEBUG:
    #         print(f"Warning: Unsupported tensor dtype {tensor.dtype} for serialization. Attempting fallback.")
    #     # Fallback for dtypes not explicitly in TORCH_DTYPE_TO_STR (e.g. bfloat16 if user adds it)
    #     # This is a basic fallback; for robust handling of unlisted types, more logic is needed.
    #     dtype_str = str(tensor.dtype)
    #     # Consider adding it to STR_TO_TORCH_DTYPE dynamically or raising error if not found later
    #     if dtype_str not in STR_TO_TORCH_DTYPE:
    #         STR_TO_TORCH_DTYPE[dtype_str] = tensor.dtype  #  Attempt dynamic addition
    #     if tensor.dtype not in TORCH_TO_NUMPY_DTYPE:
    #         # This would be an issue for deserialization if no numpy equivalent is known
    #         raise ValueError(f"Unsupported tensor dtype for serialization: {tensor.dtype}. No NumPy equivalent mapped.")

    # dtype_str_bytes = dtype_str.encode("utf-8")
    # # !H is unsigned short, 2 bytes
    # packed_dtype_str_len = struct.pack("!I", len(dtype_str_bytes))

    # try:
    #     tcp_sock.sendall(packed_dtype_str_len)  # Send dtype string length first
    #     tcp_sock.sendall(dtype_str_bytes)  # Then send the dtype string bytes
    # except Exception as e:
    #     raise ConnectionError(f"tcp sock connection error: {e}")

    # if config.DEBUG:
    #     print(f"Serializing dtype: {dtype_str} (length: {len(dtype_str_bytes)})")
    #     print("SENDER TCP: Sent packed_dtype_str_len and dtype_str in bytes")

    # 3. Shape serialization
    shape = tensor.shape
    num_dimensions = len(shape)
    # !B is unsigned char, 1 byte
    packed_num_dimensions = struct.pack("!I", num_dimensions)
    # Pack each dimension
    packed_shape_dims = b"".join(struct.pack("!I", dim) for dim in shape)

    try:
        tcp_sock.sendall(packed_num_dimensions)  # Send number of dimensions first
        tcp_sock.sendall(packed_shape_dims)  # Then send the shape dimensions
    except Exception as e:
        raise ConnectionError(f"tcp sock connection error: {e}")

    if config.DEBUG:
        print(f"Serializing shape: {shape} (num_dimensions: {num_dimensions})")
        print("SENDER TCP: Sent packed_num_dimensions and packed_shape_dims in bytes")

    # 4. Tensor data serialization
    # Ensure tensor is on CPU before converting to NumPy array
    # .numpy() on a CUDA tensor raises an error. .cpu() is a no-op if already on CPU.
    # For gradients (param.grad), they usually don't require grad themselves.
    # If serializing a tensor that requires grad and is not a leaf, .detach() might be needed.
    # For gradient values, .cpu() is the primary concern.
    tensor_numpy = tensor.cpu().numpy()
    tensor_data_bytes = tensor_numpy.tobytes()
    packed_tensor_data_len = struct.pack("!I", len(tensor_data_bytes))

    try:
        tcp_sock.sendall(packed_tensor_data_len)  # Send tensor data length first
    except Exception as e:
        raise ConnectionError(f"tcp sock connection error: {e}")

    if config.DEBUG:
        print(f"Serializing tensor data: {len(tensor_data_bytes)} bytes")
        print("SENDER TCP: Sent packed_tensor_data_len in bytes")

    # # 5. send everything but the tensor data bytes through TCP
    # # tensor_data_bytes is sent separately via MLT protocol
    # metadata_bytes = b"".join(
    #     [
    #         packed_key_len,
    #         key_bytes,
    #         packed_dtype_str_len,
    #         dtype_str_bytes,
    #         packed_num_dimensions,
    #         packed_shape_dims,
    #         packed_tensor_data_len,
    #     ]
    # )
    try:
        # tcp_sock.sendall(metadata_bytes)
        # ack = _recv_all(tcp_sock, 1)  # Expecting an ACK byte from the receiver
        # if not ack or ack != b"A":
        #     raise ConnectionError("Failed to receive ACK after sending metadata.")
        pass
    except Exception as e:
        raise ConnectionError(f"tcp sock connection error: {e}")

    if config.DEBUG:
        print("Metadata sent successfully and no ACK sent.")

    return tensor_data_bytes


def send_data_mlt(socks: dict, addrs: dict, gradient_payload_bytes: bytes) -> bool:
    """
    Sends gradient data bytes using the MLT protocol (UDP with TCP-based ACK/retransmission).
    Returns True on success, False on failure.
    """
    tcp_sock = socks["tcp"]
    udp_sock = socks["udp"]
    tcp_host, tcp_port = addrs["tcp"]
    udp_host, udp_port = addrs["udp"]

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
            # --- Phase 1: Opportunistic check for early "Stop" from server ---
            # This is useful if server wants to terminate this stream early.
            # Using a very short timeout for a non-blocking check.

            # TODO: move it into the for loop b/c STOP signal can happen at any time
            # do it a couple of more times: before for loop, at the beginning of the for loop, and at the end of the for loop
            # and after for loop
            if _check_if_told_to_stop(tcp_sock):
                if config.DEBUG:
                    print("SENDER MLT: 'Stop' signal received at the beginning of while loop. Transmission COMPLETED.")
                    print("SENDER MLT: Have not sent any UDP chunks this round")
                return True

            # --- Phase 2: Send/Resend UDP chunks based on server_ack_bitmap ---
            chunks_sent_this_round = 0
            for i in range(num_chunks):
                # check stop signal has arrived or not
                if _check_if_told_to_stop(tcp_sock):
                    if config.DEBUG:
                        print("SENDER MLT: 'Stop' signal received at the beginning of for loop. Transmission COMPLETED.")
                        print(f"SENDER MLT: Sent {chunks_sent_this_round} UDP chunks this round")
                    return True

                byte_idx, bit_idx = divmod(i, 8)
                # Check if the i-th bit in server_ack_bitmap is 0 (server hasn't acked it)
                if not ((server_ack_bitmap[byte_idx] >> bit_idx) & 1):
                    chunk_payload = chunks[i]
                    # Header: chunk_id (0-indexed), total_chunks, payload_length_of_this_chunk
                    header = struct.pack("!III", i, num_chunks, len(chunk_payload))
                    packet_to_send = header + chunk_payload
                    try:
                        udp_sock.sendto(
                            packet_to_send,
                            (udp_host, udp_port),
                        )
                        chunks_sent_this_round += 1

                    except Exception as e:
                        # Log UDP send error but continue; rely on bitmap retransmission
                        print(f"SENDER MLT: UDP sendto error for chunk {i}: {e}")

                if _check_if_told_to_stop(tcp_sock):
                    if config.DEBUG:
                        print("SENDER MLT: 'Stop' signal received at the end of for loop. Transmission COMPLETED")
                        print(f"SENDER MLT: Sent {chunks_sent_this_round} UDP chunks this round")
                    return True

            # # --- Phase 3: Send "Probe" (P) signal via TCP ---
            # # Send 'Probe' (P) and wait for 'Stop' (S) or 'Bitmap' (B)
            # if config.DEBUG:
            #     print("SENDER MLT: Sending 'Probe' (P) via TCP.")
            # try:
            #     tcp_sock.sendall(b"P")
            # except Exception as e:
            #     raise ConnectionError(f"SENDER MLT ERROR: Failed to send 'Probe' (P) signal: {e}")

            #  --- Phase 4: Receive server's response (S or B + bitmap) ---
            # if config.DEBUG:
            #     print("SENDER MLT: Waiting for server response after 'Probe' (P).")

            # Use select with a reasonable timeout for the server to respond
            # probe_response_timeout = 3.0  # seconds
            # ready_to_read, _, _ = select.select([tcp_sock], [], [], probe_response_timeout)

            # if not ready_to_read:
            #     print(f"SENDER MLT: Timeout ({probe_response_timeout}s) waiting for server response to 'Probe'.")
            #     no_progress_rounds += 1
            #     if no_progress_rounds >= max_retries_no_progress:
            #         print("SENDER MLT: Max retries with no progress reached. Aborting.")
            #         return False
            #     continue  # Retry by sending probe again after resending unacked chunks

            signal = _recv_all(tcp_sock, 1)
            if not signal:
                raise ConnectionError("SENDER MLT ERROR: Connection closed by receiver while waiting for response.")

            if signal == b"S":
                print("SENDER MLT: Received 'Stop' (S). Transfer complete.")
                return True
            elif signal == b"B":
                print("SENDER MLT: Received 'Bitmap' (B) signal, receiving bitmap.")

                bitmap_len_to_recv = len(server_ack_bitmap)
                new_bitmap_data = _recv_all(tcp_sock, bitmap_len_to_recv)

                if not new_bitmap_data or len(new_bitmap_data) != bitmap_len_to_recv:
                    raise ConnectionError("SENDER MLT ERROR: Failed to receive complete bitmap data from server.")

                if bytearray(new_bitmap_data) == server_ack_bitmap and chunks_sent_this_round > 0:
                    no_progress_rounds += 1
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


def recv_data_mlt(socks: dict, tcp_addr: tuple, recv_lock=None) -> tuple[dict | None, tuple] | None:
    """
    Receives gradient data using the MLT protocol.
    Returns a dictionary of reconstructed gradients.
    """
    final_gradients_dict = {}
    tcp_sock = socks["tcp"]
    udp_sock = socks["udp"]

    # This entire block was added to create a clear protocol start.
    # The receiver first expects an 'E' (Eval) or 'N' (No Eval) signal,
    # then the number of gradients. This makes the start of the communication
    # much more robust and less ambiguous than just waiting for data.

    # STEP 0: determine if eval_acc and epoch will be sent from the worker
    eval_signal = _recv_all(tcp_sock, 1, recv_lock) if recv_lock else _recv_all(tcp_sock, 1)
    if not eval_signal:
        if config.DEBUG:
            print("RECEIVER: Did not receive initial eval signal. Connection may be closed.")
        return None

    # Case 1: signal is 'E' (eval_acc and epoch will be sent)
    if eval_signal == b"E":
        eval_acc_bytes = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
        epoch_bytes = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
        if not eval_acc_bytes or not epoch_bytes:
            raise ValueError("RECEIVER ERROR: Failed to receive eval data after 'E' signal.")
        eval_acc = struct.unpack("!f", eval_acc_bytes)[0]
        curr_epoch = struct.unpack("!f", epoch_bytes)[0]
        final_gradients_dict["eval_acc"] = eval_acc
        final_gradients_dict["epoch"] = curr_epoch
        print(f"[Worker {tcp_addr}] RECEIVER: Received 'E' signal with eval_acc={eval_acc}, epoch={curr_epoch}")
    # Case 2: signal is 'N' (no eval data will be sent)
    elif eval_signal == b"N":
        print("RECEIVER: Received 'N' signal (no eval data).")
    else:
        print(f"[Worker {tcp_addr}] RECEIVER ERROR: Unrecognized initial signal '{eval_signal}'.")
        return None

    # for each worker, all the important metadata will always be received first through TCP
    num_subgradients_bytes = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
    if not num_subgradients_bytes:
        print("RECEIVER ERROR: Failed to receive the number of sub-gradients.")
        return None
    num_subgradients = struct.unpack("!I", num_subgradients_bytes)[0]
    if config.DEBUG:
        print(f"[Worker {tcp_addr}] RECEIVER: Expecting to receive {num_subgradients} gradients.")

    # tcp_sock.setblocking(False)  # Set TCP socket to non-blocking mode

    # Loop to receive each gradient
    for grad_idx in range(num_subgradients):
        if config.DEBUG:
            print(f"\n---[WORKER {tcp_addr}] RECEIVER: Starting reception for gradient {grad_idx + 1}/{num_subgradients} ---")

        # readable, _, _ = select.select([tcp_sock], [], [], 1.0)

        # if tcp_sock in readable:
        # --- Receive Metadata for one gradient ---
        # 1. key deserialization
        # 1.1. receive the length of packed value and UNPACK it
        packed_key_len = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
        if not packed_key_len:
            return None
        key_len = struct.unpack("!I", packed_key_len)[0]

        if config.DEBUG:
            print(f"[Worker {tcp_addr}] RECEIVER TCP: Received key length {key_len} bytes")

        # 1.2. receive the actual value and DECODE it
        key_bytes = _recv_all(tcp_sock, key_len, recv_lock) if recv_lock else _recv_all(tcp_sock, key_len)
        if not key_bytes:
            return None
        key_str = key_bytes.decode("utf-8")
        # Initialize with None
        # to be filled out during UDP transmission
        final_gradients_dict[key_str] = None

        # --------------------------------------------------------------------------------------------------------------------------
        if config.DEBUG:
            print(f"[Worker {tcp_addr}] RECEIVER TCP: Received key {key_str}: {key_bytes} bytes")
        # ----------------------------------------------------------------------------------------------------------------------------

        # # 2. dtype string deserialization
        # # 2.1. ...
        # packed_dtype_str_len = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
        # if not packed_dtype_str_len:
        #     return None
        # dtype_str_len = struct.unpack("!I", packed_dtype_str_len)[0]

        # # 2.2. ...
        # dtype_str_bytes = _recv_all(tcp_sock, dtype_str_len, recv_lock) if recv_lock else _recv_all(tcp_sock, dtype_str_len)
        # if not dtype_str_bytes:
        #     return None
        # dtype_str = dtype_str_bytes.decode("utf-8")
        dtype_str = "torch.float32" 

        torch_dtype = STR_TO_TORCH_DTYPE.get(dtype_str, None)
        numpy_dtype = STR_TO_NUMPY_DTYPE.get(dtype_str, None)
        if not torch_dtype or not numpy_dtype:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        # # --------------------------------------------------------------------------------------------------------------------------
        # if config.DEBUG:
        #     print(f"[Worker {tcp_addr}] RECEIVER TCP: Received dtype {dtype_str} (length {dtype_str_len})")
        # # ----------------------------------------------------------------------------------------------------------------------------

        # 3. shape deserialization
        # 3.1
        packed_num_dimensions = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
        if not packed_num_dimensions:
            return None
        num_dimensions = struct.unpack("!I", packed_num_dimensions)[0]
        # 3.2
        shape_list = []
        shape_read_success = True
        for i in range(num_dimensions):
            packed_dim_size_bytes = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
            if not packed_dim_size_bytes:
                shape_read_success = False
                break
            dim_size = struct.unpack("!I", packed_dim_size_bytes)[0]
            shape_list.append(dim_size)
        if not shape_read_success:
            raise ValueError("Failed to read shape dimensions")
        shape_tuple = tuple(shape_list)

        # --------------------------------------------------------------------------------------------------------------------------
        if config.DEBUG:
            print(f"[Worker {tcp_addr}] RECEIVER TCP: Received shape {shape_tuple} (num_dimensions={num_dimensions})")
        # ----------------------------------------------------------------------------------------------------------------------------

        # 4. tensor data length deserialization
        packed_tensor_data_len = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
        if not packed_tensor_data_len:
            return None
        tensor_data_len_expected = struct.unpack("!I", packed_tensor_data_len)[0]

        if config.DEBUG:
            print(f"[Worker {tcp_addr}] RECEIVER TCP: Expected tensor data length {tensor_data_len_expected} bytes")

        if config.DEBUG:
            print(
                f"[Worker {tcp_addr}] RECEIVER TCP: Metadata OK for key='{key_str}', shape={shape_tuple}, expected_data_len={tensor_data_len_expected}"
            )

        # # after receiving all the metadata, need to send a TCP ACK
        # try:
        #     tcp_sock.sendall(b"A")  # Acknowledge receipt of metadata
        #     if config.DEBUG:
        #         print(f"[Worker {tcp_addr}] RECEIVER TCP: Sent ACK for metadata of key='{key_str}'.")
        # except Exception as e:
        #     print(f"[Worker {tcp_addr}] RECEIVER TCP ERROR: Failed to send ACK for metadata of key='{key_str}': {e}")
        #     return None

        # 5. Prepare to receive the tensor data
        #  UDP WILL START SOON
        # --- Receive Tensor Chunks via MLT ---
        num_chunks_bytes = _recv_all(tcp_sock, 4, recv_lock) if recv_lock else _recv_all(tcp_sock, 4)
        if not num_chunks_bytes:
            return None
        total_chunks = struct.unpack("!I", num_chunks_bytes)[0]
        if config.DEBUG:
            print(f"[Worker {tcp_addr}] RECEIVER MLT(TCP): Expecting {total_chunks} chunks for '{key_str}'.")

        received_chunks = [None] * total_chunks
        bitmap = bytearray((total_chunks + 7) // 8)
        expected_packet_size = config.CHUNK_SIZE + 12  # 12-byte header
        original_udp_timeout = udp_sock.gettimeout()
        udp_sock.settimeout(2.0)

        # ### DRAMATIC CHANGE START ###
        # The logic inside this while loop is clearer. It now explicitly waits for either
        # a UDP packet or a TCP probe using select.select(), making it more responsive
        # and less prone to deadlocks. The early termination logic is also clarified.
        has_stopped = False
        while None in received_chunks:
            try:
                readable, _, _ = select.select([udp_sock, tcp_sock], [], [], 0.001)

                if udp_sock in readable:
                    packet, udp_addr = udp_sock.recvfrom(expected_packet_size + 100)
                    if config.DEBUG:
                        print(f"[Worker {tcp_addr}] RECEIVER MLT(UDP): Received UDP packet of size {len(packet)} bytes.")
                    if len(packet) < 12:
                        print(f"[Worker {tcp_addr}] RECEIVER MLT(UDP): Packet too small: {len(packet)} bytes. Ignoring.")
                        continue
                    seq, _, chunk_len_in_header = struct.unpack("!III", packet[:12])
                    if seq < total_chunks and received_chunks[seq] is None and len(packet[12:]) == chunk_len_in_header:
                        received_chunks[seq] = packet[12:]
                        byte_idx, bit_idx = divmod(seq, 8)
                        bitmap[byte_idx] |= 1 << bit_idx

                    # ZM 6/12/2025: add one more case for the stop signal to send out
                    # rather than only send out early stop 'S' after receiving probe 'P'
                    # we can send it out anytime once the loss tolerace threshold has been met
                    if total_chunks > 0 and (received_chunks.count(None) / total_chunks) <= config.loss_tolerance:
                        if config.DEBUG:
                            print("RECEIVER MLT: sending early stop signal before probing.")
                            print(f"[Worker {tcp_addr}] RECEIVER MLT: Loss tolerance ({config.loss_tolerance}) met. Sending 'Stop' (S).")
                        has_stopped = True

                        # flush the tcp receive buffer to avoid any stale data
                        data = _flush_recv_buffer(tcp_sock, timeout=0.1)
                        if config.DEBUG and data:
                            print(f"[Worker {tcp_addr}] RECEIVER MLT: Flushed TCP buffer right before STOP signal sent out and got: {data}")
                        # send the stop signal
                        tcp_sock.sendall(b"S")
                        break

                if tcp_sock in readable:
                    signal = _recv_all(tcp_sock, 1, recv_lock) if recv_lock else _recv_all(tcp_sock, 1)
                    # STOP signal can be sent after receiving PROBE (P)
                    # need to make sure STOP signals are not sent twice
                    # after receiving PROBE (P)
                    if signal == b"P":
                        if config.DEBUG:
                            print("RECEIVER MLT: Received 'Probe' (P), sending 'Stop' (S) OR 'Bitmap' (B).")
                        # Early termination logic OR all chunks are received
                        if total_chunks > 0 and (received_chunks.count(None) / total_chunks) <= config.loss_tolerance:
                            if config.DEBUG:
                                print(f"[Worker {tcp_addr}] RECEIVER MLT: Loss tolerance ({config.loss_tolerance}) met. Sending 'Stop' (S).")
                            has_stopped = True
                            
                            # flush the tcp receive buffer to avoid any stale data
                            data = _flush_recv_buffer(tcp_sock, timeout=0.1)
                            if config.DEBUG and data:
                                print(f"[Worker {tcp_addr}] RECEIVER MLT: Flushed TCP buffer right before STOP signal sent out and got: {data}")
                            # send the stop signal
                            tcp_sock.sendall(b"S")
                            break
                        else:
                            if config.DEBUG:
                                print(f"[Worker {tcp_addr}] RECEIVER MLT: Sending 'Bitmap' (B) {bitmap}")
                            tcp_sock.sendall(b"B")
                            tcp_sock.sendall(bitmap)
                    elif not signal:
                        raise ConnectionError("RECEIVER MLT: Did not receive proper Probe (P) signal from sender.")
            except (socket.timeout, ConnectionError):
                break
            except Exception as e:
                print(f"[Worker {tcp_addr}] RECEIVER MLT ERROR: An unexpected error occurred: {e}")
                traceback.print_exc()
                break

        # if you got out of the loop, you should set the STOP signal
        if not has_stopped:
            if config.DEBUG:
                print("RECEIVER MLT: Got out of chunk send/recv loop but had not sent STOP (S) signal")
                print(f"    Sending it right now for '{key_str}'")
            
            # flush the tcp receive buffer to avoid any stale data
            data = _flush_recv_buffer(tcp_sock, timeout=0.1)
            if config.DEBUG and data:
                print(f"[Worker {tcp_addr}] RECEIVER MLT: Flushed TCP buffer right before STOP signal sent out and got: {data}")
            # send the stop signal
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
                if expected_chunk_size != config.CHUNK_SIZE:
                    print(f"[Worker {tcp_addr}] RECEIVER MLT: Chunk size {expected_chunk_size} is less than normal {config.CHUNK_SIZE}")
                final_data_list.append(chunk[:expected_chunk_size])
            else:
                # If chunk is missing, append zeros of the *correct* size
                if config.DEBUG:
                    print(f"[Worker {tcp_addr}] RECEIVER MLT: Zero-filling missing chunk #{i} with {expected_chunk_size} bytes.")
                    print("RECEIVER MLT: It might not be missing; could be the case of meeting loss tolerance threshold")
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

    return final_gradients_dict, udp_addr
