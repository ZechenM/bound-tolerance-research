import datetime
import select
import socket
import struct
import traceback

# import pickle
import numpy as np
import torch

import config
import utility
from config import STR_TO_NUMPY_DTYPE, STR_TO_TORCH_DTYPE


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
def _check_if_told_to_stop(sock: socket.socket, expected_counter: int, server_ack_bitmap: bytearray, timeout: float = 0.001):
    """
    Polls a socket to check if it is ready for reading.
    Returns True if the received signal is 'S' (Stop)
    Returns not None if received signal is 'B' (Bitmap) and updates the server_ack_bitmap.
    """
    ready_to_read, _, _ = select.select([sock], [], [], timeout)
    if sock in ready_to_read:
        signal, received_counter = utility.recv_signal_tcp(sock)  # Read just 1 byte non-blockingly (due to select)
        if received_counter != expected_counter:
            print(f"MLT: Warning - Received counter {received_counter} does not match expected counter {expected_counter}.")
            return False, None
        if not signal:  # Connection closed
            raise ConnectionError("MLT HELPER: Connection closed by receiver during early check.")
        if signal == b"S":
            print("MLT HELPER: Received early 'Stop' (S) from server. Transmission for this gradient complete.")
            return True, None
        elif signal == b"B":
            print("MLT HELPER: Received 'Bitmap' (B) signal, receiving bitmap.")

            bitmap_len_to_recv = len(server_ack_bitmap)
            new_bitmap_data = utility.recv_all(sock, bitmap_len_to_recv)

            if not new_bitmap_data or len(new_bitmap_data) != bitmap_len_to_recv:
                raise ConnectionError("MLT HELPER ERROR: Failed to receive complete bitmap data from server.")

            if bytearray(new_bitmap_data) == server_ack_bitmap and len(new_bitmap_data) > 0:
                print("MLT HELPER: No change in bitmap. Warning: Progress stalled.")

            server_ack_bitmap = bytearray(new_bitmap_data)

            return False, server_ack_bitmap  # Continue sending data
        else:
            # This is unexpected if sender only sends S/B in response to P.
            # Could log or handle as an error. For now, we might proceed,
            # but it could indicate a de-sync.
            print(f"MLT: Warning - Unexpected TCP data '{signal}' during early check. Proceeding with caution.")

    return False, None  # No signal received, continue sending data


# --- Serialization Function ---
def serialize_gradient_to_custom_binary(tcp_sock: socket.socket, key: str, tensor: torch.Tensor):
    """
    Serializes a key (string) and a tensor (torch.Tensor) into a custom binary format.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input 'tensor' must be a torch.Tensor. Got {type(tensor)}")
    if not isinstance(key, str):
        raise TypeError(f"Input 'key' must be a string. Got {type(key)}")

    # Ensure tensor is contiguous for reliable .tobytes() behavior
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    shape = tensor.shape
    tensor_numpy = tensor.cpu().numpy()
    tensor_data_bytes = tensor_numpy.tobytes()

    metadata = {
        "key": key,
        "shape": shape,
        "tensor_data_len": len(tensor_data_bytes),
    }

    return metadata, tensor_data_bytes


def send_data_mlt(socks: dict, addrs: dict, metadata: list, gradient_payload_bytes: bytes, signal_counter: int) -> bool:
    """
    Sends gradient data bytes using the MLT protocol (UDP with TCP-based ACK/retransmission).
    Returns True on success, False on failure.
    """
    tcp_sock = socks["tcp"]
    udp_sock = socks["udp"]
    tcp_host, tcp_port = addrs["tcp"]
    udp_host, udp_port = addrs["udp"]

    bitmap_same_from_last_round = False

    chunks = _chunk_data(gradient_payload_bytes)
    num_chunks = len(chunks)
    if num_chunks == 0:
        raise ValueError("SENDER MLT: No chunks to send. Exiting.")

    try:
        metadata.append(num_chunks)

        # -------------- TCP Phase: Send Metadata --------------
        # Send metadata first
        # utility.send_signal_tcp(tcp_sock, b"M")  # M for Metadata
        utility.send_data_tcp(tcp_sock, metadata)

        if config.DEBUG:
            print(f"SENDER MLT: Metadata sent with {num_chunks} chunks.")

        server_ack_bitmap = bytearray((num_chunks + 7) // 8)
        max_retries_no_progress = 10
        no_progress_rounds = 0

        # -------------- UDP Phase: Send Chunks --------------
        while True:
            # --- (Deprecated) Phase 1: Opportunistic check for early "Stop" from server ---

            # --- Phase 2: Send/Resend UDP chunks based on server_ack_bitmap ---
            chunks_sent_this_round = 0

            # ZM on 8.3.2025 Dramatic Change: will not do the for loop thing if previous round has the same bitmap
            if bitmap_same_from_last_round:
                if config.DEBUG:
                    print("SENDER MLT: Skipping chunk sending loop due to same bitmap from last round.")
            else:
                for i in range(num_chunks):
                    # ZM on 8.3.2025: I think even once of this signal checking over 2000+ chunks just for 1 communication is too much
                    # As a result, I want to see if checking it every 100 chunks is enough
                    if i % 100 == 0:  # Check every 100 chunks
                        # Check if the sender has sent a 'Stop' (S) or 'Bitmap' (B) signal
                        if config.DEBUG:
                            print(f"SENDER MLT: Checking for 'Stop' signal at chunk {i}/{num_chunks}.")
                        cond, new_bitmap = _check_if_told_to_stop(tcp_sock, signal_counter, server_ack_bitmap)
                        if cond:
                            if config.DEBUG:
                                print("SENDER MLT: 'Stop' signal received at the beginning of for loop. Transmission COMPLETED.")
                                print(f"SENDER MLT: Sent {chunks_sent_this_round} UDP chunks this round")
                            return True
                        else:
                            if new_bitmap is not None and new_bitmap != server_ack_bitmap:
                                server_ack_bitmap = new_bitmap

                    byte_idx, bit_idx = divmod(i, 8)
                    # Check if the i-th bit in server_ack_bitmap is 0 (server hasn't acked it)
                    if not ((server_ack_bitmap[byte_idx] >> bit_idx) & 1):
                        chunk_payload = chunks[i]
                        # Header: chunk_id (0-indexed), total_chunks, payload_length_of_this_chunk
                        header = struct.pack("!III", i, num_chunks, len(chunk_payload))
                        packet_to_send = header + chunk_payload
                        try:
                            udp_sock.sendto(
                                (packet_to_send),
                                (udp_host, udp_port),
                            )
                            chunks_sent_this_round += 1

                        except Exception as e:
                            # Log UDP send error but continue; rely on bitmap retransmission
                            print(f"SENDER MLT: UDP sendto error for chunk {i}: {e}")

                        if config.DEBUG:
                            now = datetime.datetime.now()
                            time_with_ms = f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}"
                            print(f"SENDER MLT: Sent chunk {i}/{num_chunks} via UDP at {time_with_ms}.")

                    # ZM 8.3.2025: I think doing this twice in the for loop has too much overhead.
                    # cond, new_bitmap = _check_if_told_to_stop(tcp_sock, signal_counter, server_ack_bitmap)
                    # if cond:
                    #     if config.DEBUG:
                    #         print("SENDER MLT: 'Stop' signal received at the end of for loop. Transmission COMPLETED")
                    #         print(f"SENDER MLT: Sent {chunks_sent_this_round} UDP chunks this round")
                    #     return True
                    # else:
                    #     if new_bitmap is not None and new_bitmap != server_ack_bitmap:
                    #         server_ack_bitmap = new_bitmap

            # --- Phase 3: Send "Probe" (P) signal via TCP ---
            while True:
                # Send 'Probe' (P) and wait for 'Stop' (S) or 'Bitmap' (B)
                if config.DEBUG:
                    # Get the current time object
                    now = datetime.datetime.now()

                    # Format using an f-string, calculating milliseconds from microseconds
                    time_with_ms = f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}"
                    print(f"SENDER MLT: Sending 'Probe' (P) via TCP at {time_with_ms}.")
                try:
                    utility.send_signal_tcp(tcp_sock, b"P", signal_counter)
                except Exception as e:
                    raise ConnectionError(f"SENDER MLT ERROR: Failed to send 'Probe' (P) signal: {e}")

                # Use select with a reasonable timeout for the server to respond
                probe_cnt = 0
                probe_response_timeout = 0.01  # seconds
                ready_to_read, _, _ = select.select([tcp_sock], [], [], probe_response_timeout)

                while not ready_to_read:
                    now = datetime.datetime.now()
                    if probe_cnt > 10:  # Arbitrary limit to avoid infinite loop
                        print(f"SENDER MLT: No response from server after 'Probe' (P) after {probe_cnt} attempts. Will re-Probe.")
                        break

                    if config.DEBUG:
                        time_with_ms = f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}"
                        print(f"SENDER MLT: No response from server after 'Probe' (P) at {time_with_ms} ({probe_cnt} / 10). Retrying...")

                    probe_cnt += 1
                    ready_to_read, _, _ = select.select([tcp_sock], [], [], probe_response_timeout)

                if ready_to_read:
                    # If the server responds, we can break out of the loop
                    if config.DEBUG:
                        print("SENDER MLT: Server responded to 'Probe' (P).")
                    break

            #  --- Phase 4: Receive server's response (S or B + bitmap) ---
            # reset bitmap_same_from_last_round to False
            bitmap_same_from_last_round = False
            signal, received_counter = utility.recv_signal_tcp(tcp_sock)
            if not signal:
                raise ConnectionError("SENDER MLT ERROR: Connection closed by receiver while waiting for response.")

            # CRITICAL CHANGE: Check if the received counter matches the expected signal_counter
            while received_counter != signal_counter:
                print(f"SENDER MLT: Received {signal} signal with counter {received_counter}. Expected: {signal_counter}. Retrying...")
                signal, received_counter = utility.recv_signal_tcp(tcp_sock)

            if signal == b"S":
                print("SENDER MLT: Received 'Stop' (S). Transfer complete.")
                return True
            elif signal == b"B":
                print("SENDER MLT: Received 'Bitmap' (B) signal, receiving bitmap.")

                bitmap_len_to_recv = len(server_ack_bitmap)
                new_bitmap_data = utility.recv_all(tcp_sock, bitmap_len_to_recv)

                if not new_bitmap_data or len(new_bitmap_data) != bitmap_len_to_recv:
                    raise ConnectionError("SENDER MLT ERROR: Failed to receive complete bitmap data from server.")

                if bytearray(new_bitmap_data) == server_ack_bitmap:
                    bitmap_same_from_last_round = True
                    no_progress_rounds += 1
                    print(f"SENDER MLT: No change in bitmap. Progress stalled ({no_progress_rounds}/{max_retries_no_progress}).")

                    if no_progress_rounds > max_retries_no_progress:
                        # to prevent deadlock
                        print("SENDER MLT: Max retries with no progress reached. Will retransmit UDP chunks.")
                        bitmap_same_from_last_round = False
                        no_progress_rounds = 0
                else:
                    no_progress_rounds = 0

                server_ack_bitmap = bytearray(new_bitmap_data)

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


def recv_data_mlt(socks: dict, tcp_addr: tuple, expected_counter: int, recv_lock=None) -> tuple[dict | None, tuple] | None:
    """
    Receives gradient data using the MLT protocol.
    Returns a dictionary of reconstructed gradients.
    """
    final_gradients_dict = {}
    tcp_sock = socks["tcp"]
    udp_sock = socks["udp"]

    dtype_str = "torch.float32"
    torch_dtype = STR_TO_TORCH_DTYPE.get(dtype_str, None)
    numpy_dtype = STR_TO_NUMPY_DTYPE.get(dtype_str, None)

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

    metadata_list = utility.receive_data_tcp(tcp_sock)
    num_chunks = metadata_list.pop()  # Last element is the number of chunks

    if len(metadata_list) != num_subgradients:
        raise ValueError(f"[Worker {tcp_addr}] RECEIVER ERROR: Expected {num_subgradients} metadata entries, but got {len(metadata_list)}.")

    # prepare to receive UDP chunks
    received_chunks = [None] * num_chunks
    bitmap = bytearray((num_chunks + 7) // 8)
    expected_packet_size = config.CHUNK_SIZE + 12  # 12-byte header

    has_stopped = False
    udp_recv_counter = 0
    while None in received_chunks:
        try:
            readable, _, _ = select.select([udp_sock, tcp_sock], [], [], 0.001)
            now = datetime.datetime.now()
            time_with_ms = f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}"

            # if not readable:
            #     if config.DEBUG:
            #         print(f"[Worker {tcp_addr}] RECEIVER MLT: No data received in this round. Continuing to wait at {time_with_ms}...")
            #     continue

            # if udp_sock not in readable:
            #     if config.DEBUG:
            #         print(f"[Worker {tcp_addr}] RECEIVER MLT: No data received on UDP socket. Continuing to wait at {time_with_ms}...")

            # if tcp_sock not in readable:
            #     if config.DEBUG:
            #         print(f"[Worker {tcp_addr}] RECEIVER MLT: No data received on TCP socket. Continuing to wait at {time_with_ms}...")

            if udp_sock in readable:
                if config.DEBUG:
                    print(f"[Worker {tcp_addr}] RECEIVER MLT(UDP): UDP socket is readable. Waiting for data from sender at {time_with_ms}...")
                packet, udp_addr = udp_sock.recvfrom(expected_packet_size + 100)
                now = datetime.datetime.now()
                time_with_ms = f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}"
                if config.DEBUG:
                    print(
                        f"[Worker {tcp_addr}] RECEIVER MLT(UDP): Received UDP packet of size {len(packet)} bytes ({udp_recv_counter}/{num_chunks}) at {time_with_ms}."
                    )
                    udp_recv_counter += 1
                if len(packet) < 12:
                    udp_recv_counter -= 1  # Ignore this packet, it is too small
                    print(f"[Worker {tcp_addr}] RECEIVER MLT(UDP): Packet too small: {len(packet)} bytes. Ignoring.")
                    continue

                # Unpick the packet
                # ZM 7.20 no need to pickle/unpickle b/c pickle will increase the packet size here
                # packet = pickle.loads(packet)

                seq, _, chunk_len_in_header = struct.unpack("!III", packet[:12])
                if seq < num_chunks and received_chunks[seq] is None and len(packet[12:]) == chunk_len_in_header:
                    received_chunks[seq] = packet[12:]
                    byte_idx, bit_idx = divmod(seq, 8)
                    bitmap[byte_idx] |= 1 << bit_idx

                # Check for early stop signal
                if num_chunks > 0 and (received_chunks.count(None) / num_chunks) <= config.loss_tolerance:
                    if config.DEBUG:
                        print(f"[Worker {tcp_addr}] RECEIVER MLT: Loss tolerance ({config.loss_tolerance}) met. Sending 'Stop' (S).")
                    has_stopped = True

                    # Send the stop signal
                    utility.send_signal_tcp(tcp_sock, b"S", expected_counter)
                    break

            # CRITICAL CHANGE FOR SIGNAL HANDLING
            if tcp_sock in readable:
                if config.DEBUG:
                    print(f"[Worker {tcp_addr}] RECEIVER MLT(TCP): TCP socket is readable. Waiting for signal from sender at {time_with_ms}...")
                signal, received_counter = utility.recv_signal_tcp(tcp_sock)

                if not signal:
                    raise ValueError(f"[Worker {tcp_addr}] RECEIVER ERROR: Failed to receive signal from TCP socket.")

                while received_counter != expected_counter:
                    print(
                        f"[Worker {tcp_addr}] RECEIVER MLT: Received {signal} signal with counter {received_counter}. Expected: {expected_counter}. Retrying..."
                    )
                    # If we receive a 'Probe' (P), we should resend unacked chunks
                    signal, received_counter = utility.recv_signal_tcp(tcp_sock)

                if signal == b"P":
                    # Get the current time object
                    now = datetime.datetime.now()

                    # Format using an f-string, calculating milliseconds from microseconds
                    time_with_ms = f"{now:%Y-%m-%d %H:%M:%S}.{now.microsecond // 1000:03d}"
                    if config.DEBUG:
                        print(
                            f"[Worker {tcp_addr}] RECEIVER MLT: Received 'Probe' (P) signal with counter {received_counter} from sender at {time_with_ms}."
                        )

                    if (received_chunks.count(None) / num_chunks) <= config.loss_tolerance:
                        if config.DEBUG:
                            print(f"[Worker {tcp_addr}] RECEIVER MLT: Loss tolerance ({config.loss_tolerance}) met. Sending 'Stop' (S).")
                        has_stopped = True

                        utility.send_signal_tcp(tcp_sock, b"S", received_counter)
                        break
                    else:
                        if config.DEBUG:
                            print(f"[Worker {tcp_addr}] RECEIVER MLT: Sending 'Bitmap' (B) {bitmap} with counter {received_counter} to sender.  ")
                        utility.send_signal_tcp(tcp_sock, b"B", received_counter)
                        # Send the bitmap back to the sender
                        tcp_sock.sendall(bitmap)
                elif not signal:
                    raise ConnectionError("RECEIVER MLT: Did not receive proper Probe (P) signal from sender.")
        except socket.timeout as e:
            print(f"[Worker {tcp_addr}] RECEIVER MLT: Socket timeout occurred: {e}")
            return None
        except socket.error as e:
            print(f"[Worker {tcp_addr}] RECEIVER MLT: Socket error occurred: {e}")
            return None
        except Exception as e:
            print(f"[Worker {tcp_addr}] RECEIVER MLT: Unexpected error occurred: {e}")
            return None

    if not has_stopped:
        if config.DEBUG:
            print("RECEIVER MLT: got out of chunk send/recv loop but had not sent STOP signal yet.")

        # # flush the tcp receive buffer to avoid any stale data
        # data = _flush_recv_buffer(tcp_sock, timeout=0.1)
        # if config.DEBUG and data:
        #     print(f"[Worker {tcp_addr}] RECEIVER MLT: Flushed TCP buffer right before STOP signal sent out and got: {data}")
        # send the stop signal
        utility.send_signal_tcp(tcp_sock, b"S", expected_counter)

    # ------------- Final Phase: Reconstruct Gradients -------------
    for i, chunk in enumerate(received_chunks):
        if chunk is None:
            # ZM 7/14: this should be fine because the only place t hat could go wrong is the last chunk
            # which could be smaller than config.CHUNK_SIZE
            # but our tensor reconstruction logic will handle that
            received_chunks[i] = b"\x00" * config.CHUNK_SIZE  # Fill missing chunks with empty bytes

    concatenated_data = b"".join(received_chunks)
    cursor = 0

    for metadata in metadata_list:
        if not isinstance(metadata, dict):
            raise ValueError(f"[Worker {tcp_addr}] RECEIVER ERROR: Metadata must be a dictionary. Got {type(metadata)}.")

        key = metadata.get("key")
        shape = metadata.get("shape")
        tensor_data_len_expected = metadata.get("tensor_data_len")

        if shape is None:
            raise ValueError(f"[Worker {tcp_addr}] RECEIVER ERROR: Tensor shape is None for key '{key}'.")
        if tensor_data_len_expected is None:
            raise ValueError(f"[Worker {tcp_addr}] RECEIVER ERROR: Expected tensor data length is None for key '{key}'.")
        if key is None:
            raise ValueError(f"[Worker {tcp_addr}] RECEIVER ERROR: Key is None in metadata for tensor reconstruction.")

        reconstructed_tensor = torch.zeros(shape, dtype=torch_dtype)

        if tensor_data_len_expected > 0:
            tensor_data = concatenated_data[cursor : cursor + tensor_data_len_expected]

            # Convert bytes to numpy array and then to torch tensor
            numpy_array = np.frombuffer(tensor_data, dtype=numpy_dtype)
            reconstructed_tensor = torch.from_numpy(numpy_array.copy()).reshape(shape).to(torch_dtype)
            cursor += tensor_data_len_expected

        final_gradients_dict[key] = reconstructed_tensor

        if config.DEBUG:
            print(f"[Worker {tcp_addr}] RECONSTRUCTION: Success for key '{key}'.")

    return final_gradients_dict, udp_addr
