import numpy as np
import argparse
import pickle
import random
import socket
import struct
from enum import Enum

import torch

from compression import *
from config import (
    TORCH_DTYPE_TO_STR,
    STR_TO_TORCH_DTYPE,
    TORCH_TO_NUMPY_DTYPE,
    STR_TO_NUMPY_DTYPE,
)
import select

DEBUG = 1

print(f"Compression Method: {compression_method}")


class TrainingPhase(Enum):
    BEGIN = 0
    MID = 1
    FINAL = 2


class Server:
    def __init__(self, host, port, num_workers=3):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.training_phase = TrainingPhase.BEGIN
        self.server_socket = None
        self.prev_avg_acc = 0.0
        self.worker_eval_acc = []
        self.worker_epochs = []

        self.drop_rate = 0.0  # X% probability to zero out gradients

        # v-threshold configurations and trackers
        self.enable_v_threshold = False
        self.v = 0  # just like TCP, start with 0, update additively
        self.iter_count = 0  # increment every iter processed. If reached self.update_v_per, clear, and update v accordingly
        self.update_v_per = 100  # update v per 100 iter
        # self.AIDM_add = 0.1       # under-dropped, v = v + self.AIDM_add
        self.AIMD_decrease = 0.5  # over-dropped, v = v * self.AIMD_decrease
        # self.overall_max_abs_value = 0
        self.median_tracker = 0
        self.counter = 0
        self.total_params = 0
        self.zeroed_params = 0

        # gradient communication protocal configurations
        self.protocol = protocol
        if self.protocol == "MLT":
            self.chunk_size = 1024  # TODO: avoid hard-coding. Make it automatically aligh with the trainer.
            self.send_data = self.send_data_TCP  # TODO: MLT send is not working now, need UDP port. Temp using TCP
            self.recv_data = self.recv_data_MLT
            self.UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.UDP_socket.bind(("0.0.0.0", self.port + 1))  # TODO: each worker should allocate one UDP socket
            self.loss_tolerance = loss_tolerance
        elif self.protocol == "TCP":
            self.send_data = self.send_data_TCP
            self.recv_data = self.recv_data_TCP
        else:
            raise TypeError(f"protocol {self.protocol} is not supported (TCP | MLT)")

        self.write_to_server_port()
        self.start_server()
        self.run_server()
        # Thread(target=self.handle_user_input, daemon=True).start()

    def write_to_server_port(self):
        print("Writing server port to .server_port file...")
        with open(".server_port", "w") as f:
            f.write(str(self.port))
            f.flush()

    def start_server(self) -> None:
        # Create a socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_workers)
        self.is_listening = True
        print(f"Server listening on {self.host}:{self.port}...")

        print(f"V-threshold enabled: {self.enable_v_threshold}")

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
        # print("Closed all worker connections.")

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

    def _training_phase_update(self):
        if len(self.worker_eval_acc) < 3:
            self.has_eval_acc = False
            return

        curr_avg_acc = sum(self.worker_eval_acc) / len(self.worker_eval_acc)
        acc_diff = curr_avg_acc - self.prev_avg_acc

        proposed_phase = self.training_phase  # Initially set as current phase

        if any(self.worker_epochs) == 0:
            proposed_phase = TrainingPhase.BEGIN
        elif acc_diff > 0.07:
            proposed_phase = TrainingPhase.BEGIN
        elif curr_avg_acc > 0.5:
            if acc_diff < 0.03:
                proposed_phase = TrainingPhase.FINAL
            elif acc_diff <= 0.07:
                proposed_phase = TrainingPhase.MID
            else:
                proposed_phase = TrainingPhase.BEGIN
        # Ensure no phase transitions if the accuracy is not above 50%, even if acc_diff is low
        else:
            proposed_phase = TrainingPhase.BEGIN

        # Ensure one-way state transitions:
        if proposed_phase.value >= self.training_phase.value:
            self.training_phase = proposed_phase

        self.prev_avg_acc = curr_avg_acc

        print(f"All worker eval acc: {self.worker_eval_acc}")
        print(f"All worker epochs: {self.worker_epochs}")
        print(f"Current averaged accuracy: {self.prev_avg_acc}")
        print(f"Current training phase: {self.training_phase}")

        self.worker_eval_acc.clear()
        self.worker_epochs.clear()


    def recv_data_TCP(self, TCP_sock):
        # Receive the size of the incoming data
        size_data = self.recv_all(TCP_sock, 4)
        if not size_data:
            raise ValueError("Failed to receive data size.")

        size = struct.unpack("!I", size_data)[0]

        # Receive the actual data
        data = self.recv_all(TCP_sock, size)
        if not data:
            raise ValueError("Failed to receive data.")

        # response with ACK
        # TCP_sock.sendall(b"A")

        compressed_grad = pickle.loads(data)
        grad = decompress(compressed_grad)

        if "eval_acc" in grad:
            self.has_eval_acc = True
            self.worker_eval_acc.append(grad["eval_acc"])
            self.worker_epochs.append(grad["epoch"])
            del grad["epoch"]
            del grad["eval_acc"]
        
        return grad


    def recv_data_MLT(self, TCP_sock):
        # there will be 3 connections from each worker
        # for each worker, all the important metadata will always be received first through TCP
        num_sub_gradients = self.recv_all(TCP_sock, 4)
        if not num_sub_gradients:
            return None
        # num_sub_gradients is interchangable as layers
        # we are sending gradients layer by layer, so key-val pair will be reconstructed iteration by iteration
        num_sub_gradients = struct.unpack("!I", num_sub_gradients)[0]
        final_gradients_dict = {}
        
        for _ in range(num_sub_gradients):
            # 1. key deserialization
            # 1.1. receive the length of packed value and UNPACK it
            packed_key_len = self.recv_all(TCP_sock, 4)
            if not packed_key_len:
                return None
            key_len = struct.unpack("!I", packed_key_len)[0]

            # 1.2. receive the actual value and DECODE it
            key_bytes = self.recv_all(TCP_sock, key_len)
            if not key_bytes:
                return None
            key_str = key_bytes.decode("utf-8")
            # Initialize with None
            # to be filled out during UDP transmission
            final_gradients_dict[key_str] = None  

            # 2. dtype string deserialization
            # 2.1. ...
            packed_dtype_str_len = self.recv_all(TCP_sock, 2)
            if not packed_dtype_str_len:
                return None
            dtype_str_len = struct.unpack("!H", packed_dtype_str_len)[0]

            # 2.2. ...
            dtype_str_bytes = self.recv_all(TCP_sock, dtype_str_len)
            if not dtype_str_bytes:
                return None
            dtype_str = dtype_str_bytes.decode("utf-8")
            
            torch_dtype = STR_TO_TORCH_DTYPE.get(dtype_str, None)
            numpy_dtype = STR_TO_NUMPY_DTYPE.get(dtype_str, None)
            if not torch_dtype or not numpy_dtype:
                raise ValueError(f"Unsupported dtype: {dtype_str}")
            
            # 3. shape deserialization
            # 3.1 
            packed_num_dimensions = self.recv_all(TCP_sock, 1)
            if not packed_num_dimensions:
                return None
            num_dimensions = struct.unpack("!B", packed_num_dimensions)[0]
            # 3.2
            shape_list = []
            shape_read_success = True
            for i in range(num_dimensions):
                packed_dim_size_bytes = self.recv_all(TCP_sock, 4)
                if not packed_dim_size_bytes:
                    shape_read_success = False
                    break
                dim_size = struct.unpack("!I", packed_dim_size_bytes)[0]
                shape_list.append(dim_size)
            if not shape_read_success:
                raise ValueError("Failed to read shape dimensions")
            shape_tuple = tuple(shape_list)

            print(f"Shape: {shape_list}")
            
            # 4. UDP for receiving the tensor data (by chunks)
            packed_tensor_data_len = self.recv_all(TCP_sock, 8)
            if not packed_tensor_data_len:
                return None
            tensor_data_len_expected = struct.unpack("!Q", packed_tensor_data_len)[0]

            # 5. Prepare to receive the tensor data
            size_data = TCP_sock.recv(4)
            if not size_data:
                return None
            total_chunks = struct.unpack("!I", size_data)[0]

            # Initialize storage and bitmap
            received_chunks = [None] * total_chunks
            bitmap = bytearray((total_chunks + 7) // 8)  # 1 bit per chunk
            expected_packet_size = self.chunk_size + 12  # 12-byte header
            socket_timeout = 2.0  # Adjust based on network conditions

            # Set socket timeout
            self.UDP_socket.settimeout(socket_timeout)
            TCP_sock.setblocking(False)  # Make TCP non-blocking
            while None in received_chunks:
                try:
                    readable, _, _ = select.select([self.UDP_socket, TCP_sock], [], [], 0.001)
                    if TCP_sock in readable:
                        signal = TCP_sock.recv(1)
                        if signal == b"P":
                            TCP_sock.sendall(b"B")
                            TCP_sock.sendall(bitmap)
                            if DEBUG: print(bitmap)
                        else:
                            print(f"recv_data_MLT: cannot recognize signal from server:{signal}")

                    if self.UDP_socket in readable:
                        # Receive packet with extra buffer space
                        packet, _ = self.UDP_socket.recvfrom(expected_packet_size + 100)
                        if DEBUG: print("received packets")
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
                        if missing_rate < self.loss_tolerance:
                            TCP_sock.sendall(b"S")
                            if DEBUG: print("recv_data_MLT: early termination")
                            break

                except socket.timeout:
                    print("Timeout waiting for packets")
                    break
                except Exception as e:
                    print(f"Error receiving packet: {e}")
                    break

                # Reset socket timeout
                self.UDP_socket.settimeout(None)
                TCP_sock.setblocking(True)
                # self.end_time = time.perf_counter()
                # self.calc_network_latency(is_send=False)

                # if chunk not received, fill with 0
                for i, chunk in enumerate(received_chunks):
                    if chunk == None:
                        if DEBUG: print("fill with zeros")
                        received_chunks[i] = bytes(self.chunk_size)

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
                    print(f"Reconstruction WARNING for '{key_str}': TCP expected_data_len ({tensor_data_len_expected}) "
                        f"mismatches bytes for shape*dtype ({bytes_expected_by_shape_dtype}). "
                        f"Attempting reconstruction with received buffer of size {len(final_tensor_data_as_bytes)}.")
                    # If tensor_data_len_expected is not a multiple of itemsize, np.frombuffer might behave unexpectedly
                    # or only read up to the last full element.
                    # It's safer to ensure the buffer used matches bytes_expected_by_shape_dtype if possible,
                    # or handle the discrepancy by creating zeros if reconstruction is impossible.
                    if len(final_tensor_data_as_bytes) >= bytes_expected_by_shape_dtype:
                        # Use only the part of the buffer that corresponds to the shape
                        buffer_for_reconstruction = final_tensor_data_as_bytes[:bytes_expected_by_shape_dtype]
                        np_array = np.frombuffer(buffer_for_reconstruction, dtype=numpy_dtype)
                        if np_array.size == num_elements_in_shape: # Check if number of elements is correct
                            np_array = np_array.reshape(shape_tuple)
                            reconstructed_tensor = torch.from_numpy(np_array).to(torch_dtype)
                        else: # Should not happen if buffer_for_reconstruction was sized correctly
                            print(f"  Reconstruction ERROR for '{key_str}' (mismatch case): Element count mismatch. Creating zero tensor.")
                            reconstructed_tensor = torch.zeros(shape_tuple, dtype=torch_dtype)
                    else: # Not enough data in the buffer (even with zero-filling) for the shape
                        print(f"  Reconstruction ERROR for '{key_str}' (mismatch case): Not enough data in buffer ({len(final_tensor_data_as_bytes)}) for shape ({bytes_expected_by_shape_dtype} bytes needed). Creating zero tensor.")
                        reconstructed_tensor = torch.zeros(shape_tuple, dtype=torch_dtype)
                else: # tensor_data_len_expected == bytes_expected_by_shape_dtype
                    # hopefully every time we fall into this case 
                    np_array = np.frombuffer(final_tensor_data_as_bytes, dtype=numpy_dtype)
                    np_array = np_array.reshape(shape_tuple)
                    reconstructed_tensor = torch.from_numpy(np_array).to(torch_dtype)
                    print(f"  Reconstruction: Tensor reconstructed for '{key_str}'.")

                final_gradients_dict[key_str] = reconstructed_tensor

            except ValueError as ve:
                print(f"  Reconstruction ERROR for '{key_str}': ValueError (likely reshape failed due to size mismatch) - {ve}. Creating zero tensor.")
                final_gradients_dict[key_str] = torch.zeros(shape_tuple, dtype=torch_dtype)
            except Exception as e:
                print(f"  Reconstruction ERROR for '{key_str}': Unexpected error - {e}. Skipping this gradient from dict.")
                # Decide if to break or continue for other gradients
                # break # Safer to break if unexpected reconstruction error occurs
                continue # Or try to process next gradient if error is isolated

        print(f"\nReceiver: Finished processing loop. Total gradients in dictionary: {len(final_gradients_dict)}")
        return final_gradients_dict


    def send_data_TCP(self, TCP_sock, gradient):
        # Send the size of the data first
        TCP_sock.sendall(struct.pack("!I", len(gradient)))
        # Sendall the actual data
        TCP_sock.sendall(gradient)


    def send_data_MLT(self, TCP_sock, gradient):
        # divide the gradients, start the timer and send size information to the server throught TCP
        chunks = self._chunk_gradient(gradient)
        # self.start_time = time.perf_counter()
        # TCP_sock.sendall(b"A")  # send over a ready-go signal
        TCP_sock.sendall(struct.pack("!I", len(chunks)))
        bitmap = bytearray((len(chunks) + 7) // 8)  # 1 bit per chunk
        TCP_sock.setblocking(False)  # Make TCP non-blocking
        while True:
            readable, _, _ = select.select([TCP_sock], [], [], 0.001)
            if TCP_sock in readable:
                signal = TCP_sock.recv(1)
                if signal == b"S":
                    break
                else:
                    print(f"send_data_MLT: server sending improper signal: {signal}")
            # Using UDP to send over gradient in the form of chunks. Label each chunk with consecutive ID
            for i in range(len(chunks)):
                byte_index = i // 8
                bit_index = i % 8
                received = (bitmap[byte_index] >> bit_index) & 1
                if not received:
                    chunk = chunks[i]
                    header = struct.pack("!III", i, len(chunks), len(chunk))
                    packet = header + chunk
                    self.UDP_socket.sendto(packet, (self.server_host, self.server_port + 1))

            try:
                # try to send "probe" signal to the server through TCP network.
                TCP_sock.sendall(b"P")  # "probe" signal
                signal = TCP_sock.recv(1)
                if signal == b"S":  # "stop" signal
                    break
                elif signal == b"B":  # "bitmap signal"
                    bitmap = TCP_sock.recv(len(bitmap))
                else:
                    raise ValueError(f"cannot recognize signal from server: {signal}")
            except Exception as e:
                print("send_data_MLT: ", e)
                break

        TCP_sock.setblocking(True)
        # self.end_time = time.perf_counter()
        # self.calc_network_latency(is_send=True)


    def recv_send(self):
        """Receive gradients from all workers and send back averaged gradients"""
        gradients = []
        self.has_eval_acc = False

        for conn in self.connections:
            grad = self.recv_data(conn)
            gradients.append(grad)
            # print(f"Received gradients from worker {self.conn_addr_map[conn]}")

        # Received gradients from all workers
        # print("All gradients received.")

        # check accuracy and update training phase accrodingly
        if self.has_eval_acc:
            self._training_phase_update()

        # based on training phase, update the drop rate
        if self.training_phase == TrainingPhase.BEGIN:
            self.drop_rate = 0.0
        elif self.training_phase == TrainingPhase.MID:
            self.drop_rate = 0.0
        elif self.training_phase == TrainingPhase.FINAL:
            self.drop_rate = 0.0

        if self.has_eval_acc:
            print(f"Current drop rate: {self.drop_rate}\n")

        # print(f"1. type of gradients: {type(gradients)}, should be list")
        # print(f"2. len of gradients: {len(gradients)}, should be 3")
        # print(f"3. type of gradients[0]: {type(gradients[0])}, should be dict")
        # print("gradients looks like:", gradients)
        avg_gradients = {}
        if not self.enable_v_threshold:
            # Existing averaging logic:
            for key in gradients[0].keys():
                if random.random() < self.drop_rate:  # x% probability to zero out gradients
                    avg_gradients[key] = torch.zeros_like(gradients[0][key])
                    # print(f"Gradient '{key}' zeroed out randomly.")
                else:
                    avg_gradients[key] = torch.stack([grad[key] for grad in gradients]).mean(dim=0)
        else:  # v-threshold is enabled
            for grad_dict in gradients:
                for key in grad_dict:
                    tensor = grad_dict[key]

                    # Update max absolute value
                    # self.overall_max_abs_value = max(self.overall_max_abs_value, tensor.abs().max().item())
                    self.median_tracker += tensor.mean().item()
                    self.counter += 1

                    # Count parameters
                    self.total_params += tensor.numel()

                    # Zero out small values
                    mask = tensor.abs() < self.v
                    self.zeroed_params += mask.sum().item()
                    tensor[mask] = 0  # In-place modification

            # right gradients are set to 0 in-place, calculate average
            for key in gradients[0].keys():
                avg_gradients[key] = torch.stack([grad[key] for grad in gradients]).mean(dim=0)

            print(
                f"current v: {self.v}",
                f"iter_count {self.iter_count}",
                f"zeroed_param {self.zeroed_params}",
                f"total params: {self.total_params}",
                f"drop rate: {self.zeroed_params / self.total_params}",
                flush=True,
            )

            self.iter_count += 1
            if self.iter_count >= self.update_v_per:
                # clear, and update v accordingly
                self.iter_count = 0
                actual_drop_rate = self.zeroed_params / self.total_params
                if actual_drop_rate < self.drop_rate:
                    # increase v, zero more gradients, increase drop rate
                    # see https://github.com/ZechenM/bound-tolerance-research/issues/9
                    # self.v += self.overall_max_abs_value * 0.0001
                    real_median = self.median_tracker / self.counter
                    self.v += abs(real_median) * 0.5

                else:  # actual_drop_rate >= self.drop_rate
                    # decrease v, allow more grdients to get passed, decrease drop rate
                    self.v *= self.AIMD_decrease
                self.overall_max_abs_value = 0
                self.total_params = 0
                self.zeroed_params = 0

        # Compress the averaged gradients
        compressed_avg_gradients = compress(avg_gradients)
        avg_gradients_data = pickle.dumps(compressed_avg_gradients)

        # Send averaged gradients back to all workers
        for conn in self.connections:
            self.send_data(conn, avg_gradients_data)
            # print(f"Sent averaged gradients to worker {self.conn_addr_map[conn]}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="60001")
    args = parser.parse_args()

    server_host = str(args.host)
    server_port = int(args.port)
    print(f"Starting server at port with {server_port}...")
    server = Server(server_host, server_port)
