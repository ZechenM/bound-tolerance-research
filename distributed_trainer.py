import pickle
import socket
import struct
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer
from typing import List
import select
import traceback
from config import TORCH_DTYPE_TO_STR, STR_TO_TORCH_DTYPE, TORCH_TO_NUMPY_DTYPE

DEBUG = 1


def custom_collate_fn(batch):
    """
    Custom collate function to properly batch data for our model
    """
    # Print batch structure for debugging
    # if len(batch) > 0:
    # print(f"collate_fn: batch size={len(batch)}, first item keys={batch[0].keys()}")

    # Collate function to preserve the structure of the batch
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])

    return {"pixel_values": pixel_values, "labels": labels}


class DistributedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract our custom parameters before passing to parent
        self.server_host = kwargs.pop("server_host", "localhost")
        self.server_port = kwargs.pop("server_port", 60000)
        self.worker_id = kwargs.pop("worker_id", 0)
        self.device = kwargs.pop("device", torch.device("cpu"))  # Get device from kwargs
        self.network_latency_list = []
        self.start_time = 0
        self.end_time = 0
        self.past_epoch = 0.0
        self.protocol = kwargs.pop("protocol", "MLT")
        self.loss_tolerance = kwargs.pop("loss_tolerance", 0.03)
        if self.protocol == "MLT":
            self.chunk_size = 1024  # TODO: avoid hard-coding. Make it automatically aligh with the server.
            self.send_data = self.send_data_MLT
            self.recv_data = self.recv_data_TCP  # TODO: MLT recv is not working now, need UDP port. Temp using TCP
            self.UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.UDP_socket.bind(("0.0.0.0", self.server_port + 2 + self.worker_id))
        elif self.protocol == "TCP":
            self.send_data = self.send_data_TCP
            self.recv_data = self.recv_data_TCP
        else:
            raise TypeError(f"protocol {self.protocol} is not supported (TCP | MLT)")

        # Initialize parent class with remaining arguments
        super().__init__(*args, **kwargs)

    def send_data_TCP(self, sock, data):
        data_bytes = pickle.dumps(data)
        self.start_time = time.perf_counter()
        # !I == >I for unsigned int, 4 bytes
        sock.sendall(struct.pack("!I", len(data_bytes)))
        sock.sendall(data_bytes)
        self.end_time = time.perf_counter()
        self.calc_network_latency(True)
    
    # --- Serialization Function ---
    def serialize_gradient_to_custom_binary(self, tcp_sock: socket.socket, key: str, tensor: torch.Tensor) -> dict[str, bytes]:
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
        packed_key_len = struct.pack(">I", len(key_bytes))

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
                raise ValueError(
                    f"Unsupported tensor dtype for serialization: {tensor.dtype}. No NumPy equivalent mapped."
                )

        dtype_str_bytes = dtype_str.encode("utf-8")
        packed_dtype_str_len = struct.pack(">H", len(dtype_str_bytes))

        # 3. Shape serialization
        shape = tensor.shape
        num_dimensions = len(shape)
        packed_num_dimensions = struct.pack(">B", num_dimensions)
        # Pack each dimension
        packed_shape_dims = b"".join(struct.pack(">I", dim) for dim in shape)

        # 4. Tensor data serialization
        # Ensure tensor is on CPU before converting to NumPy array
        # .numpy() on a CUDA tensor raises an error. .cpu() is a no-op if already on CPU.
        # For gradients (param.grad), they usually don't require grad themselves.
        # If serializing a tensor that requires grad and is not a leaf, .detach() might be needed.
        # For gradient values, .cpu() is the primary concern.
        tensor_numpy = tensor.cpu().numpy()
        tensor_data_bytes = tensor_numpy.tobytes()
        packed_tensor_data_len = struct.pack(">Q", len(tensor_data_bytes))
        
        # 5. send everything but the tensor data bytes through TCP
        # tensor_data_bytes is sent separately via MLT protocol
        metadata_bytes = b"".join([
            packed_key_len,
            key_bytes,
            packed_dtype_str_len,
            dtype_str_bytes,
            packed_num_dimensions,
            packed_shape_dims,
            packed_tensor_data_len,
        ])
        tcp_sock.sendall(metadata_bytes)
        self.send_data_MLT(tcp_sock, tensor_data_bytes)

    def _chunk_gradient(self, data_bytes: bytes) -> List[bytes]:
        """Serialize gradient and break into chunks"""
        return [data_bytes[i : i + self.chunk_size] for i in range(0, len(data_bytes), self.chunk_size)]

    def send_data_MLT(
        self, tcp_sock: socket.socket, gradient_payload_bytes: bytes
    ) -> bool:
        """
        Sends gradient data using the MLT protocol.
        Returns True on success, False on failure.
        """
        chunks = self._chunk_gradient(gradient_payload_bytes)
        num_chunks = len(chunks)

        try:
            # 1. Send the total number of chunks for this gradient via TCP
            if DEBUG:
                print(f"MLT: Sending num_chunks = {num_chunks} via TCP.")
            tcp_sock.sendall(struct.pack("!I", num_chunks))

            if num_chunks == 0:
                if DEBUG:
                    print("MLT: No chunks to send. Probing for server acknowledgement.")
                # Even with 0 chunks, we need to "probe" to get the "Stop" signal.
                # The server needs to handle num_chunks=0 gracefully.
                # (The loop below will send 'P' and expect 'S').
                pass

            # Local bitmap representing chunks acknowledged by the server.
            # Initially, all are 0 (server has not acknowledged any).
            server_ack_bitmap = bytearray((num_chunks + 7) // 8)
            max_retries_no_progress = (
                3  # Number of rounds with no new acks before aborting
            )
            no_progress_rounds = 0

            while True:  # Retransmission loop
                # --- Phase 1: Opportunistic check for early "Stop" from server ---
                # This is useful if server wants to terminate this stream early.
                # Using a very short timeout for a non-blocking check.
                ready_to_read, _, _ = select.select([tcp_sock], [], [], 0.001)
                if tcp_sock in ready_to_read:
                    signal = tcp_sock.recv(
                        1
                    )  # Read just 1 byte non-blockingly (due to select)
                    if not signal:  # Connection closed
                        if DEBUG:
                            print("MLT: TCP connection closed by server (early check).")
                        return False
                    if signal == b"S":
                        if DEBUG:
                            print(
                                "MLT: Received early 'Stop' (S) from server. Transmission for this gradient complete."
                            )
                        return True
                    else:
                        # This is unexpected if server only sends S/B in response to P.
                        # Could log or handle as an error. For now, we might proceed,
                        # but it could indicate a de-sync.
                        print(
                            f"MLT: Warning - Unexpected TCP data '{signal}' during early check. Proceeding with caution."
                        )

                # --- Phase 2: Send/Resend UDP chunks based on server_ack_bitmap ---
                chunks_sent_this_round = 0
                for i in range(num_chunks):
                    byte_idx, bit_idx = divmod(i, 8)
                    # Check if the i-th bit in server_ack_bitmap is 0 (server hasn't acked it)
                    if not ((server_ack_bitmap[byte_idx] >> bit_idx) & 1):
                        chunk_payload = chunks[i]
                        # Header: chunk_id (0-indexed), total_chunks, payload_length_of_this_chunk
                        header = struct.pack(
                            "!III", i, num_chunks, len(chunk_payload)
                        )
                        packet_to_send = header + chunk_payload
                        try:
                            # TODO: is (self.server_host, self.server_port + 1) correct?
                            self.UDP_socket.sendto(
                                packet_to_send,
                                (self.server_host, self.server_port + 1),
                            )
                            chunks_sent_this_round += 1
                        except Exception as e:
                            # Log UDP send error but continue; rely on bitmap retransmission
                            print(f"MLT: UDP sendto error for chunk {i}: {e}")
                if DEBUG:
                    print(
                        f"MLT: Sent {chunks_sent_this_round} UDP chunks this round."
                    )

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
                ready_to_read, _, _ = select.select(
                    [tcp_sock], [], [], probe_response_timeout
                )

                if not ready_to_read:
                    print(
                        f"MLT: Timeout ({probe_response_timeout}s) waiting for server response to 'Probe'."
                    )
                    no_progress_rounds += 1
                    if no_progress_rounds >= max_retries_no_progress:
                        print("MLT: Max retries with no progress reached. Aborting.")
                        return False
                    continue  # Retry by sending probe again after resending unacked chunks

                signal = tcp_sock.recv(1)
                if not signal:  # Connection closed or _recv_all_tcp failed
                    if DEBUG:
                        print(
                            "MLT: Failed to receive signal from server or connection closed after probe."
                        )
                    return False

                if DEBUG:
                    print(f"MLT: Received signal '{signal}' from server.")

                if signal == b"S":  # "Stop" signal
                    if DEBUG:
                        print(
                            "MLT: Received 'Stop' (S). Transmission for this gradient complete."
                        )
                    return True
                elif signal == b"B":  # "Bitmap" signal
                    if DEBUG:
                        print(
                            "MLT: Received 'Bitmap' (B)"
                        )

                    new_bitmap_data = tcp_sock.recv(len(server_ack_bitmap))
                    # Check if bitmap indicates progress
                    if (
                        bytearray(new_bitmap_data) == server_ack_bitmap
                        and chunks_sent_this_round > 0
                    ):
                        # We sent chunks, but the bitmap didn't change.
                        no_progress_rounds += 1
                        if DEBUG:
                            print(
                                f"MLT: No change in bitmap after sending chunks. Progress stalled ({no_progress_rounds}/{max_retries_no_progress})."
                            )
                    else:
                        no_progress_rounds = 0  # Progress was made

                    server_ack_bitmap = bytearray(new_bitmap_data)
                    if no_progress_rounds >= max_retries_no_progress:
                        print(
                            f"MLT: Max retries ({max_retries_no_progress}) with no progress in bitmap. Aborting."
                        )
                        return False

                    # Check if all chunks are now acknowledged (optional optimization here,
                    # as the loop condition and server sending 'S' is the primary completion mechanism)
                    if num_chunks > 0 and all(
                        (server_ack_bitmap[i // 8] >> (i % 8)) & 1
                        for i in range(num_chunks)
                    ):
                        if DEBUG:
                            print(
                                "MLT: All chunks appear to be acknowledged by bitmap. Next probe should yield 'S'."
                            )
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
            traceback.print_exc()
            return False
        finally:
            # Assuming tcp_sock was blocking and we didn't change its global state.
            # If you had tcp_sock.setblocking(False) at the start of the function,
            # you'd restore it here: tcp_sock.setblocking(True)
            pass

        # Should have exited loop via 'S' signal for successful completion
        if DEBUG:
            print("MLT: Exited main loop unexpectedly (should have received 'S').")
        return False  # Should not be reached if logic is correct and S is received

    def recv_data_TCP(self, sock):
        # First receive the ACK from the server
        # ack = sock.recv(1)
        # if ack != b"A":
        #     print(f"Warning: Expected ACK but received: {ack}")

        # Now receive the actual data with size header
        size_data = sock.recv(4)
        if not size_data:
            return None
        size = struct.unpack("!I", size_data)[0]

        self.start_time = time.perf_counter()
        data = b""
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data += packet

        self.end_time = time.perf_counter()
        self.calc_network_latency(False)
        return pickle.loads(data)

    def recv_data_MLT(self, TCP_sock):
        """
        Receive data in MLT style. This is for receiving averaged gradient from the server

        Work process:
        1. server send "A" ACK signal, to notify the start of work
        2. server send over number of chunks
        3. server send over chunks through UDP channel. This function keeps tracks of chunks received in bitmap
        4. At any time, a "P" probe signal can be received through TCP channel. If so, send the server the bitmap
        5. client (this function) will sent a "S" signal to early terminate, when loss is within boundary.

        Remark: an outdated bitmap can be sent, therefore a chunk might be resent. It's ok, new chunk will contain
                the same content, and will overwrite recorded chunk to reduce code complexity. Outdated bitmap
                will never result in chunk desolate (chunk not sent but bitmap is set to 1).
        """
        # Receive ready-go signal
        # ack = TCP_sock.recv(1)
        # if ack != b"A":
        #     print(f"Warning: Expected ACK but received: {ack}")

        # Receive total number of chunks
        size_data = TCP_sock.recv(4)
        if not size_data:
            return None
        total_chunks = struct.unpack("!I", size_data)[0]

        self.start_time = time.perf_counter()

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
                    else:
                        print(f"recv_data_MLT: cannot recognize signal from server:{signal}")

                if self.UDP_socket in readable:
                    # Receive packet with extra buffer space
                    packet, _ = self.UDP_socket.recvfrom(expected_packet_size + 100)

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
                    missing_rate = received_chunks.count(None) / total_chunks
                    if missing_rate < self.loss_tolerance:
                        TCP_sock.sendall(b"S")
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
        self.end_time = time.perf_counter()
        self.calc_network_latency(is_send=False)

        # if chunk not received, fill with 0
        for i, chunk in enumerate(received_chunks):
            if chunk == None:
                received_chunks[i] = bytes(self.chunk_size)

        # Reconstruct original data
        data_bytes = b"".join(received_chunks)

        return pickle.loads(data_bytes)

    def send_recv(self, gradients):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # print(self.server_host, self.server_port)
            s.connect((self.server_host, self.server_port))
            print(f"Worker {self.worker_id} connected to server.")
            if self.protocol == "MLT":
                # Send data using MLT protocol
                for key, tensor in gradients.items():
                    self.serialize_gradient_to_custom_binary(s, key, tensor)
            else:                            
                self.send_data(s, gradients)
            avg_gradients = self.recv_data(s)
            if avg_gradients is None:
                return False, None
        return True, avg_gradients

    def get_train_dataloader(self):
        """
        Override the default dataloader to use our custom collate function
        """
        # print("Using custom get_train_dataloader")
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=custom_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(self, model, inputs, num_items_in_batch):
        """
        Override the training step to implement distributed training
        Args:
            model: The model to train
            inputs: The inputs for the current step
            num_items_in_batch: Number of items in the current batch
        """
        # Ensure model is on the correct device
        model = model.to(self.device)
        model.train()

        current_epoch = self.state.epoch
        print(f"Training Step Running - Worker {self.worker_id}, Current Epoch: {current_epoch}")

        # evaluate the model at the end of each epoch
        sent_eval = False
        if current_epoch is not None:
            epoch_diff = current_epoch - self.past_epoch
            if abs(int(current_epoch) - current_epoch) < 1e-10 and epoch_diff > 0.999 or current_epoch == 0:
                self.past_epoch = current_epoch
                eval_results = self.evaluate()
                eval_acc = eval_results["eval_accuracy"]
                curr_epoch = eval_results["epoch"]
                sent_eval = True

                print(f"Sent to server eval acc {eval_acc} at epoch {current_epoch}.")

        # Handle empty inputs - this should not happen if get_train_dataloader is working properly
        if not inputs or not isinstance(inputs, dict) or "pixel_values" not in inputs:
            print(f"Warning: Invalid inputs detected: {inputs}")
            # Create dummy data to prevent crashing - we'll fix the data flow
            x = torch.randn(4, 3, 224, 224).to(self.device)
            labels = torch.randint(0, 100, (4,)).to(self.device)
        else:
            # Ensure inputs are on the correct device
            x = inputs["pixel_values"].to(self.device)
            labels = inputs["labels"].to(self.device)

        # Forward pass
        outputs = model(x)

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, labels)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        # Get gradients and ensure they're on CPU for communication
        gradients = {name: param.grad.cpu() for name, param in model.named_parameters() if param.grad is not None}
        if sent_eval:
            gradients["eval_acc"] = eval_acc
            gradients["epoch"] = curr_epoch

        # Send gradients to server and receive averaged gradients
        update, avg_gradients = self.send_recv(gradients)

        if not update:
            print(f"Worker {self.worker_id} failed to receive averaged gradients.")
            return loss.detach()

        # Update model parameters with averaged gradients
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in avg_gradients:
                    param.grad = avg_gradients[name].to(self.device)

        # FIXED: ValueError: Calculated loss must be on the original device: mps:0 but device in use is cpu
        # actually what fixed the above error is to enable mps mode
        return loss.detach().to(self.device)

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """
        Override the train method to force training to continue when resuming
        """
        print("--------------------------------")
        print(f"Worker {self.worker_id} starting training")
        print("--------------------------------")

        # Get the original resume_from_checkpoint value
        resume_checkpoint = resume_from_checkpoint if resume_from_checkpoint is not None else self.args.resume_from_checkpoint

        # Set a flag to track if we're resuming
        is_resuming = resume_checkpoint is not None

        if is_resuming:
            print(f"Worker {self.worker_id} resuming from checkpoint: {resume_checkpoint}")

            # Override the state to ensure we don't skip training
            self.state.global_step = 0

            # Force max_steps to be higher than what's in the checkpoint
            # This ensures training continues even if checkpoint says we're done
            if isinstance(resume_checkpoint, str) and "checkpoint" in resume_checkpoint:
                try:
                    checkpoint_step = int(resume_checkpoint.split("-")[-1])
                    print(f"Detected checkpoint step: {checkpoint_step}")
                    # Make sure max_steps is higher than checkpoint step
                    if self.args.max_steps <= checkpoint_step:
                        print(f"Adjusting max_steps from {self.args.max_steps} to {checkpoint_step + 20500}")
                        self.args.max_steps = checkpoint_step + 20500  # should be double if I want to do 2 time more epoch
                except:
                    print("Could not parse checkpoint step")

        # Call the parent's train method with our adjusted settings
        # This will still load the checkpoint state but won't skip training
        result = super().train(resume_from_checkpoint=resume_checkpoint, trial=trial, ignore_keys_for_eval=ignore_keys_for_eval)

        # print(f"Worker {self.worker_id} training completed successfully")
        # self.print_total_network_latency()

        return result

    def calc_network_latency(self, is_send):
        self.network_latency_list.append(self.end_time - self.start_time)
        if is_send:
            print(f"Send Network latency: {self.end_time - self.start_time}")
        else:
            print(f"Recv Network latency: {self.end_time - self.start_time}")
        self.start_time = 0
        self.end_time = 0

    def print_total_network_latency(self):
        print(f"Total network latency for worker {self.worker_id}: {sum(self.network_latency_list)}")

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Override the default eval dataloader to use our custom collate function
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=custom_collate_fn,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override the prediction step to ensure inputs are properly handled
        Args:
            model: The model to evaluate
            inputs: The inputs for prediction
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore in the model output
        """
        model = model.to(self.device)
        model.eval()

        with torch.no_grad():
            # Ensure inputs are on the correct device
            x = inputs["pixel_values"].to(self.device)
            labels = inputs["labels"].to(self.device)

            # Forward pass
            outputs = model(x)

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs, labels)

            # For evaluation, we need to return the loss, outputs, and labels
            if prediction_loss_only:
                return (loss, None, None)
            else:
                return (loss, outputs, labels)
