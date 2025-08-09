import socket
import struct
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer

import mlt
import utility


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


class DistributedTrainerMultithreading(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract our custom parameters before passing to parent
        self.server_host = kwargs.pop("server_host", "localhost")
        self.tcp_port = kwargs.pop("server_port", 9999)
        # ID has to be passed in - should not give it a valid default value
        self.id = kwargs.pop("worker_id", -666)
        if self.id == -666:
            raise ValueError("Cannot identify the worker id for this worker")
        self.device = kwargs.pop("device", torch.device("cpu"))  # Get device from kwargs
        self.network_latency_list = []
        self.start_time = 0
        self.end_time = 0
        self.past_epoch = 0.0
        self.protocol = kwargs.pop("protocol", "MLT")
        self.loss_tolerance = kwargs.pop("loss_tolerance", 0.03)
        self.signal_counter = 0  # Initialize signal counter for MLT protocol
        self.metadata_list = []  # Store metadata for MLT protocol
        self.has_sent_metadata = False  # Track if metadata has been sent

        # network latency measurement
        self.start_time = 0
        self.end_time = 0
        self.network_latency_list = []

        # Initialize parent class with remaining arguments
        super().__init__(*args, **kwargs)

    def calculate_network_latency(self):
        """
        Calculate the network latency based on the start and end times.
        """

        latency = self.end_time - self.start_time
        self.network_latency_list.append(latency)

        print(f"[Worker {self.id}] (W->S->W) Network latency: {latency:.6f} seconds")

    def print_total_network_latency(self):
        """
        Print the total network latency for all operations.
        """
        if self.network_latency_list:
            total_latency = sum(self.network_latency_list)
            print(f"[Worker {self.id}] Total network latency: {total_latency:.6f} seconds")
        else:
            print(f"[Worker {self.id}] No network latency recorded.")

    def connect(self):
        """Establishes connection with the server and gets the dedicated UDP port."""
        try:
            self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_sock.connect((self.server_host, self.tcp_port))
            worker_addr = self.tcp_sock.getsockname()
            print(f"[Worker {self.id}] Successfully connected to server at {self.server_host}:{self.tcp_port}")
            print(f"[Worker {self.id}] Worker itself address: {worker_addr}")

            port_data = mlt._recv_all(self.tcp_sock, 4)
            if not port_data:
                print(f"[Worker {self.id}] Failed to receive dedicated UDP port from server.")
                return False

            self.dedicated_server_udp_port = struct.unpack("!I", port_data)[0]
            print(f"[Worker {self.id}] Received dedicated server UDP port: {self.dedicated_server_udp_port}")

            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return True

        except Exception as e:
            print(f"[Worker {self.id}] Failed to connect to server: {e}")
            return False

    def close(self):
        """closes the worker's sockets."""
        if self.tcp_sock:
            self.tcp_sock.close()
        if self.udp_sock:
            self.udp_sock.close()

        print(f"[Worker {self.id}] Connections closed.")

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
        print(f"Training Step Running - Worker {self.id}, Current Epoch: {current_epoch}")

        # evaluate the model at the end of each epoch
        self.sent_eval = False
        if current_epoch is not None:
            epoch_diff = current_epoch - self.past_epoch
            if abs(int(current_epoch) - current_epoch) < 1e-10 and epoch_diff > 0.999 or current_epoch == 0:
                self.past_epoch = current_epoch
                eval_results = self.evaluate()
                self.eval_acc = eval_results["eval_accuracy"]
                self.curr_epoch = eval_results["epoch"]
                self.sent_eval = True
                print(f"Current epoch: {current_epoch}, Self curr epoch: {self.curr_epoch}")

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
        gradients: dict[str, torch.Tensor | float] = {name: param.grad.cpu() for name, param in model.named_parameters() if param.grad is not None}
        # ZM 6/7/2025: for MLT, we don't add these 2 fields to the gradients dictionary
        # instead, we send a signal to tell the receiver whether we are sending them or not

        # ZM 6/27/2025: this file will not consider the case where the protocol is TCP
        # so the following logic will never be triggered
        if self.sent_eval and self.protocol == "TCP":
            gradients["eval_acc"] = self.eval_acc
            gradients["epoch"] = self.curr_epoch

        # ZM 8/6/2025: metadata will only be sent once at the very beginning of the training
        if not self.has_sent_metadata:
            for key, tensor in gradients.items():
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Gradient for key '{key}' is not a tensor: {type(tensor)}")
                metadata, _ = mlt.serialize_gradient_to_custom_binary(self.tcp_sock, key, tensor)
                self.metadata_list.append(metadata)

            # send metadata to the server
            utility.send_data_tcp(self.tcp_sock, self.metadata_list)
            self.has_sent_metadata = True
        else:
            print(f"[Worker {self.id}] Metadata already sent: {len(self.metadata_list)} items. E.g. {self.metadata_list[0]}")

        # --------------- 6/27 UPDATES: bring in MLT -------------------------------
        self.start_time = time.perf_counter()  # Start measuring network latency

        # 1. send local gradients
        self.send_gradients(gradients)
        self.signal_counter += 1  # Increment signal counter after sending

        # 2. Wait to receive the globally averaged gradients from the server

        print(f"[Worker {self.id}] Gradients sent. Waiting to receive averaged model back...")
        socks = {"tcp": self.tcp_sock, "udp": self.udp_sock}
        result = mlt.recv_data_mlt(socks, (self.server_host, self.tcp_port), self.signal_counter, self.metadata_list)
        self.signal_counter += 1  # Increment signal counter after receiving

        self.end_time = time.perf_counter()
        self.calculate_network_latency()

        if result is None:
            raise ValueError(f"[Worker {self.id}] Server disconnected. Shutting down.")

        averaged_gradients, _ = result
        print(f"[Worker {self.id}] Successfully received averaged gradients.")

        if averaged_gradients is None:
            raise ValueError(f"[WORKER {self.id}] failed to receive averaged gradients back from the server")

        # Send gradients to server and receive averaged gradients
        avg_gradients = averaged_gradients

        # Update model parameters with averaged gradients
        # ZM 6/7/2025: that is like an extra check that we won't
        # include the custom key-val pairs in the gradients
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in avg_gradients:
                    param.grad = avg_gradients[name].to(self.device)

        # FIXED: ValueError: Calculated loss must be on the original device: mps:0 but device in use is cpu
        # actually what fixed the above error is to enable mps mode
        return loss.detach().to(self.device)

    def send_gradients(self, gradients_dict):
        """Sends a dictionary of gradients to the server using the mlt protocol."""
        if not self.tcp_sock or not self.udp_sock:
            print(f"[Worker {self.id}] Not connected. Cannot send gradients.")
            return

        print(f"[Worker {self.id}] Starting to send {len(gradients_dict)} gradients...")

        if self.sent_eval:
            self.tcp_sock.sendall(b"E")
            self.tcp_sock.sendall(struct.pack("!f", self.eval_acc))
            self.tcp_sock.sendall(struct.pack("!f", self.curr_epoch))
            print(f"WORKER: Sent eval signal 'E' with data: acc={self.eval_acc}, epoch={self.curr_epoch}")
        else:
            self.tcp_sock.sendall(b"N")
            print("WORKER: Sent 'no eval' signal 'N'.")

        # instead of sending each gradient one by one, we will send them all at once
        payload_bytes_list: list[bytes] = []

        socks = {"tcp": self.tcp_sock, "udp": self.udp_sock}
        addrs = {"udp": (self.server_host, self.dedicated_server_udp_port)}
        addrs["tcp"] = (self.server_host, self.tcp_port)

        for key, tensor in gradients_dict.items():
            _, payload_bytes = mlt.serialize_gradient_to_custom_binary(self.tcp_sock, key, tensor)
            if payload_bytes is None:
                raise ValueError(f"[Worker {self.id}] Failed to serialize tensor data for key '{key}'.")
            payload_bytes_list.append(payload_bytes)

        # concatenate payload bytes into a single bytes object
        all_payload_bytes = b"".join(payload_bytes_list)

        success = mlt.send_data_mlt(socks, addrs, all_payload_bytes, self.signal_counter)
        if not success:
            raise ValueError(f"[Worker {self.id}] Failed to send tensor data using MLT protocol.")

        print(f"[Worker {self.id}] Finished sending gradients.")

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """
        Override the train method to force training to continue when resuming
        """
        print("--------------------------------")
        print(f"Worker {self.id} starting training")
        print("--------------------------------")

        # Get the original resume_from_checkpoint value
        resume_checkpoint = resume_from_checkpoint if resume_from_checkpoint is not None else self.args.resume_from_checkpoint

        # Set a flag to track if we're resuming
        is_resuming = resume_checkpoint is not None

        if is_resuming:
            print(f"Worker {self.id} resuming from checkpoint: {resume_checkpoint}")

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
                except Exception:
                    print("Could not parse checkpoint step")

        # Call the parent's train method with our adjusted settings
        # This will still load the checkpoint state but won't skip training
        result = super().train(
            resume_from_checkpoint=resume_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

        print(f"Worker {self.id} training completed successfully")
        self.print_total_network_latency()

        return result

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
