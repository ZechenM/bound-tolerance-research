import pickle
import socket
import struct
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer
from typing import List


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
        if self.protocol == "MLT":
            self.chunk_size = 1024
            self.send_data = self.send_data_MLT
            self.recv_data = self.recv_data_MLT
        elif self.protocol == "TCP":
            self.send_data = self.send_data_TCP
            self.recv_data = self.recv_data_TCP
        else:
            raise TypeError(f"protocol {self.protocol} is not supported (TCP | MLT)")
        self.UDP_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Initialize parent class with remaining arguments
        super().__init__(*args, **kwargs)

    def send_data_TCP(self, sock, data):
        data_bytes = pickle.dumps(data)
        self.start_time = time.perf_counter()
        sock.sendall(struct.pack("!I", len(data_bytes)))
        sock.sendall(data_bytes)
        self.end_time = time.perf_counter()
        self.calc_network_latency(True)

    def _chunk_gradient(self, gradient) -> List[bytes]:
        """Serialize gradient and break into chunks"""
        data_bytes = pickle.dumps(gradient)
        return [data_bytes[i : i + self.chunk_size] for i in range(0, len(data_bytes), self.chunk_size)]

    def send_data_MLT(self, TCP_sock, gradient):
        """
        Using MLT to communicate with the server about gradient update.

        MLT:
        1. divide the gradients into fix-sized chunks, label chunks into consecutive IDs
        2. Using UDP network to send over chunks. When it's done, send over a "probe" signal to request receiver status
        3. If "stop" signal is not received, and status bitmap is received, re-transmit missing gradient to the server
        """
        # divide the gradients, start the timer and send size information to the server throught TCP
        chunks = self._chunk_gradient(gradient)
        self.start_time = time.perf_counter()
        TCP_sock.sendall(struct.pack("!I", len(chunks)))
        stop = False
        bitmap = bytearray((len(chunks) + 7) // 8)  # 1 bit per chunk
        while not stop:
            # Using UDP to send over gradient in the form of chunks. Label each chunk with consecutive ID
            # for seq, chunk in enumerate(chunks):
            #     header = struct.pack("!III", seq, len(chunks), len(chunk))
            #     packet = header + chunk
            #     self.UDP_socket.sendto(packet, (self.server_host, self.server_port))

            for i in range(len(chunks)):
                byte_index = i // 8
                bit_index = i % 8
                received = (bitmap[byte_index] >> bit_index) & 1
                if not received:
                    chunk = chunks[i]
                    header = struct.pack("!III", i, len(chunks), len(chunk))
                    packet = header + chunk
                    self.UDP_socket.sendto(packet, (self.server_host, self.server_port))

            try:
                # try to send "probe" signal to the server through TCP network.
                TCP_sock.sendall(b"P")  # "probe" signal
                signal = TCP_sock.recv(1)
                if signal == b"S":  # "stop" signal
                    stop = True
                elif signal == b"B":  # "bitmap signal"
                    bitmap = TCP_sock.recv(len(bitmap))
                else:
                    raise ValueError(f"cannot recognize signal from server: {signal}")
            except Exception as e:
                print("from send_data_MLT", e)
                stop = True

        self.end_time = time.perf_counter()
        self.calc_network_latency(is_send=True)

    def recv_data_TCP(self, sock):
        # First receive the ACK from the server
        ack = sock.recv(1)
        if ack != b"A":
            print(f"Warning: Expected ACK but received: {ack}")

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
        self.start_time = time.perf_counter()

        self.end_time = time.perf_counter()
        self.calc_network_latency(is_send=False)
        return  # averaged gradient

    def send_recv(self, gradients):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # print(self.server_host, self.server_port)
            s.connect((self.server_host, self.server_port))
            print(f"Worker {self.worker_id} connected to server.")
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
