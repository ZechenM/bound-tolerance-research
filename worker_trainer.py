import glob
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments

from distributed_trainer import DistributedTrainer
from my_datasets import CIFAR10Dataset

from config import *

resume_from_checkpoint = False

train_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    learning_rate=0.001,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=False,
    dataloader_pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
    report_to="none",
    logging_first_step=True,  # Log metrics for the first step
    # fp8=True,
)


class Worker:
    def __init__(self, worker_id, host="localhost", port="60001"):
        self.worker_id = worker_id
        self.server_host = host
        self.server_port = port
        print(f"Worker {self.worker_id} connecting to server at {self.server_host}:{self.server_port}")

        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")

        # # Load untrained EfficientNetB0 model
        self.model = models.efficientnet_b0(weights=None)

        # Load denseNet169 model
        # self.model = models.densenet169(weights=None)
        # Modify the model's classifier to output 10 classes (CIFAR10)
        # in_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(in_features, 10)
        self.model = self.model.to(self.device)
        print(f"Model moved to {self.device}")

        # Get the absolute path to the distributed-ml-training directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        SPLIT_DIR = os.path.join(current_dir, "data", "cifar10_splits")

        # Load training dataset split
        train_split_path = os.path.join(SPLIT_DIR, f"train_{self.worker_id}.pth")
        if not os.path.exists(train_split_path):
            raise FileNotFoundError(
                f"Training dataset split not found at {train_split_path}. Please run prepare_data.py first to generate CIFAR10 splits."
            )

        # Load test dataset split
        test_split_path = os.path.join(SPLIT_DIR, "test.pth")
        if not os.path.exists(test_split_path):
            raise FileNotFoundError(
                f"Test dataset split not found at {test_split_path}. Please run prepare_data.py first to generate CIFAR10 splits."
            )

        # Load both dataset splits
        # with safe_globals([torch.utils.data.dataset.Subset]):
        train_subset = torch.load(train_split_path, weights_only=False)
        test_subset = torch.load(test_split_path, weights_only=False)
        print(f"Loaded CIFAR10 training split containing {len(train_subset)} samples")
        print(f"Loaded CIFAR10 test split containing {len(test_subset)} samples")

        # Convert the subsets into our custom dataset format
        self.train_dataset = CIFAR10Dataset(train_subset)
        self.eval_dataset = CIFAR10Dataset(test_subset)
        print(f"Created CIFAR10 datasets for worker {self.worker_id}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.eval_dataset)}")

        # Verify DataLoader output
        # for batch in self.train_dataset:
        #     print(f"train_dataset Batch keys: {batch.keys()}")
        #     break  # Just to check the first batch

    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        Args:
            eval_pred: tuple of (predictions, labels)
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}

    def find_latest_checkpoint(self):
        """Find the latest checkpoint directory for this worker."""
        checkpoint_dir = f"./results_worker_{self.worker_id}"
        if not os.path.exists(checkpoint_dir):
            return None

        # Find all checkpoint directories
        checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
        if not checkpoint_dirs:
            return None

        # Extract checkpoint numbers and find the latest one
        checkpoint_numbers = [int(d.split("-")[-1]) for d in checkpoint_dirs]
        latest_checkpoint_num = max(checkpoint_numbers)
        latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint-{latest_checkpoint_num}")

        # Verify that model file exists
        model_path = os.path.join(latest_checkpoint, "model.safetensors")
        if os.path.exists(model_path):
            return latest_checkpoint
        return None

    def train_worker(self):
        # Initialize the distributed trainer
        self.training_args = train_args
        self.training_args.output_dir = f"./results_worker_{self.worker_id}"
        self.training_args.logging_dir = f"./logs_worker_{self.worker_id}"

        # Check for latest checkpoint
        latest_checkpoint = self.find_latest_checkpoint()
        # latest_checkpoint = None
        if latest_checkpoint:
            print(f"Found latest checkpoint at {latest_checkpoint}. Will resume training from this point.")
        else:
            print("No checkpoint found. Starting training from scratch.")

        trainer = DistributedTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            server_host=self.server_host,
            server_port=self.server_port,
            worker_id=self.worker_id,
            device=self.device,
            protocol=protocol,
        )

        # Start training
        print(f"Worker {self.worker_id} starting training with evaluation...")
        if resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)
        else:
            print("Starting training from scratch")
            train_result = trainer.train()

        # Training completed
        print(f"Worker {self.worker_id} training DONE: {train_result}")

        # Explicitly evaluate after training
        eval_results = trainer.evaluate()
        print(f"Worker {self.worker_id} evaluation DONE: {eval_results}")

        trainer.print_total_network_latency()


def main():
    if len(sys.argv) < 2:
        print("Invalid usage.")
        print("USAGE: python worker_trainer.py <WORKER_ID> [<SERVER_IP>] [<SERVER_PORT>]")
        sys.exit(1)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    worker_id = int(sys.argv[1])
    host = str(sys.argv[2]) if len(sys.argv) > 2 else "localhost"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 60001

    worker = Worker(worker_id, host=host, port=port)
    worker.train_worker()


if __name__ == "__main__":
    main()
