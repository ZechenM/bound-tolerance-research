import argparse
import os
import pickle
from enum import Enum

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


class myDataset(Enum):
    MNIST = 0
    CIFAR10 = 1
    CIFAR100 = 2


def prepare_mnist_data():
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Split the dataset into 3 parts
    split_sizes = [20000, 20000, 20000]  # Each worker gets 20,000 samples
    split_datasets = random_split(mnist_dataset, split_sizes)

    # Create dataloaders for each worker
    dataloaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in split_datasets]

    return dataloaders


def prepare_cifar100_data(total_samples=375):
    """
    Prepare CIFAR-100 dataset for distributed training with VGG-13
    Args:
        total_samples (int): Total number of samples to use (default: 1125 for 375 samples per worker)
    """
    DATA_DIR = "./data/cifar100_data"
    SPLIT_DIR = "./data/cifar100_splits"
    TRAIN_SPLITS = 3
    TEST_RATIO = 0.2  # 20% of the dataset is used for testing

    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Transformations for VGG-13
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # VGG-13 expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print(f"Loading CIFAR-100 dataset (subset of {total_samples} samples)...")

    # Load CIFAR-100 dataset
    dataset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform)

    # Select a subset of samples if specified
    if total_samples is not None:
        indices = torch.randperm(len(dataset))[:total_samples]
        dataset = Subset(dataset, indices)

    # Split into train and test
    num_test = int(len(dataset) * TEST_RATIO)
    num_train = len(dataset) - num_test
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    # Split train dataset into 3 parts
    split_size = num_train // TRAIN_SPLITS
    split_sizes = [split_size] * (TRAIN_SPLITS - 1) + [num_train - split_size * (TRAIN_SPLITS - 1)]
    train_splits = random_split(train_dataset, split_sizes)

    # Save datasets for later use
    torch.save(test_dataset, os.path.join(SPLIT_DIR, "test.pth"))
    for i, split in enumerate(train_splits):
        torch.save(split, os.path.join(SPLIT_DIR, f"train_{i}.pth"))

    print(f"Dataset preparation complete. Splits saved in {SPLIT_DIR}")
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {num_train}")
    print(f"Test samples: {num_test}")
    print(f"Samples per worker: {split_size}")


def prepare_mnist_test_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    mnist_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    return DataLoader(mnist_dataset, batch_size=64, shuffle=True)


def prepare_cifar10_data(total_samples=None, samples_per_class=300):
    """
    Prepare CIFAR-10 dataset for distributed training with EfficientNetB0
    Args:
        total_samples (int): Total number of samples to use (default: None to use all samples)
    """
    DATA_DIR = "./data/cifar10_data"
    SPLIT_DIR = "./data/cifar10_splits"
    TRAIN_SPLITS = 3
    TEST_RATIO = 0.2  # 20% of the dataset is used for testing

    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Transformations for EfficientNetB0
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # EfficientNetB0 expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Loading CIFAR-10 dataset...")

    # Load CIFAR-10 dataset
    dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    print(len(dataset))

    # Select a subset of samples if specified
    if total_samples is not None:
        print(f"Selecting a random subset of {total_samples} samples...")
        indices = torch.randperm(len(dataset))[:total_samples]
        dataset = Subset(dataset, indices)

    print(f"CIFAR-10 dataset LOADED with {len(dataset)} samples")

    # Select exactly {samples_per_class} samples per class
    class_counts = {i: 0 for i in range(10)}
    selected_indices = []

    for idx, (_, label) in enumerate(dataset):
        if class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1
        if all(count == samples_per_class for count in class_counts.values()):
            break

    dataset = Subset(dataset, selected_indices)
    print(f"Dataset UPDATED with exactly {len(dataset)} samples ({samples_per_class} per class).")

    # Split into train and test
    num_test = int(len(dataset) * TEST_RATIO)
    num_train = len(dataset) - num_test
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    # Split train dataset into specified number of parts
    split_size = num_train // TRAIN_SPLITS
    split_sizes = [split_size] * (TRAIN_SPLITS - 1) + [num_train - split_size * (TRAIN_SPLITS - 1)]
    train_splits = random_split(train_dataset, split_sizes)

    # Save datasets for later use
    torch.save(test_dataset, os.path.join(SPLIT_DIR, "test.pth"))
    for i, split in enumerate(train_splits):
        torch.save(split, os.path.join(SPLIT_DIR, f"train_{i}.pth"))

    print(f"Dataset preparation complete. Splits saved in {SPLIT_DIR}")
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {num_train}")
    print(f"Test samples: {num_test}")
    print(f"Samples per worker: {split_sizes[0]}")


def mnist_train_test_split():
    # Prepare MNIST data
    dataloaders = prepare_mnist_data()
    for i, dataloaders in enumerate(dataloaders):
        with open(f"dataloader_{i}.pkl", "wb") as f:
            pickle.dump(dataloaders, f)
    print("MNIST data prepared and split into 3 dataloaders.")

    test_dataloader = prepare_mnist_test_data()
    filename = "dataloader_test.pkl"
    with open(filename, "wb") as f:
        pickle.dump(test_dataloader, f)
    print(f"Test dataset Dataloader stored as {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")

    args = parser.parse_args()

    match args.dataset:
        case "mnist":
            mnist_train_test_split()
        case "cifar10":
            # Prepare CIFAR-10 data for EfficientNetB0 / DenseNet
            prepare_cifar10_data(60000, 300)  # 300 samples per class; 3000 samples in total
            #  2400 -> training; 600 -> test validation
        case "cifar100":
            prepare_cifar100_data()
