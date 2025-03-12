import torch
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.images = []
        self.labels = []
        
        # Extract images and labels from the subset
        for img, label in subset:
            self.images.append(img)
            self.labels.append(label)
            
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Ensure proper tensor formats - don't rewrap tensors
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float)
        else:
            # Ensure it's a float tensor
            img = img.float()
            
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            # Ensure it's a long tensor for labels
            label = label.long()
            
        return {
            'pixel_values': img,
            'labels': label
        }

    def __init__(self, subset):
        self.subset = subset
        self.images = []
        self.labels = []
        
        # Extract images and labels from the subset
        for img, label in subset:
            self.images.append(img)
            self.labels.append(label)
            
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        # Debug print to verify data
        img = self.images[idx]
        label = self.labels[idx]
        
        # Ensure proper tensor formats - don't rewrap tensors
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float)
        else:
            # Ensure it's a float tensor
            img = img.float()
            
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            # Ensure it's a long tensor for labels
            label = label.long()
            
        # print(f"Returning data for index {idx}: {type(img)}, shape: {img.shape}")
        
        return {
            'pixel_values': img,
            'labels': label
        }