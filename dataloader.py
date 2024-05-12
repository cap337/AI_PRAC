import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(features_np,labels_np,batch_size=10):
    # Convert NumPy arrays to PyTorch tensors
    features = torch.tensor(features_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.float32)

    # Create TensorDataset
    dataset = TensorDataset(features, labels)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader