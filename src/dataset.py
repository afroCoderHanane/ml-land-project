"""
Dataset module for EuroSAT land use classification.
Maps 10 EuroSAT classes to binary classification: City vs Farmland.
"""

import os
from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np


# EuroSAT class mapping
# Original classes: Annual Crop, Forest, Herbaceous Vegetation, Highway,
#                   Industrial, Pasture, Permanent Crop, Residential, River, Sea Lake
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Binary mapping: 0 = City, 1 = Farmland
CITY_CLASSES = {'Residential', 'Industrial', 'Highway'}
FARMLAND_CLASSES = {'AnnualCrop', 'PermanentCrop', 'Pasture'}
EXCLUDED_CLASSES = {'Forest', 'HerbaceousVegetation', 'River', 'SeaLake'}

CLASS_TO_BINARY = {
    'Residential': 0,    # City
    'Industrial': 0,     # City
    'Highway': 0,        # City
    'AnnualCrop': 1,     # Farmland
    'PermanentCrop': 1,  # Farmland
    'Pasture': 1,        # Farmland
}

BINARY_CLASS_NAMES = {0: 'City', 1: 'Farmland'}


class BinaryEuroSATDataset(torch.utils.data.Dataset):
    """
    Wrapper for EuroSAT dataset that maps to binary classification.
    Excludes classes not in City or Farmland categories.
    """
    
    def __init__(self, eurosat_dataset):
        """
        Args:
            eurosat_dataset: Original EuroSAT dataset from torchvision
        """
        self.eurosat_dataset = eurosat_dataset
        
        # Filter indices to only include City and Farmland classes
        self.valid_indices = []
        self.binary_labels = []
        
        for idx in range(len(eurosat_dataset)):
            _, label = eurosat_dataset[idx]
            class_name = EUROSAT_CLASSES[label]
            
            if class_name in CLASS_TO_BINARY:
                self.valid_indices.append(idx)
                self.binary_labels.append(CLASS_TO_BINARY[class_name])
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Returns (image, binary_label) tuple"""
        original_idx = self.valid_indices[idx]
        image, _ = self.eurosat_dataset[original_idx]
        binary_label = self.binary_labels[idx]
        return image, binary_label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Returns distribution of binary classes"""
        labels_array = np.array(self.binary_labels)
        return {
            'City': int(np.sum(labels_array == 0)),
            'Farmland': int(np.sum(labels_array == 1))
        }


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Returns data transforms for training or validation/test.
    
    Args:
        is_training: If True, includes data augmentation
    
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(
    data_root: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test dataloaders for EuroSAT dataset.
    
    Args:
        data_root: Root directory for data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        train_split: Proportion of data for training
        val_split: Proportion of data for validation (test = 1 - train - val)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Download and load EuroSAT dataset (training set with augmentation)
    print("Loading EuroSAT dataset...")
    train_transform = get_transforms(is_training=True)
    eurosat_train = datasets.EuroSAT(
        root=data_root,
        download=True,
        transform=train_transform
    )
    
    # Create binary dataset wrapper
    binary_dataset_train = BinaryEuroSATDataset(eurosat_train)
    
    # Split dataset
    total_size = len(binary_dataset_train)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset splits:")
    print(f"  Training:   {train_size} samples")
    print(f"  Validation: {val_size} samples")
    print(f"  Test:       {test_size} samples")
    print(f"  Total:      {total_size} samples")
    
    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        binary_dataset_train,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # For validation and test, we need datasets without augmentation
    # Create separate datasets for val/test
    val_transform = get_transforms(is_training=False)
    eurosat_val = datasets.EuroSAT(
        root=data_root,
        download=False,  # Already downloaded
        transform=val_transform
    )
    binary_dataset_val = BinaryEuroSATDataset(eurosat_val)
    
    # Get the same indices for val and test
    val_dataset_clean = Subset(binary_dataset_val, val_dataset.indices)
    test_dataset_clean = Subset(binary_dataset_val, test_dataset.indices)
    
    # Print class distribution
    dist = binary_dataset_train.get_class_distribution()
    print(f"\nClass distribution:")
    print(f"  City:     {dist['City']} samples ({dist['City']/total_size*100:.1f}%)")
    print(f"  Farmland: {dist['Farmland']} samples ({dist['Farmland']/total_size*100:.1f}%)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset_clean,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset_clean,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Computes class weights for handling class imbalance.
    
    Args:
        train_loader: Training data loader
    
    Returns:
        Tensor of class weights
    """
    # Count samples in each class
    class_counts = torch.zeros(2)
    
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Compute weights (inverse frequency)
    total_samples = class_counts.sum()
    class_weights = total_samples / (2 * class_counts)
    
    print(f"\nClass weights for loss function:")
    print(f"  City:     {class_weights[0]:.3f}")
    print(f"  Farmland: {class_weights[1]:.3f}")
    
    return class_weights


if __name__ == "__main__":
    # Test the dataset loading
    print("Testing dataset loading...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    
    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:10]}")
    
    # Get class weights
    weights = get_class_weights(train_loader)
