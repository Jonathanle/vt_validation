


import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Assuming datasetsplitter.create_fold_dataloaders is in a file named data_utils.py
from trainer import DatasetSplitter

class MockDataset(Dataset):
    """Mock dataset with imbalanced classes for testing"""
    def __init__(self, num_samples=1000, class_ratios=[0.7, 0.2, 0.1]):
        # Create imbalanced dataset
        self.num_classes = len(class_ratios)
        samples_per_class = [int(num_samples * ratio) for ratio in class_ratios]
        
        # Ensure we have exactly num_samples due to rounding
        samples_per_class[-1] += num_samples - sum(samples_per_class)
        
        # Create features and labels
        features = []
        labels = []
        
        for class_idx, num_class_samples in enumerate(samples_per_class):
            features.extend([torch.randn(10) for _ in range(num_class_samples)])
            labels.extend([class_idx] * num_class_samples)
        
        self.features = torch.stack(features)
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

@pytest.fixture
def mock_dataset():
    """Fixture that creates a dataset with known imbalanced classes"""
    return MockDataset(num_samples=1000, class_ratios=[0.7, 0.2, 0.1])

@pytest.fixture
def small_mock_dataset():
    """Smaller dataset for detailed testing"""
    return MockDataset(num_samples=100, class_ratios=[0.6, 0.3, 0.1])


@pytest.fixture
def dataset_splitter():
    return DatasetSplitter()



def test_number_of_folds(mock_dataset, dataset_splitter):
    """Test if correct number of fold pairs are created"""
    n_splits = 5
    fold_loaders = dataset_splitter.create_fold_dataloaders(mock_dataset, n_splits=n_splits)
    assert len(fold_loaders) == n_splits
    
    # Check each fold contains train and val loader
    for train_loader, val_loader in fold_loaders:
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

def test_fold_sizes(mock_dataset, dataset_splitter):
    """Test if folds are roughly equal in size"""
    n_splits = 5
    fold_loaders = dataset_splitter.create_fold_dataloaders(mock_dataset, n_splits=n_splits)
    
    val_sizes = []
    train_sizes = []
    
    for train_loader, val_loader in fold_loaders:
        val_sizes.append(len(val_loader.sampler))
        train_sizes.append(len(train_loader.sampler))
    
    # Check validation splits are roughly equal
    val_size_diff = max(val_sizes) - min(val_sizes)
    assert val_size_diff <= 1, f"Validation fold sizes differ by {val_size_diff}"
    
    # Check training splits are roughly equal
    train_size_diff = max(train_sizes) - min(train_sizes)
    assert train_size_diff <= 1, f"Training fold sizes differ by {train_size_diff}"
    
    # Check that train + val equals total dataset size
    for train_size, val_size in zip(train_sizes, val_sizes):
        assert train_size + val_size == len(mock_dataset)

def test_stratification(small_mock_dataset, dataset_splitter):
    """Test if classes are stratified properly across folds"""
    n_splits = 5
    fold_loaders = dataset_splitter.create_fold_dataloaders(small_mock_dataset, n_splits=n_splits)
    
    # Get original class distribution
    all_labels = [small_mock_dataset[i][1].item() for i in range(len(small_mock_dataset))]
    original_dist = Counter(all_labels)
    original_ratios = {k: v/len(all_labels) for k, v in original_dist.items()}
    
    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        # Get validation set labels
        val_labels = []
        for batch_idx, (_, labels) in enumerate(val_loader):
            val_labels.extend(labels.tolist())
        
        val_dist = Counter(val_labels)
        val_ratios = {k: v/len(val_labels) for k, v in val_dist.items()}
