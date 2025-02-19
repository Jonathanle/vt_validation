# conftest.py
import pytest
import sys
import os
import torch


import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import Subset, DataLoader
"""
motto - certainty reduces cognitive load when debugging


conftest - file for defining fixtures and things to make. here you define the concerns.


advantages of test - long term memory and automated methods of testing
bedrock verification to work on / less cognitive load on considering potential for errors , less attention 
"""
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import Evaluator

from dataset import Preprocessor, LGEDataset
from scoring_model import CNNScorer


def add(x , y):
    return x + y


@pytest.fixture(scope = 'module')
def evaluator():

    def generate_evaluator(model):
        evaluator = Evaluator(model, "cuda:0")
        return evaluator
    return generate_evaluator


@pytest.fixture(scope = 'module')
def train_dataloader():
    # Code Requires ./dataprocessed mri_data repository
    pp = Preprocessor()
    X, y, ids = pp.transform(3)

    dataset = LGEDataset(X, y, ids)

    indices = list(range(len(dataset)))
    labels = [dataset[i][1] for i in indices]

    train_idx, test_idx = train_test_split(
        indices,
        stratify = labels, 
        test_size = 0.3,
        random_state = 42
    )


    train_dataset = Subset(dataset, train_idx)

    train_dataloader = DataLoader(train_dataset, batch_size = 10)

    return train_dataloader



@pytest.fixture(scope = 'module')
def cnn_model():
    model =  CNNScorer()
    model.to("cuda:0")

    return model
    # consider using yield to get the function out

@pytest.fixture(scope = 'function')
def X_tensor_cuda():

    X = torch.zeros(1, 3, 128, 128).to("cuda:0")
    return X

@pytest.fixture(scope = 'function')
def X_tensor():
    X = torch.zeros(1, 1, 128, 128)

    return X


@pytest.fixture(scope = 'module')
def X_tensor_generate():
    def generate_tensor(n_slices):
        X = torch.zeros(1, n_slices, 128, 128)
        return X
    return generate_tensor

@pytest.fixture
def metric_test_cases():
    return [
        # (predictions, targets, expected_outputs)
        # Perfect prediction case
        {
            'predictions': np.array([0.9, 0.1, 0.9, 0.1]),
            'targets': np.array([1, 0, 1, 0]),
            'expected': {
                'auc': 1.0,
                'accuracy': 1.0,
                'true_positives': 2,
                'false_negatives': 0,
                'true_negatives': 2,
                'false_positives': 0
            }
        },
        # Completely wrong predictions
        {
            'predictions': np.array([0.1, 0.9, 0.1, 0.9]),
            'targets': np.array([1, 0, 1, 0]),
            'expected': {
                'auc': 0.0,
                'accuracy': 0.0,
                'true_positives': 0,
                'false_negatives': 2,
                'true_negatives': 0,
                'false_positives': 2
            }
        },
        # Mixed case
        {
            'predictions': np.array([0.7, 0.3, 0.4, 0.8]),
            'targets': np.array([1, 0, 1, 0]),
            'expected': {
                'auc': 0.5,
                'accuracy': 0.5,
                'true_positives': 1,
                'false_negatives': 1,
                'true_negatives': 1,
                'false_positives': 1
            }
        }
    ]

@pytest.fixture
def constant_predictions_case():
    return {
        'predictions': np.array([0.6, 0.6, 0.6, 0.6]),
        'targets': np.array([1, 0, 1, 0]),
        'expected': {
            'auc': 0.5,
            'accuracy': 0.5,
            'true_positives': 2,
            'false_negatives': 0,
            'true_negatives': 0,
            'false_positives': 2
        }
    }

# Miscellaneous funtions  / learning actions testing what NOT to do


# Shared fixtures
@pytest.fixture(scope ='function')
def sample_data():
    return {
        "name": "test",
        "value": 42,
        "value2": 10,
    }

@pytest.fixture(scope ='module')
def numpy_array():
    list1 = ['a', 'b', 'c']

    return list1




# Test DO NOT ADD tests here this increase the complexity of a modlue
def test_addition(sample_data):

    x = sample_data['value']
    y = sample_data['value2'] 
    assert add(x,y) == 52
