import sys
import os 
import numpy as np
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import Evaluator
from scoring_model import CNNScorer
"""
note with attention based testing - i can create similar rpoblems really faster 
than debugging and repeat used environments9

i dont want ot be learning about AI and have problems then realize that 

it was an infrastructural issue that I had. 
replication of environments - docuemtnation + confidence

for thigns of epistemic concern + consideration for long term goals, consider using testing to verify


# df problem using pytest parametersization.

"""
"""
@pytest.fixture(scope = 'function'):
def target_pred_values():
    predictions = np.array([1, 1, 0, 0,])
    values = 
"""


def test_dataloader_shape_probability_values(cnn_model, evaluator, train_dataloader):
    evaluate_object = evaluator(cnn_model) # fixture generator handles cuda 

    pred, target = evaluate_object._get_predictions(train_dataloader)

    assert len(pred.shape) == 1  # np array has shape (n,)
    assert pred.shape == target.shape # the shapes must be equal.



def test_dataloader_shape(cnn_model, evaluator, train_dataloader):
    evaluate_object = evaluator(cnn_model) # fixture generator handles cuda 

    pred, target = evaluate_object._get_predictions(train_dataloader)


    assert len(pred.shape) == 1  # np array has shape (n,)
    assert pred.shape == target.shape # the shapes must be equal.


def test_dataloader_same_dtype(cnn_model, evaluator, train_dataloader):
    evaluate_object = evaluator(cnn_model) # fixture generator handles cuda 

    pred, target = evaluate_object._get_predictions(train_dataloader)

    assert pred.dtype == np.float32 # i learned here that the dtypes from torch are float32
    assert pred.dtype == target.dtype # they both have matching dtypes somehow for target


def test_get_metrics(cnn_model, evaluator, metric_test_cases):
    for test_case in metric_test_cases:
        predictions = test_case['predictions']
        targets = test_case['targets']
        expected = test_case['expected']

        evaluate_object = evaluator(cnn_model) 
        # Call the function
        auc, accuracy, tp, fn, tn, fp = evaluate_object._get_metrics(predictions, targets)
        
        # Assert all metrics
        assert np.isclose(auc, expected['auc'], atol=1e-7)
        assert np.isclose(accuracy, expected['accuracy'], atol=1e-7)
        assert tp == expected['true_positives']
        assert fn == expected['false_negatives']
        assert tn == expected['true_negatives']
        assert fp == expected['false_positives']

def test_get_constant(cnn_model, evaluator, constant_predictions_case):
    test_case = constant_predictions_case 
    predictions = test_case['predictions']
    targets = test_case['targets']
    expected = test_case['expected']

    evaluate_object = evaluator(cnn_model) 
    # Call the function
    auc, accuracy, tp, fn, tn, fp = evaluate_object._get_metrics(predictions, targets)
    
    # Assert all metrics
    assert np.isclose(auc, expected['auc'], atol=1e-7)
    assert np.isclose(accuracy, expected['accuracy'], atol=1e-7)
    assert tp == expected['true_positives']
    assert fn == expected['false_negatives']
    assert tn == expected['true_negatives']
    assert fp == expected['false_positives']
