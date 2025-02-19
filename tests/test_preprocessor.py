import pytest
import numpy as np
import pandas as pd
import torch
import os
import shutil
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dataset import Preprocessor



# Mock data creation helper functions
def create_mock_excel(tmp_path):
    """Create a mock excel file with test data"""
    data = {
        'studyid': [1, 2, 3],
        'sustainedvt': [False, True, False],
        'cardiacarrest': [False, False, True],
        'scd': [True, False, False]
    }
    df = pd.DataFrame(data)
    excel_path = tmp_path / "test_labels.xlsx"
    df.to_excel(excel_path, index=False)
    return excel_path

def create_mock_images(tmp_path):
    """Create mock MRI image data"""
    for patient_id in [1, 2, 3]:
        patient_dir = tmp_path / str(patient_id)
        patient_dir.mkdir(exist_ok=True)
        
        # Create 6 slices for each patient
        for slice_no in range(6):
            # Create random image data of size 128x128
            img_data = np.random.rand(128, 128)
            np.save(patient_dir / f'raw_{slice_no}.npy', img_data)

@pytest.fixture
def test_data_directory(tmp_path):
    """Fixture to create and clean up test data"""
    # Create mock data
    excel_path = create_mock_excel(tmp_path)
    create_mock_images(tmp_path)
    
    yield tmp_path
    
    # Cleanup
    shutil.rmtree(tmp_path)

def test_preprocessor_shape(test_data_directory):
    """Test if the preprocessor outputs the correct shape"""
    # Initialize preprocessor with test data
    preprocessor = Preprocessor(
        images_file_path=str(test_data_directory),
        labels_filepath=str(test_data_directory / "test_labels.xlsx")
    )
    
    # Transform data with masks included
    X, y, keys = preprocessor.transform(n_slices=6, include_masks=True)
    
    # Test shapes
    assert isinstance(X, torch.Tensor), "X should be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "y should be a torch.Tensor"
    
    # Test X shape: (n_patients, 2, n_slices, 128, 128)
    expected_shape = (3, 6, 2, 128, 128)  # 3 patients, 2 channels (image + mask), 6 slices
    assert X.shape == expected_shape, f"Expected shape {expected_shape}, got {X.shape}"
    
    # Test y shape: (n_patients,)
    assert y.shape == (3,), f"Expected y shape (3,), got {y.shape}"
    
    # Test that keys match number of patients
    assert len(keys) == 3, f"Expected 3 keys, got {len(keys)}"

def test_preprocessor_no_masks(test_data_directory):
    """Test preprocessor without masks"""
    preprocessor = Preprocessor(
        images_file_path=str(test_data_directory),
        labels_filepath=str(test_data_directory / "test_labels.xlsx")
    )
    
    # Transform data without masks
    X, y, keys = preprocessor.transform(n_slices=6, include_masks=False)
    
    # Test X shape without masks: (n_patients, n_slices, 128, 128)
    expected_shape = (3, 6, 128, 128)
    assert X.shape == expected_shape, f"Expected shape {expected_shape}, got {X.shape}"

def test_preprocessor_missing_data(test_data_directory):
    """Test preprocessor behavior with missing slice data"""
    # Remove a slice from one patient to test handling of missing data
    os.remove(test_data_directory / "1" / "raw_0.npy")
    
    preprocessor = Preprocessor(
        images_file_path=str(test_data_directory),
        labels_filepath=str(test_data_directory / "test_labels.xlsx")
    )
    
    X, y, keys = preprocessor.transform(n_slices=6, include_masks=True)
    
    # Should only have 2 patients due to missing data
    expected_shape = (2, 6, 2, 128, 128)
    assert X.shape == expected_shape, f"Expected shape {expected_shape}, got {X.shape}"
    assert len(keys) == 2, "Expected 2 valid patients"

def test_preprocessor_composite_outcome(test_data_directory):
    """Test if composite outcome is calculated correctly"""
    preprocessor = Preprocessor(
        images_file_path=str(test_data_directory),
        labels_filepath=str(test_data_directory / "test_labels.xlsx")
    )
    
    _, y, _ = preprocessor.transform(n_slices=6)
    
    # Given our mock data, all patients should have at least one True value
    assert torch.all(y == True)

if __name__ == "__main__":
    pytest.main([__file__])