"""
Test script for pycomp metrics on all labeled datasets using pytest
"""

import numpy as np
import pytest
import sys
import os
import glob

# Add src to path to import pycomp - adjust path for test directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pycomp


def get_dataset_paths():
    """Get all data.npy file paths from test/labeled-datasets directory"""
    test_dir = os.path.dirname(__file__)
    labeled_datasets_dir = os.path.join(test_dir, 'labeled-datasets')
    
    # Find all data.npy files in subdirectories
    data_files = []
    for root, dirs, files in os.walk(labeled_datasets_dir):
        if 'data.npy' in files:
            data_files.append(os.path.join(root, 'data.npy'))
    
    return sorted(data_files)


@pytest.mark.parametrize("data_path", get_dataset_paths())
def test_pds_on_labeled_datasets(data_path):
    """Test PDS function on each labeled dataset"""
    dataset_name = os.path.basename(os.path.dirname(data_path))
    
    try:
        # Load the dataset
        data = np.load(data_path)
        print(f"\nTesting PDS on {dataset_name}: shape {data.shape}")
        
        # Test PDS function
        pds_result = pycomp.pds(data)
        
        # Assertions
        assert isinstance(pds_result, (float, np.floating)), f"PDS result should be float for {dataset_name}"
        assert np.isfinite(pds_result), f"PDS result should be finite for {dataset_name}"
        
        print(f"PDS result for {dataset_name}: {pds_result:.4f}")
        
    except Exception as e:
        pytest.fail(f"PDS function failed on {dataset_name}: {str(e)}")


@pytest.mark.parametrize("data_path", get_dataset_paths())
def test_mnc_on_labeled_datasets(data_path):
    """Test MNC function on each labeled dataset"""
    dataset_name = os.path.basename(os.path.dirname(data_path))
    
    try:
        # Load the dataset
        data = np.load(data_path)
        print(f"\nTesting MNC on {dataset_name}: shape {data.shape}")
        
        # Choose k based on dataset size (but not larger than n_samples - 1)
        k = min(10, data.shape[0] - 1)
        if k <= 0:
            pytest.skip(f"Dataset {dataset_name} too small for MNC (needs at least 2 samples)")
        
        # Test MNC function
        mnc_result = pycomp.mnc(data, k=k)
        
        # Assertions
        assert isinstance(mnc_result, (float, np.floating)), f"MNC result should be float for {dataset_name}"
        assert np.isfinite(mnc_result), f"MNC result should be finite for {dataset_name}"
        
        print(f"MNC result for {dataset_name} (k={k}): {mnc_result:.4f}")
        
    except Exception as e:
        pytest.fail(f"MNC function failed on {dataset_name}: {str(e)}")


def test_dataset_loading():
    """Test that all datasets can be loaded successfully"""
    data_paths = get_dataset_paths()
    print(f"\nFound {len(data_paths)} datasets")
    
    loaded_count = 0
    for data_path in data_paths:
        dataset_name = os.path.basename(os.path.dirname(data_path))
        try:
            data = np.load(data_path)
            print(f"✓ {dataset_name}: {data.shape}")
            loaded_count += 1
        except Exception as e:
            print(f"✗ {dataset_name}: Failed to load - {str(e)}")
    
    assert loaded_count > 0, "No datasets could be loaded"
    print(f"\nSuccessfully loaded {loaded_count}/{len(data_paths)} datasets")