"""
Test script for pycomp metrics on random Gaussian data using pytest
"""

import numpy as np
import pytest
import sys
import os

# Add src to path to import pycomp - adjust path for test directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pycomp


@pytest.fixture
def gaussian_data():
    """Generate random Gaussian data (2000 x 30) for testing"""
    np.random.seed(42)  # For reproducibility
    return np.random.randn(2000, 30)


def test_data_generation(gaussian_data):
    """Test that the generated data has correct shape and properties"""
    assert gaussian_data.shape == (2000, 30)
    assert abs(np.mean(gaussian_data)) < 0.1  # Should be close to 0
    assert 0.9 < np.std(gaussian_data) < 1.1  # Should be close to 1


def test_pds_function(gaussian_data):
    """Test PDS function on random Gaussian data"""
    pds_result = pycomp.pds(gaussian_data)
    
    # Check that result is a float
    assert isinstance(pds_result, (float, np.floating))
    
    # Check that result is finite
    assert np.isfinite(pds_result)
    
    print(f"PDS result: {pds_result:.4f}")


def test_mnc_function(gaussian_data):
    """Test MNC function on random Gaussian data"""
    k = 10
    mnc_result = pycomp.mnc(gaussian_data, k=k)
    
    # Check that result is a float
    assert isinstance(mnc_result, (float, np.floating))
    
    # Check that result is finite
    assert np.isfinite(mnc_result)
    
    print(f"MNC result: {mnc_result:.4f}")


def test_both_functions(gaussian_data):
    """Test that both functions can be called successfully"""
    pds_result = pycomp.pds(gaussian_data)
    mnc_result = pycomp.mnc(gaussian_data, k=10)
    
    assert np.isfinite(pds_result)
    assert np.isfinite(mnc_result)
    
    print(f"Both functions executed successfully!")
    print(f"PDS: {pds_result:.4f}, MNC: {mnc_result:.4f}")