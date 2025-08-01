"""Tests for confidence calibration module."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from llm_tab_cleaner.confidence import ConfidenceCalibrator


class TestConfidenceCalibrator:
    """Test suite for ConfidenceCalibrator class."""
    
    def test_init(self):
        """Test ConfidenceCalibrator initialization."""
        calibrator = ConfidenceCalibrator()
        assert calibrator is not None
        assert hasattr(calibrator, 'fit')
        assert hasattr(calibrator, 'calibrate')
    
    def test_fit_with_empty_data(self):
        """Test fit method with empty data."""
        calibrator = ConfidenceCalibrator()
        predictions = []
        ground_truth = []
        
        # Should handle empty data gracefully
        calibrator.fit(predictions, ground_truth)
    
    def test_fit_with_sample_data(self):
        """Test fit method with sample data."""
        calibrator = ConfidenceCalibrator()
        predictions = [0.9, 0.8, 0.7, 0.6]
        ground_truth = [True, True, False, False]
        
        calibrator.fit(predictions, ground_truth)
        
        # Should not raise any exceptions and should store training data
        assert calibrator is not None
    
    def test_calibrate_before_fit(self):
        """Test calibrate method before fitting."""
        calibrator = ConfidenceCalibrator()
        confidence = 0.8
        
        # Should return the original confidence if not fitted
        result = calibrator.calibrate(confidence)
        assert result == confidence
    
    def test_calibrate_after_fit(self):
        """Test calibrate method after fitting."""
        calibrator = ConfidenceCalibrator()
        predictions = [0.9, 0.8, 0.7, 0.6, 0.5]
        ground_truth = [True, True, True, False, False]
        
        calibrator.fit(predictions, ground_truth)
        
        # Should return calibrated confidence
        result = calibrator.calibrate(0.8)
        assert isinstance(result, (int, float))
        assert 0 <= result <= 1
    
    def test_calibrate_edge_cases(self):
        """Test calibrate method with edge cases."""
        calibrator = ConfidenceCalibrator()
        
        # Test with 0 confidence
        result = calibrator.calibrate(0.0)
        assert result == 0.0
        
        # Test with 1.0 confidence
        result = calibrator.calibrate(1.0)
        assert result == 1.0
        
        # Test with negative confidence (should be clamped)
        result = calibrator.calibrate(-0.1)
        assert result >= 0.0
        
        # Test with confidence > 1.0 (should be clamped)
        result = calibrator.calibrate(1.1)
        assert result <= 1.0
    
    def test_fit_with_mismatched_lengths(self):
        """Test fit method with mismatched array lengths."""
        calibrator = ConfidenceCalibrator()
        predictions = [0.9, 0.8]
        ground_truth = [True, True, False]  # Different length
        
        # Should handle mismatched lengths gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            calibrator.fit(predictions, ground_truth)
    
    def test_multiple_fit_calls(self):
        """Test multiple calls to fit method."""
        calibrator = ConfidenceCalibrator()
        
        # First fit
        predictions1 = [0.9, 0.8]
        ground_truth1 = [True, False]
        calibrator.fit(predictions1, ground_truth1)
        
        # Second fit (should update the calibrator)
        predictions2 = [0.7, 0.6]
        ground_truth2 = [True, True]
        calibrator.fit(predictions2, ground_truth2)
        
        # Should work without issues
        result = calibrator.calibrate(0.7)
        assert isinstance(result, (int, float))