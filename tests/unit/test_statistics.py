"""
Unit tests for statistics functions
"""
import pytest
from datetime import datetime


@pytest.mark.unit
class TestStatisticsTracking:
    """Test statistics tracking"""
    
    def test_update_statistics_increments_total(self):
        """Test that total predictions increment"""
        from app.app import stats, update_statistics
        
        initial_total = stats['total_predictions']
        update_statistics('plastic', 0.95, mode='upload')
        
        assert stats['total_predictions'] == initial_total + 1
    
    def test_update_statistics_upload_mode(self):
        """Test upload mode statistics"""
        from app.app import stats, update_statistics
        
        initial_upload = stats['upload_predictions']
        update_statistics('plastic', 0.95, mode='upload')
        
        assert stats['upload_predictions'] == initial_upload + 1
    
    def test_update_statistics_live_mode(self):
        """Test live mode statistics"""
        from app.app import stats, update_statistics
        
        initial_live = stats['live_predictions']
        update_statistics('metal', 0.87, mode='live')
        
        assert stats['live_predictions'] == initial_live + 1
    
    def test_confidence_stats_high(self):
        """Test high confidence tracking"""
        from app.app import stats, update_statistics
        
        initial_high = stats['confidence_stats']['high']
        update_statistics('plastic', 0.92, mode='upload')
        
        assert stats['confidence_stats']['high'] == initial_high + 1
    
    def test_confidence_stats_medium(self):
        """Test medium confidence tracking"""
        from app.app import stats, update_statistics
        
        initial_medium = stats['confidence_stats']['medium']
        update_statistics('paper', 0.65, mode='upload')
        
        assert stats['confidence_stats']['medium'] == initial_medium + 1
    
    def test_confidence_stats_low(self):
        """Test low confidence tracking"""
        from app.app import stats, update_statistics
        
        initial_low = stats['confidence_stats']['low']
        update_statistics('organic', 0.45, mode='live')
        
        assert stats['confidence_stats']['low'] == initial_low + 1