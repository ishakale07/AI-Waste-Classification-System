"""
Unit tests for model utility functions
"""
import pytest
import numpy as np
from PIL import Image


@pytest.mark.unit
class TestImagePreprocessing:
    """Test image preprocessing functions"""
    
    def test_preprocess_image_correct_shape(self, create_test_image):
        """Test that preprocessed image has correct shape"""
        from app.app import preprocess_image
        
        img_bytes = create_test_image()
        img = Image.open(img_bytes)
        processed = preprocess_image(img)
        
        assert processed.shape == (1, 224, 224, 3)
    
    def test_preprocess_image_normalized(self, create_test_image):
        """Test that image is normalized to 0-1 range"""
        from app.app import preprocess_image
        
        img_bytes = create_test_image()
        img = Image.open(img_bytes)
        processed = preprocess_image(img)
        
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0
    
    def test_preprocess_converts_rgba_to_rgb(self):
        """Test that RGBA images are converted to RGB"""
        from app.app import preprocess_image
        
        # Create RGBA image
        img = Image.new('RGBA', (224, 224), (255, 0, 0, 128))
        processed = preprocess_image(img)
        
        # Should have 3 channels (RGB)
        assert processed.shape[-1] == 3


@pytest.mark.unit
class TestDisposalInfo:
    """Test disposal information retrieval"""
    
    def test_get_disposal_info_plastic(self):
        """Test disposal info for plastic"""
        from app.app import get_disposal_info
        
        info = get_disposal_info('plastic')
        
        assert info['bin'] == 'Yellow/Blue Recycling Bin'
        assert 'tips' in info
        assert isinstance(info['tips'], list)
        assert len(info['tips']) > 0
    
    def test_get_disposal_info_all_categories(self):
        """Test that all categories have disposal info"""
        from app.app import get_disposal_info
        
        categories = ['plastic', 'metal', 'organic', 'paper', 'glass', 
                     'cardboard', 'textile', 'vegetation', 'miscellaneous']
        
        for category in categories:
            info = get_disposal_info(category)
            assert 'bin' in info
            assert 'tips' in info
            assert 'examples' in info
    
    def test_get_disposal_info_unknown_category(self):
        """Test disposal info for unknown category returns default"""
        from app.app import get_disposal_info
        
        info = get_disposal_info('unknown_category')
        
        assert info['bin'] == 'General Waste Bin'