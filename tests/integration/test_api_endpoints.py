"""
Integration tests for API endpoints
"""
import pytest
import json
import io
from PIL import Image


@pytest.mark.integration
class TestPredictEndpoint:
    """Test /predict endpoint"""
    
    def test_predict_with_valid_image(self, client, create_test_image):
        """Test prediction with valid image"""
        img_bytes = create_test_image()
        
        response = client.post(
            '/predict',
            data={'file': (img_bytes, 'test.jpg')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
        assert 'predicted_class' in data
        assert 'confidence' in data
    
    def test_predict_without_file(self, client):
        """Test prediction without file"""
        response = client.post('/predict')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] == False
    
    def test_predict_with_empty_filename(self, client):
        """Test prediction with empty filename"""
        response = client.post(
            '/predict',
            data={'file': (io.BytesIO(b''), '')},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400


@pytest.mark.integration
class TestPredictFrameEndpoint:
    """Test /predict_frame endpoint for live mode"""
    
    def test_predict_frame_with_valid_base64(self, client, create_test_image):
        """Test frame prediction with valid base64 image"""
        import base64
        
        img_bytes = create_test_image()
        img_b64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{img_b64}"
        
        response = client.post(
            '/predict_frame',
            data=json.dumps({'image': image_data}),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] == True
    
    def test_predict_frame_without_image_data(self, client):
        """Test frame prediction without image data"""
        response = client.post(
            '/predict_frame',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code == 400


@pytest.mark.integration
class TestStatsEndpoint:
    """Test /stats endpoint"""
    
    def test_stats_returns_json(self, client):
        """Test that stats endpoint returns JSON"""
        response = client.get('/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, dict)
    
    def test_stats_has_required_fields(self, client):
        """Test that stats has all required fields"""
        response = client.get('/stats')
        data = json.loads(response.data)
        
        assert 'total_predictions' in data
        assert 'recent_predictions' in data
        assert 'category_counts' in data


@pytest.mark.integration
class TestPageEndpoints:
    """Test page rendering endpoints"""
    
    def test_index_page_loads(self, client):
        """Test that index page loads"""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_live_page_loads(self, client):
        """Test that live camera page loads"""
        response = client.get('/live')
        assert response.status_code == 200
    
    def test_analytics_page_loads(self, client):
        """Test that analytics page loads"""
        response = client.get('/analytics')
        assert response.status_code == 200