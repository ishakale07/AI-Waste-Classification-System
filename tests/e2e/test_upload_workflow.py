"""
E2E tests for upload mode workflow
"""
import pytest
import time
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@pytest.mark.e2e
@pytest.mark.upload
class TestUploadWorkflow:
    """Test complete upload workflow"""
    
    def test_upload_image_complete_flow(self, driver, live_server):
        """Test uploading an image and viewing results"""
        # Navigate to home page
        driver.get(live_server)
        
        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        upload_area = wait.until(
            EC.presence_of_element_located((By.ID, 'uploadArea'))
        )
        
        # Find file input
        file_input = driver.find_element(By.ID, 'fileInput')
        
        # Upload test image
        test_image_path = os.path.abspath('tests/fixtures/test_plastic.jpg')
        file_input.send_keys(test_image_path)
        
        # Wait for results
        preview_section = wait.until(
            EC.visibility_of_element_located((By.ID, 'previewSection'))
        )
        
        # Verify results are displayed
        predicted_category = driver.find_element(By.ID, 'predictedCategory')
        assert predicted_category.text != '-'
        
        confidence_badge = driver.find_element(By.ID, 'confidenceBadge')
        assert 'Confident' in confidence_badge.text
        
        print(f"✓ Prediction: {predicted_category.text}")
        print(f"✓ Confidence: {confidence_badge.text}")
    
    def test_upload_another_image(self, driver, live_server):
        """Test uploading multiple images"""
        driver.get(live_server)
        wait = WebDriverWait(driver, 10)
        
        # First upload
        file_input = driver.find_element(By.ID, 'fileInput')
        test_image = os.path.abspath('tests/fixtures/test_plastic.jpg')
        file_input.send_keys(test_image)
        
        # Wait for results
        wait.until(EC.visibility_of_element_located((By.ID, 'previewSection')))
        
        # Click "Upload Another"
        reset_buttons = driver.find_elements(By.TAG_NAME, 'button')
        upload_another_btn = [btn for btn in reset_buttons if 'Another' in btn.text][0]
        upload_another_btn.click()
        
        # Verify upload area is visible again
        upload_area = wait.until(
            EC.visibility_of_element_located((By.ID, 'uploadArea'))
        )
        assert upload_area.is_displayed()


@pytest.mark.e2e
@pytest.mark.upload
class TestUploadUIElements:
    """Test UI elements on upload page"""
    
    def test_page_title_present(self, driver, live_server):
        """Test that page title is correct"""
        driver.get(live_server)
        assert 'AI Waste Classifier' in driver.title
    
    def test_navigation_links_present(self, driver, live_server):
        """Test that navigation links are present"""
        driver.get(live_server)
        
        # Find navigation links
        links = driver.find_elements(By.TAG_NAME, 'a')
        link_texts = [link.text for link in links]
        
        # Should have link to live mode and analytics
        assert any('Live' in text or 'Camera' in text for text in link_texts)
        assert any('Analytics' in text for text in link_texts)
    
    def test_upload_area_clickable(self, driver, live_server):
        """Test that upload area is clickable"""
        driver.get(live_server)
        
        upload_area = driver.find_element(By.ID, 'uploadArea')
        assert upload_area.is_displayed()
        assert upload_area.is_enabled()