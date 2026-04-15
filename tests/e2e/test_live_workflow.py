"""
E2E tests for live camera mode workflow
"""
import pytest
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@pytest.mark.e2e
@pytest.mark.live
class TestLiveWorkflow:
    """Test live camera mode workflow"""
    
    def test_live_page_loads(self, driver, live_server):
        """Test that live camera page loads correctly"""
        driver.get(f'{live_server}/live')
        
        wait = WebDriverWait(driver, 10)
        
        # Verify page elements
        video = wait.until(
            EC.presence_of_element_located((By.ID, 'video'))
        )
        assert video is not None
        
        start_btn = driver.find_element(By.ID, 'startBtn')
        assert start_btn.is_displayed()
        assert start_btn.is_enabled()
    
    def test_settings_panel_present(self, driver, live_server):
        """Test that settings panel is present"""
        driver.get(f'{live_server}/live')
        
        confidence_slider = driver.find_element(By.ID, 'confidenceThreshold')
        interval_slider = driver.find_element(By.ID, 'predictionInterval')
        
        assert confidence_slider.is_displayed()
        assert interval_slider.is_displayed()
    
    def test_confidence_threshold_adjustable(self, driver, live_server):
        """Test adjusting confidence threshold"""
        driver.get(f'{live_server}/live')
        
        slider = driver.find_element(By.ID, 'confidenceThreshold')
        value_display = driver.find_element(By.ID, 'confidenceValue')
        
        initial_value = value_display.text
        
        # Use JavaScript to set slider value
        driver.execute_script("arguments[0].value = 75; arguments[0].dispatchEvent(new Event('input'));", slider)
        
        time.sleep(0.5)
        new_value = value_display.text
        
        assert new_value == '75'
        assert new_value != initial_value


@pytest.mark.e2e
@pytest.mark.live
class TestLiveUIElements:
    """Test UI elements on live page"""
    
    def test_navigation_to_upload_mode(self, driver, live_server):
        """Test navigation link to upload mode"""
        driver.get(f'{live_server}/live')
        
        links = driver.find_elements(By.TAG_NAME, 'a')
        upload_link = [link for link in links if 'Upload' in link.text][0]
        
        upload_link.click()
        time.sleep(1)
        
        # Should be on home page
        assert '/live' not in driver.current_url
    
    def test_fps_display_hidden_initially(self, driver, live_server):
        """Test that FPS display is hidden initially"""
        driver.get(f'{live_server}/live')
        
        fps_display = driver.find_element(By.ID, 'fpsDisplay')
        
        # Should be hidden (display: none)
        assert not fps_display.is_displayed()