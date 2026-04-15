"""
E2E tests for analytics page workflow
"""
import pytest
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@pytest.mark.e2e
@pytest.mark.analytics
class TestAnalyticsWorkflow:
    """Test analytics page workflow"""
    
    def test_analytics_page_loads(self, driver, live_server):
        """Test that analytics page loads"""
        driver.get(f'{live_server}/analytics')
        
        wait = WebDriverWait(driver, 10)
        
        # Wait for page to load
        title = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, 'h1'))
        )
        
        assert 'Analytics' in title.text
    
    def test_overview_statistics_present(self, driver, live_server):
        """Test that overview statistics are present"""
        driver.get(f'{live_server}/analytics')
        
        total_count = driver.find_element(By.ID, 'totalCount')
        avg_confidence = driver.find_element(By.ID, 'avgConfidence')
        top_category = driver.find_element(By.ID, 'topCategory')

        assert total_count.is_displayed()
        assert avg_confidence.is_displayed()
        assert top_category.is_displayed()
    
    def test_navigation_back_to_classifier_present(self, driver, live_server):
        """Test that navigation back to the classifier is present"""
        driver.get(f'{live_server}/analytics')
        
        links = driver.find_elements(By.TAG_NAME, 'a')
        link_texts = [link.text for link in links]

        assert any('Classifier' in text or 'Back' in text for text in link_texts)
    
    def test_recent_classifications_list_displays(self, driver, live_server):
        """Test that the recent classifications list displays"""
        driver.get(f'{live_server}/analytics')
        
        wait = WebDriverWait(driver, 10)
        recent_list = wait.until(
            EC.presence_of_element_located((By.ID, 'recentList'))
        )
        
        assert recent_list.is_displayed()
    
    def test_charts_render(self, driver, live_server):
        """Test that Plotly charts render"""
        driver.get(f'{live_server}/analytics')
        
        wait = WebDriverWait(driver, 15)
        
        # Wait for charts to render
        time.sleep(3)
        
        # Check if chart containers exist
        category_chart = driver.find_element(By.ID, 'categoryChart')
        pie_chart = driver.find_element(By.ID, 'pieChart')
        
        assert category_chart.is_displayed()
        assert pie_chart.is_displayed()


@pytest.mark.e2e
@pytest.mark.analytics
@pytest.mark.smoke
class TestAnalyticsSmokeTests:
    """Smoke tests for analytics page"""
    
    def test_no_javascript_errors(self, driver, live_server):
        """Test that there are no JavaScript errors"""
        driver.get(f'{live_server}/analytics')
        
        # Wait for page to fully load
        time.sleep(3)
        
        # Get browser console logs
        logs = driver.get_log('browser')
        
        # Ignore transient third-party network failures from the Plotly CDN.
        errors = [
            log for log in logs
            if log['level'] == 'SEVERE'
            and 'plotly' not in log.get('message', '').lower()
            and 'favicon' not in log.get('message', '').lower()
        ]

        assert len(errors) == 0, f"JavaScript errors found: {errors}"
