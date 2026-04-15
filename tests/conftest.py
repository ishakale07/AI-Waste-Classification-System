"""
Pytest configuration and fixtures - Windows Compatible
"""
import pytest
import os
import sys
import time
import threading
import socket
import tempfile
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# ============================================================================
# Flask App Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def app():
    """Create Flask app for testing"""
    try:
        from app.app import app as flask_app
    except ImportError:
        import importlib.util
        app_path = os.path.join(project_root, 'app', 'app.py')
        spec = importlib.util.spec_from_file_location("app_module", app_path)
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        flask_app = app_module.app
    
    flask_app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'DEBUG': False,
        'UPLOAD_FOLDER': os.path.join(project_root, 'tests', 'fixtures', 'uploads')
    })
    
    os.makedirs(flask_app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    yield flask_app
    
    # Cleanup
    import shutil
    if os.path.exists(flask_app.config['UPLOAD_FOLDER']):
        try:
            shutil.rmtree(flask_app.config['UPLOAD_FOLDER'])
        except:
            pass


@pytest.fixture(scope='function')
def client(app):
    """Create test client"""
    with app.test_client() as client:
        yield client


# ============================================================================
# Selenium Fixtures - Windows Compatible
# ============================================================================

@pytest.fixture(scope='function')
def chrome_options():
    """Chrome options for Selenium"""
    options = Options()
    options.add_argument('--headless=new')  # New headless mode
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-first-run')
    options.add_argument('--no-default-browser-check')
    options.add_argument('--disable-background-networking')
    options.add_argument('--disable-background-timer-throttling')
    options.add_argument('--disable-renderer-backgrounding')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-extensions')
    options.add_argument('--remote-debugging-port=0')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
    
    return options


@pytest.fixture(scope='function')
def driver(chrome_options):
    """Create Selenium WebDriver - Windows Compatible"""
    driver = None
    profile_dir = tempfile.mkdtemp(prefix='chrome-test-profile-', dir=project_root)
    chrome_options.add_argument(f'--user-data-dir={profile_dir}')
    
    try:
        try:
            # Try Method 1: webdriver-manager (auto-downloads correct driver)
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service as ChromeService
            
            print("Installing ChromeDriver...")
            driver_path = ChromeDriverManager().install()
            print(f"ChromeDriver installed at: {driver_path}")
            
            service = ChromeService(driver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
        except Exception as e:
            print(f"webdriver-manager failed: {e}")
            
            try:
                # Method 2: Try using Chrome directly (if chromedriver in PATH)
                driver = webdriver.Chrome(options=chrome_options)
                
            except Exception as e2:
                print(f"Direct Chrome failed: {e2}")
                
                # Method 3: Skip test if Chrome not available
                pytest.skip(f"Chrome WebDriver not available. Error: {e2}")
        
        if driver:
            driver.implicitly_wait(10)
            driver.set_page_load_timeout(30)
            
            yield driver
            
            try:
                driver.quit()
            except:
                pass
        else:
            pytest.skip("Could not create Chrome WebDriver")
    finally:
        try:
            shutil.rmtree(profile_dir, ignore_errors=True)
        except:
            pass


@pytest.fixture(scope='function')
def live_server(app):
    """Start Flask test server"""
    class LiveServerURL(str):
        """String-like live server URL that also satisfies pytest-flask."""

        def __new__(cls, url, flask_app):
            obj = str.__new__(cls, url)
            obj.app = flask_app
            return obj

    def run_app():
        app.run(host='127.0.0.1', port=5555, debug=False, use_reloader=False)
    
    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()
    
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', 5555), timeout=1):
                break
        except OSError:
            time.sleep(0.2)
    else:
        pytest.fail("Flask live server did not start on http://127.0.0.1:5555 within 10 seconds")
    
    yield LiveServerURL('http://127.0.0.1:5555', app)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def create_test_image():
    """Create a test image"""
    from PIL import Image
    import io
    
    def _create_image(format='JPEG', size=(224, 224), color=(255, 0, 0)):
        img = Image.new('RGB', size, color)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes
    
    return _create_image
