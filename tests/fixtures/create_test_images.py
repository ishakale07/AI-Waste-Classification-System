"""
Create test images for E2E tests
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_images():
    """Create sample test images"""
    fixtures_dir = os.path.dirname(__file__)
    
    categories = ['plastic', 'metal', 'paper', 'organic']
    colors = {
        'plastic': (255, 100, 100),
        'metal': (100, 200, 200),
        'paper': (200, 200, 100),
        'organic': (100, 255, 100)
    }
    
    for category in categories:
        img = Image.new('RGB', (224, 224), colors[category])
        draw = ImageDraw.Draw(img)
        
        # Add text
        text = category.upper()
        draw.text((50, 100), text, fill=(0, 0, 0))
        
        # Save
        img.save(os.path.join(fixtures_dir, f'test_{category}.jpg'))
        print(f"Created test_{category}.jpg")

if __name__ == '__main__':
    create_test_images()