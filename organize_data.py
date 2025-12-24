# organize_data.py
import os
import shutil
from sklearn.model_selection import train_test_split
import random

def organize_dataset(source_dir, dest_dir):
    """
    Organize dataset into train/validation/test splits
    """
    folder_mapping = {
        '1-Cardboard': 'cardboard',
        '2-Food Organics': 'organic',
        '3-Glass': 'glass',
        '4-Metal': 'metal',
        '5-Miscellaneous Trash': 'miscellaneous',
        '6-Paper': 'paper',
        '7-Plastic': 'plastic',
        '8-Textile Trash': 'textile',
        '9-Vegetation': 'vegetation'
    }
    
    categories = list(folder_mapping.values())
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for category in categories:
            os.makedirs(f'{dest_dir}/{split}/{category}', exist_ok=True)
    
    print("Created directory structure...")
    
    # Process each folder
    for folder_name, category in folder_mapping.items():
        category_path = os.path.join(source_dir, folder_name)
        
        if not os.path.exists(category_path):
            print(f"Warning: Folder not found: {category_path}")
            continue
        
        # Get all image files
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(images) == 0:
            print(f"Warning: No images found in {folder_name}")
            continue
        
        print(f"\nProcessing {folder_name} ({len(images)} images)...")
        
        # Shuffle images
        random.shuffle(images)
        
        # 70% train, 15% validation, 15% test
        train_split = int(0.7 * len(images))
        val_split = int(0.85 * len(images))
        
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]
        
        print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        # Copy files to train
        for img in train_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(dest_dir, 'train', category, img)
            shutil.copy2(src, dst)
        
        # Copy files to validation
        for img in val_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(dest_dir, 'val', category, img)
            shutil.copy2(src, dst)
        
        # Copy files to test
        for img in test_images:
            src = os.path.join(category_path, img)
            dst = os.path.join(dest_dir, 'test', category, img)
            shutil.copy2(src, dst)
    
    print("\n" + "="*50)
    print("Dataset organized successfully!")
    print("="*50)
    
    # Print summary
    print("\nSummary:")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dest_dir, split)
        total = 0
        for category in categories:
            cat_path = os.path.join(split_path, category)
            if os.path.exists(cat_path):
                count = len(os.listdir(cat_path))
                total += count
                print(f"  {split}/{category}: {count} images")
        print(f"  {split.upper()} TOTAL: {total} images\n")

# Run this
if __name__ == "__main__":
    source_directory = 'data/raw'  # Where your numbered folders are
    destination_directory = 'data/processed'  # Where organized data will go
    
    print("Starting dataset organization...")
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")
    print("="*50)
    
    organize_dataset(source_directory, destination_directory)