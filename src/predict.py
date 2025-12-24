import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
import sys

def predict_image(image_path, model_path='data/models/best_model.h5'):
    # Load model and labels
    model = keras.models.load_model(model_path)
    
    with open('data/models/class_labels.json', 'r') as f:
        class_labels = json.load(f)
    
    # Load and preprocess image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_idx]
    confidence = predictions[predicted_idx]
    
    # Display results
    print("\n" + "="*50)
    print(f"Image: {image_path}")
    print("="*50)
    print(f"✓ Predicted: {predicted_class.upper()}")
    print(f"✓ Confidence: {confidence*100:.2f}%")
    print("\nAll predictions:")
    for i, label in enumerate(class_labels):
        print(f"  {label:15s}: {predictions[i]*100:5.2f}%")
    print("="*50)
    
    return predicted_class, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        print("Example: python src/predict.py test_image.jpg")
    else:
        predict_image(sys.argv[1])