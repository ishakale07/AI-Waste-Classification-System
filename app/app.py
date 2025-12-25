from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import json
from datetime import datetime
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = 'data/models/best_model.h5'
LABELS_PATH = 'data/models/class_labels.json'

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"WARNING: Model not found at {MODEL_PATH}")
    print("Please train your model first using: python src/train.py")
    model = None
    class_labels = []
else:
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
        
        # Load class labels
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        print(f"✓ Loaded {len(class_labels)} classes: {class_labels}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None
        class_labels = []

# Statistics storage
stats = {
    'total_predictions': 0,
    'category_counts': {label: 0 for label in class_labels},
    'recent_predictions': []
}

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def get_disposal_info(category):
    """Get disposal instructions for each category"""
    disposal_info = {
        'plastic': {
            'color': '#FF6B6B',
            'icon': '♻️',
            'bin': 'Yellow/Blue Recycling Bin',
            'tips': [
                'Clean and dry before recycling',
                'Remove caps and labels if possible',
                'Crush bottles to save space'
            ],
            'examples': 'Bottles, containers, bags, packaging'
        },
        'metal': {
            'color': '#4ECDC4',
            'icon': '🔩',
            'bin': 'Blue Recycling Bin',
            'tips': [
                'Rinse cans before recycling',
                'Aluminum foil can be recycled if clean',
                'Crush cans to save space'
            ],
            'examples': 'Cans, foil, metal containers'
        },
        'organic': {
            'color': '#95E1D3',
            'icon': '🌱',
            'bin': 'Green Compost Bin',
            'tips': [
                'Compost fruit and vegetable scraps',
                'Avoid meat and dairy in home compost',
                'Use for garden fertilizer'
            ],
            'examples': 'Food scraps, peels, plant materials'
        },
        'paper': {
            'color': '#F38181',
            'icon': '📄',
            'bin': 'Blue Recycling Bin',
            'tips': [
                'Keep paper dry and clean',
                'Remove plastic windows from envelopes',
                'Flatten boxes before recycling'
            ],
            'examples': 'Newspapers, cardboard, office paper'
        },
        'glass': {
            'color': '#A8E6CF',
            'icon': '🍶',
            'bin': 'Green Recycling Bin',
            'tips': [
                'Rinse containers before recycling',
                'Remove caps and lids',
                'Keep separate from other recyclables'
            ],
            'examples': 'Bottles, jars, containers'
        },
        'cardboard': {
            'color': '#FFD93D',
            'icon': '📦',
            'bin': 'Blue Recycling Bin',
            'tips': [
                'Flatten boxes to save space',
                'Remove tape and labels',
                'Keep dry and clean'
            ],
            'examples': 'Boxes, packaging, tubes'
        },
        'textile': {
            'color': '#C7B8EA',
            'icon': '👕',
            'bin': 'Textile Recycling Bin',
            'tips': [
                'Donate wearable clothes',
                'Recycle torn textiles separately',
                'Keep clean and dry'
            ],
            'examples': 'Clothes, fabric scraps, linens'
        },
        'vegetation': {
            'color': '#90EE90',
            'icon': '🌿',
            'bin': 'Green Waste Bin',
            'tips': [
                'Compost yard waste',
                'Mulch for garden use',
                'Keep separate from food waste'
            ],
            'examples': 'Leaves, grass, branches'
        },
        'miscellaneous': {
            'color': '#CCCCCC',
            'icon': '🗑️',
            'bin': 'General Waste Bin',
            'tips': [
                'Check if items can be recycled',
                'Dispose according to local guidelines',
                'Consider donation if usable'
            ],
            'examples': 'Mixed materials, non-recyclables'
        }
    }
    
    return disposal_info.get(category, {
        'color': '#CCCCCC',
        'icon': '🗑️',
        'bin': 'General Waste Bin',
        'tips': ['Check local disposal guidelines'],
        'examples': 'General waste items'
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    try:
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        # Get all predictions
        all_predictions = [
            {
                'category': class_labels[i],
                'confidence': float(predictions[i]),
                'percentage': f"{float(predictions[i])*100:.2f}"
            }
            for i in range(len(predictions))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get disposal information
        disposal_info = get_disposal_info(predicted_class)
        
        # Update statistics
        stats['total_predictions'] += 1
        stats['category_counts'][predicted_class] += 1
        stats['recent_predictions'].insert(0, {
            'category': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        stats['recent_predictions'] = stats['recent_predictions'][:10]
        
        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{predicted_class}_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_percentage': f"{confidence*100:.2f}",
            'all_predictions': all_predictions,
            'disposal_info': disposal_info,
            'image_path': f'/uploads/{filename}'
        })
        
    except Exception as e:
        print(f"ERROR in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats')
def get_stats():
    return jsonify(stats)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if model is None:
        return jsonify({
            'success' : 'False',
            'error' : 'Model not loaded'
        }), 500
    
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'success' : 'False',
                'error' : 'No image data provided'
            }), 400
        
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose = 0)[0]
        predicted_class_idx= np.argmax(predictions)
        predicted_class = class_labels[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])

        top_3_predictions = [
            {
                'category' : class_labels[i],
                'confidence' : float(predictions[i]),
                'percentage' : f"{float(predictions[i])*100:.1f}"
            }
            for i in np.argsort(predictions)[-3:][::-1]
        ]

        disposal_info = get_disposal_info(predicted_class)

        return jsonify({
            'success' : True,
            'predicted_class' : predicted_class,
            'confidence' : confidence,
            'confidence_percentage' : f"{confidence*100:.1f}",
            'top_predictions' : top_3_predictions,
            'disposal_info' : disposal_info
        })
    
    except Exception as e:
        print(f"ERROR in frame prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success' : False,
            'error' : str(e),
        }), 500

@app.route('/live')
def live():
    return render_template('live.html')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AI WASTE CLASSIFIER - WEB APPLICATION")
    print("="*60)
    print(f"Model loaded: {model is not None}")
    print(f"Classes: {class_labels}")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("\nStarting server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='127.0.0.1')


