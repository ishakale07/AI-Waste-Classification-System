# AI Waste Classification System

A web-based application that uses deep learning to automatically classify waste items into categories (plastic, metal, organic, paper, etc.) to promote proper waste disposal and recycling.

## Features

- **Real-time Camera Classification**: Point your camera at waste items for instant live classification
- **Image Upload Classification**: Upload images and get instant predictions
- **High Accuracy**: 85-95% classification accuracy using transfer learning (MobileNetV2)
- **Confidence Scores**: See detailed prediction probabilities for all categories
- **Disposal Guidelines**: Get specific instructions for proper waste disposal
- **Analytics Dashboard**: Track classification statistics and trends
- **Adjustable Settings**: Configure confidence threshold and prediction intervals for live mode
- **User-Friendly Interface**: Modern, responsive web design with mobile support
- **Multi-Mode Operation**: Switch between upload mode and live camera mode

## Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras, Transfer Learning (MobileNetV2)
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Plotly.js
- **Image Processing**: OpenCV, PIL
- **Real-time Processing**: WebRTC for camera access, Canvas API for frame capture

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/waste-classifier.git
cd waste-classifier
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the trained model:
   - Place `best_model_finetuned.h5` in `data/models/`
   - Place `class_labels.json` in `data/models/`

## Usage

1. Start the Flask server:
```bash
python app/app.py
```

2. Open browser and navigate to:
```
http://localhost:5000
```

### Upload Mode

1. Navigate to the home page
2. Click or drag-and-drop an image of a waste item
3. View classification results with confidence scores
4. See disposal instructions

### Live Camera Mode

1. Click "📹 Live Camera" in the navigation
2. Click "Start Camera" and grant camera permissions
3. Point your camera at waste items
4. See real-time classifications appear automatically
5. Adjust settings:
   - **Confidence Threshold**: Filter predictions below a certain confidence
   - **Prediction Interval**: Control how often frames are analyzed (100ms - 2000ms)

### Analytics Dashboard

1. Navigate to `/analytics`
2. View classification statistics
3. See distribution charts and recent predictions


## Training Your Own Model

1. Organize dataset in `data/processed/` following the structure:
```
data/processed/
├── train/
│   ├── plastic/
│   ├── metal/
│   ├── organic/
│   └── paper/
├── val/
└── test/
```

2. Run training script:
```bash
python src/train.py
```

3. Evaluate model:
```bash
python src/evaluate.py
```

## Model Performance

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Training Accuracy**: 88%
- **Validation Accuracy**: 85%
- **Test Accuracy**: 83%
- **Inference Time**: 
  - Upload mode: <100ms per image
  - Live mode: 200-500ms per frame (2-5 FPS)
- **Categories**: Plastic, Metal, Organic, Paper, Glass, Cardboard, Textile, Vegetation, Miscellaneous

## Project Structure
```
waste-classification/
├── data/
│   ├── processed/       # Organized dataset (train/val/test)
│   └── models/          # Trained models
├── app/
│   ├── app.py          # Flask application with prediction endpoints
│   ├── templates/       # HTML templates
│   │   ├── index.html  # Upload mode interface
│   │   ├── live.html   # Live camera interface
│   │   └── analytics.html  # Statistics dashboard
│   └── static/          # CSS, JS, uploads
├── src/
│   ├── train.py        # Model training
│   ├── evaluate.py     # Model evaluation
│   └── predict.py      # Single image prediction
├── notebooks/
│   └── 01_data_exploration.ipynb
├── requirements.txt
└── README.md
```

## Future Enhancements

-  Multi-language support
-  Voice announcements for classifications
-  Batch processing for multiple items
-  Object detection for multiple items in frame
-  Offline mode with cached model
-  Mobile app (React Native/Flutter)
-  Hardware integration (Arduino/Raspberry Pi)
-  Community feedback system
-  Gamification and achievements
-  Integration with recycling centers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset: RealWaste Image Classification / Kaggle Waste Classification
- Base Model: MobileNetV2 (ImageNet pre-trained)
- Framework: TensorFlow/Keras

## Contact
Isha Kale - ishakale07@gmail.com
Project Link: https://github.com/ishakale07/AI-Waste-Classification-System