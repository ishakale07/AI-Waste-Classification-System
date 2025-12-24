# AI Waste Classification System

A web-based application that uses deep learning to automatically classify waste items into categories (plastic, metal, organic, paper, etc.) to promote proper waste disposal and recycling.

## Features

- **Real-time Classification**: Upload images and get instant predictions
- **High Accuracy**: 85-95% classification accuracy using transfer learning
- **Confidence Scores**: See detailed prediction probabilities for all categories
- **Disposal Guidelines**: Get specific instructions for proper waste disposal
- **Analytics Dashboard**: Track classification statistics and trends
- **User-Friendly Interface**: Modern, responsive web design

## Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras, Transfer Learning (MobileNetV2)
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Plotly.js
- **Image Processing**: OpenCV, PIL

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
   - Place `best_model.h5` in `data/models/`
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

3. Upload an image of waste item and get classification results

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
- **Training Accuracy**: 92%
- **Validation Accuracy**: 89%
- **Test Accuracy**: 87%
- **Inference Time**: <100ms per image

## Project Structure
```
waste-classification/
├── data/
│   ├── processed/       # Organized dataset
│   └── models/          # Trained models
├── app/
│   ├── app.py          # Flask application
│   ├── templates/       # HTML templates
│   └── static/          # CSS, JS, uploads
├── src/
│   ├── train.py        # Model training
│   ├── evaluate.py     # Model evaluation
│   └── utils.py        # Utility functions
├── requirements.txt
└── README.md
```

## Future Enhancements

- Mobile app development
- Real-time video classification
- Multi-language support
- Integration with IoT devices
- Deployment to cloud platforms
- Database integration for persistent storage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset: TrashNet / Kaggle Waste Classification
- Base Model: MobileNetV2 (ImageNet pre-trained)
- Framework: TensorFlow/Keras

## Contact
Isha Kake - ishakale07@gmail.com
Project Link: https://github.com/yourusername/waste-classifier