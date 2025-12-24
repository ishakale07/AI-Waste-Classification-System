import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
TEST_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test')

# Load model and labels
print("Loading model...")
model = keras.models.load_model(os.path.join(MODELS_DIR, 'waste_classifier_final_improved.h5'))

with open(os.path.join(MODELS_DIR, 'class_labels.json'), 'r') as f:
    class_labels = json.load(f)

print(f"Model loaded successfully!")
print(f"Classes: {class_labels}")

# Load test data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
print("\n" + "="*60)
print("EVALUATING MODEL ON TEST SET")
print("="*60)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\n📊 Test Results:")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}")

# Predictions
print("\nGenerating predictions...")
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_classes, predicted_classes, 
                          target_names=class_labels, 
                          digits=3))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'), dpi=300)
print(f"\n✓ Confusion matrix saved to {MODELS_DIR}")
plt.show()

# Per-class accuracy
class_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(12, 6))
bars = plt.bar(class_labels, class_accuracy, color='steelblue', edgecolor='black')
plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim([0, 1.0])

# Add percentage labels on bars
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.1%}',
             ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'per_class_accuracy.png'), dpi=300)
print(f"✓ Per-class accuracy plot saved to {MODELS_DIR}")
plt.show()

# Show misclassified examples
def show_misclassified(num_samples=10):
    misclassified_idx = np.where(predicted_classes != true_classes)[0]
    
    if len(misclassified_idx) == 0:
        print("\n🎉 No misclassifications found! Perfect score!")
        return
    
    print(f"\n❌ Found {len(misclassified_idx)} misclassifications")
    
    samples = np.random.choice(misclassified_idx, 
                               min(num_samples, len(misclassified_idx)), 
                               replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.ravel()
    
    for i, idx in enumerate(samples):
        img_path = test_generator.filepaths[idx]
        img = plt.imread(img_path)
        
        true_label = class_labels[true_classes[idx]]
        pred_label = class_labels[predicted_classes[idx]]
        confidence = predictions[idx][predicted_classes[idx]]
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.1%})',
                         fontsize=9, color='red')
    
    plt.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'misclassified_examples.png'), dpi=300)
    print(f"✓ Misclassified examples saved to {MODELS_DIR}")
    plt.show()

show_misclassified(10)

# Final Summary
print("\n" + "="*60)
print("EVALUATION SUMMARY")
print("="*60)
print(f"✓ Total test images: {len(test_generator.filenames)}")
print(f"✓ Test accuracy: {test_accuracy*100:.2f}%")
print(f"✓ Correct predictions: {np.sum(predicted_classes == true_classes)}")
print(f"✓ Incorrect predictions: {np.sum(predicted_classes != true_classes)}")
print(f"\n✓ Best performing class: {class_labels[np.argmax(class_accuracy)]} ({class_accuracy.max():.1%})")
print(f"✓ Worst performing class: {class_labels[np.argmin(class_accuracy)]} ({class_accuracy.min():.1%})")
print("\n" + "="*60)
print("Evaluation complete!")
print("="*60)