import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30  # Increased from 20
NUM_CLASSES = 9

# Fix the paths - use relative paths from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Gets src directory
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # Goes up to project root

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Verify paths exist
print("Checking paths...")
print(f"Project root: {PROJECT_ROOT}")
print(f"Train dir: {TRAIN_DIR}")
print(f"Train dir exists: {os.path.exists(TRAIN_DIR)}")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found at: {TRAIN_DIR}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,          # Increased from 20
    width_shift_range=0.3,      # Increased from 0.2
    height_shift_range=0.3,     # Increased from 0.2
    horizontal_flip=True,
    vertical_flip=True,         # NEW - Added vertical flip
    zoom_range=0.3,             # Increased from 0.2
    shear_range=0.3,            # Increased from 0.2
    brightness_range=[0.8, 1.2], # NEW - Added brightness variation
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load data
print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("\nLoading validation data...")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Save class labels
class_labels = list(train_generator.class_indices.keys())
print(f"\nClasses found: {class_labels}")
print(f"Number of classes: {len(class_labels)}")

# Update NUM_CLASSES based on actual data
NUM_CLASSES = len(class_labels)

# Model Architecture - Transfer Learning with MobileNetV2
print("\nBuilding improved model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Initially freeze all layers
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),        # NEW - Added batch normalization
    layers.Dense(512, activation='relu'),  # Increased from 256
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),  # Increased from 128
    layers.BatchNormalization(),        # NEW - Added batch normalization
    layers.Dropout(0.4),                # Adjusted dropout
    layers.Dense(128, activation='relu'),  # NEW - Additional layer
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
print(model.summary())

# Callbacks
checkpoint = keras.callbacks.ModelCheckpoint(
    os.path.join(MODELS_DIR, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# =============================================================================
# STAGE 1: Train with frozen base model
# =============================================================================
print("\n" + "="*60)
print("STAGE 1: Training with frozen base model")
print("="*60)

history_stage1 = model.fit(
    train_generator,
    epochs=15,  # Train for 15 epochs first
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)

# =============================================================================
# STAGE 2: Fine-tune with unfrozen layers
# =============================================================================
print("\n" + "="*60)
print("STAGE 2: Fine-tuning with unfrozen layers")
print("="*60)

# Unfreeze the base model
base_model.trainable = True

# Freeze the first 100 layers (keep early feature detectors frozen)
for layer in base_model.layers[:100]:
    layer.trainable = False

print(f"Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

# Recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Much lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Reset callbacks for stage 2
checkpoint_stage2 = keras.callbacks.ModelCheckpoint(
    os.path.join(MODELS_DIR, 'best_model_finetuned.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping_stage2 = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,  # More patience for fine-tuning
    restore_best_weights=True,
    verbose=1
)

reduce_lr_stage2 = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-8,
    verbose=1
)

# Continue training
history_stage2 = model.fit(
    train_generator,
    epochs=15,  # Additional 15 epochs
    validation_data=val_generator,
    callbacks=[checkpoint_stage2, early_stopping_stage2, reduce_lr_stage2],
    verbose=1
)

# Combine histories
history = history_stage1
for key in history.history.keys():
    history.history[key].extend(history_stage2.history[key])

# Save final model
model.save(os.path.join(MODELS_DIR, 'waste_classifier_final_improved.h5'))
print(f"\nImproved model saved to {MODELS_DIR}")

# Save class labels
import json
with open(os.path.join(MODELS_DIR, 'class_labels.json'), 'w') as f:
    json.dump(class_labels, f)
print(f"Class labels saved to {MODELS_DIR}")

# Plot training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))
    print(f"Training history plot saved to {MODELS_DIR}")
    plt.show()

plot_history(history)

print("\n" + "="*50)
print("Training complete!")
print("="*50)