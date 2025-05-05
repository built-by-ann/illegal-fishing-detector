# train_model.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to avoid GPU hangs on some systems

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

print("🔥 Script is running!")

# Load data
print("📥 Loading X and y...")
X = np.load("data/training/X.npy")
y = np.load("data/training/y.npy")
print(f"✅ Loaded {X.shape[0]} samples. Shape: {X.shape}, Labels: {np.unique(y)}")

# Add channel dimension for CNN
X = X[..., np.newaxis]  # shape: (samples, 224, 224, 1)

# Train/test split
print("🔀 Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# Build CNN model
print("🛠️ Building model...")
model = models.Sequential([
    layers.Input(shape=(224, 224, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

print("⚙️ Compiling model...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
print("🚀 Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# Evaluate
print("📊 Evaluating model on test set...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")

# Classification report
print("🧠 Generating predictions and classification report...")
preds = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, preds))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# Plot accuracy and loss
print("📈 Plotting training history...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
