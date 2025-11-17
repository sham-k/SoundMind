# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

# Config
DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/emotion_model.h5"
ENCODER_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"
EPOCHS = 150
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.15

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Separate features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Emotions: {np.unique(y)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
)

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Feature scaling - CRITICAL for neural networks
print("\nScaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build improved model with batch normalization
print("\nBuilding model...")
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),

    # First block
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Second block
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Third block
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Fourth block
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Output layer
    layers.Dense(num_classes, activation='softmax')
])

# Use a lower initial learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# Callbacks - improved monitoring
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# Train
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save model and encoder
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(le, f)

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nModel saved to {MODEL_PATH}")
print(f"Label encoder saved to {ENCODER_PATH}")
print(f"Feature scaler saved to {SCALER_PATH}")

# Print per-class accuracy
print("\n" + "="*50)
print("CLASS-WISE PERFORMANCE")
print("="*50)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
for i, emotion in enumerate(le.classes_):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = (y_pred[mask] == y_test[mask]).mean()
        print(f"{emotion:12s}: {class_acc:.4f} ({mask.sum()} samples)")

print("\nTraining complete!")
