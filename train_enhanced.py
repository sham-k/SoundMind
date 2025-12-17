# -*- coding: utf-8 -*-
"""
Enhanced training script with CNN architecture for improved emotion recognition.
Expected accuracy improvement: 66% -> 75-80%
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, models
import pickle
import os
import matplotlib.pyplot as plt

# Config
DATA_PATH = "data/processed/features_enhanced.csv"
MODEL_PATH = "models/emotion_model_enhanced.h5"
ENCODER_PATH = "models/label_encoder_enhanced.pkl"
SCALER_PATH = "models/scaler_enhanced.pkl"
HISTORY_PATH = "models/training_history.png"

EPOCHS = 200
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42


def create_cnn_model(input_shape, num_classes):
    """
    Create a 1D CNN model optimized for audio feature classification.
    CNNs are better than dense networks for audio because they can learn
    local patterns and are translation-invariant.
    """
    model = models.Sequential([
        # Reshape for CNN (add channel dimension)
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

        # First Convolutional Block
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Second Convolutional Block
        layers.Conv1D(256, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Third Convolutional Block
        layers.Conv1D(512, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),

        # Fourth Convolutional Block
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.4),

        # Dense layers
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_hybrid_model(input_shape, num_classes):
    """
    Hybrid CNN + Dense model for even better performance.
    Combines local pattern recognition (CNN) with global context (Dense).
    """
    model = models.Sequential([
        # Reshape for CNN
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),

        # CNN Branch
        layers.Conv1D(64, kernel_size=8, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(128, kernel_size=8, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(256, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(256, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(512, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),

        # Dense layers
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Output
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_PATH, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {HISTORY_PATH}")


def main():
    print("=" * 70)
    print("ENHANCED EMOTION RECOGNITION TRAINING")
    print("=" * 70)
    print("\nImprovements:")
    print("  1. Enhanced features (MFCC, Chroma, Mel, Contrast, Tonnetz, etc.)")
    print("  2. Data augmentation (3x more training samples)")
    print("  3. CNN architecture (better for audio patterns)")
    print("  4. Class weights (balanced learning)")
    print("\n" + "=" * 70)

    # Load data
    print("\nLoading enhanced dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: {DATA_PATH} not found!")
        print("Please run: python preprocessing_enhanced.py")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape}")

    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"\nFeatures: {X.shape[1]} dimensions")
    print(f"Samples: {X.shape[0]}")
    print(f"Emotions: {np.unique(y)}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    print(f"\nClass distribution:")
    for i, emotion in enumerate(le.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {emotion:12s}: {count:4d} samples ({count/len(y)*100:.1f}%)")

    # Calculate class weights for balanced training
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights computed for balanced training")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"\nData split:")
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Testing:    {X_test.shape[0]} samples")

    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build model (try both and pick the best one)
    print("\nBuilding CNN model...")
    # model = create_cnn_model(X_train.shape[1], num_classes)
    model = create_hybrid_model(X_train.shape[1], num_classes)  # Use hybrid for best results

    # Optimizer with learning rate schedule
    initial_lr = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    print("=" * 70)
    model.summary()
    print("=" * 70)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
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
    print("=" * 70)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")

    # Improvement comparison
    baseline_acc = 0.6569
    improvement = (test_acc - baseline_acc) * 100
    print(f"\nBaseline:      {baseline_acc*100:.2f}%")
    print(f"New Model:     {test_acc*100:.2f}%")
    print(f"Improvement:   {'+' if improvement > 0 else ''}{improvement:.2f}%")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print("\n" + "=" * 70)
    print("MODEL ARTIFACTS SAVED")
    print("=" * 70)
    print(f"Model:         {MODEL_PATH}")
    print(f"Encoder:       {ENCODER_PATH}")
    print(f"Scaler:        {SCALER_PATH}")

    # Per-class performance
    print("\n" + "=" * 70)
    print("PER-CLASS PERFORMANCE")
    print("=" * 70)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot training history
    plot_training_history(history)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Update app to use new model (emotion_model_enhanced.h5)")
    print("2. Test with real audio samples")
    print("3. Deploy updated model to production")


if __name__ == "__main__":
    main()
