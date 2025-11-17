# -*- coding: utf-8 -*-
"""Quick test script to verify the prediction pipeline works."""

from app.utils import EmotionPredictor

# Initialize predictor
print("Loading model...")
predictor = EmotionPredictor(
    model_path="models/emotion_model.h5",
    encoder_path="models/label_encoder.pkl",
    scaler_path="models/scaler.pkl"
)

# Test with a sample audio file
test_audio = "data/raw/Actor_16/03-01-05-01-02-01-16.wav"
print(f"\nTesting with: {test_audio}")

# Make prediction
result = predictor.predict(test_audio)

# Display results
print("\n" + "="*50)
print("PREDICTION RESULTS")
print("="*50)
print(f"Emotion: {result['emotion'].upper()}")
print(f"Confidence: {result['confidence']*100:.2f}%")
print("\nAll probabilities:")
for emotion, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {emotion:12s}: {prob*100:.2f}%")

print("\nPrediction successful!")
