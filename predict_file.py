#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple command-line tool to predict emotion from an audio file.
Usage: python predict_file.py <path_to_audio.wav>
"""

import sys
from app.utils import EmotionPredictor

EMOTION_EMOJIS = {
    'angry': '😠',
    'calm': '😌',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprised': '😲'
}

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python predict_file.py <path_to_audio.wav>")
        print("\nExample:")
        print("  python predict_file.py data/raw/Actor_01/03-01-05-01-01-01-01.wav")
        print("\nOr try one of these sample files:")
        print("  python predict_file.py data/raw/Actor_16/03-01-05-01-02-01-16.wav  # Angry")
        print("  python predict_file.py data/raw/Actor_16/03-01-03-01-02-01-16.wav  # Happy")
        print("  python predict_file.py data/raw/Actor_16/03-01-04-01-02-01-16.wav  # Sad")
        sys.exit(1)

    audio_file = sys.argv[1]

    print("\n" + "="*60)
    print("🎧 SOUNDMIND - Emotion Prediction")
    print("="*60)

    # Load model
    print("\n📦 Loading model...")
    predictor = EmotionPredictor(
        model_path="models/emotion_model.h5",
        encoder_path="models/label_encoder.pkl",
        scaler_path="models/scaler.pkl"
    )

    # Predict
    print(f"🎵 Analyzing: {audio_file}")
    result = predictor.predict(audio_file)

    # Display results
    emotion = result['emotion']
    confidence = result['confidence'] * 100
    emoji = EMOTION_EMOJIS.get(emotion, '🎭')

    print("\n" + "="*60)
    print("🎯 RESULTS")
    print("="*60)
    print(f"\n  {emoji}  Emotion: {emotion.upper()}")
    print(f"  📊 Confidence: {confidence:.1f}%\n")

    print("All Probabilities:")
    print("-" * 60)
    for emo, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
        emoji = EMOTION_EMOJIS.get(emo, '🎭')
        bar = "█" * int(prob * 50)
        print(f"  {emoji} {emo:12s} {bar} {prob*100:5.1f}%")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
