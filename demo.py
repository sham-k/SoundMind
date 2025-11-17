# -*- coding: utf-8 -*-
"""
Demo script to showcase SoundMind emotion recognition capabilities.
Tests predictions on multiple sample audio files.
"""

from app.utils import EmotionPredictor
import os

# Emotion emojis for display
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
    print("\n" + "="*60)
    print("   🎧 SOUNDMIND - AI EMOTION RECOGNITION DEMO")
    print("="*60)

    # Load model
    print("\n📦 Loading trained model...")
    predictor = EmotionPredictor(
        model_path="models/emotion_model.h5",
        encoder_path="models/label_encoder.pkl",
        scaler_path="models/scaler.pkl"
    )
    print("✓ Model loaded successfully!")

    # Find sample audio files
    print("\n🔍 Finding sample audio files...")
    sample_dir = "data/raw/Actor_16"

    if not os.path.exists(sample_dir):
        print(f"❌ Sample directory not found: {sample_dir}")
        return

    audio_files = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.endswith('.wav')
    ][:5]  # Test first 5 files

    print(f"✓ Found {len(audio_files)} sample files")

    # Run predictions
    print("\n" + "="*60)
    print("🎯 RUNNING PREDICTIONS")
    print("="*60)

    for i, audio_path in enumerate(audio_files, 1):
        filename = os.path.basename(audio_path)

        print(f"\n[{i}/{len(audio_files)}] {filename}")
        print("-" * 60)

        try:
            result = predictor.predict(audio_path)

            emotion = result['emotion']
            confidence = result['confidence'] * 100
            emoji = EMOTION_EMOJIS.get(emotion, '🎭')

            print(f"  {emoji} Predicted Emotion: {emotion.upper()}")
            print(f"  📊 Confidence: {confidence:.1f}%")

            # Show top 3 probabilities
            top_emotions = sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            print(f"  📈 Top 3:")
            for emo, prob in top_emotions:
                emoji = EMOTION_EMOJIS.get(emo, '🎭')
                bar = "█" * int(prob * 20)
                print(f"     {emoji} {emo:10s} {bar} {prob*100:5.1f}%")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "="*60)
    print("✨ Demo complete! Visit http://localhost:8501 to try the web app")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
