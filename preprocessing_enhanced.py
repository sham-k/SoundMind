# -*- coding: utf-8 -*-
"""
Enhanced preprocessing with data augmentation and richer feature extraction.
This script will significantly improve model performance.
"""
import numpy as np
import pandas as pd
import librosa
import os
from pathlib import Path
import soundfile as sf

# Configuration
DATASET_PATH = "data/raw"
OUTPUT_PATH = "data/processed/features_enhanced.csv"
SAMPLE_RATE = 22050
DURATION = 3  # seconds

# Data augmentation parameters
AUGMENTATION_FACTOR = 3  # Generate 3 augmented versions per sample


def add_noise(data, noise_factor=0.005):
    """Add random white noise to audio."""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data.astype(type(data[0]))


def time_stretch(data, rate=1.0):
    """Stretch or compress audio in time without changing pitch."""
    return librosa.effects.time_stretch(data, rate=rate)


def pitch_shift(data, sr, n_steps=0):
    """Shift pitch of audio without changing tempo."""
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)


def time_shift(data, shift_max=0.2):
    """Shift audio forward or backward in time."""
    shift = np.random.randint(int(SAMPLE_RATE * shift_max))
    direction = np.random.randint(0, 2)
    if direction == 1:
        shifted_data = np.roll(data, shift)
    else:
        shifted_data = np.roll(data, -shift)
    return shifted_data


def extract_enhanced_features(file_path, augment=False):
    """
    Extract comprehensive audio features including:
    - MFCC (Mel-frequency cepstral coefficients)
    - Chroma (pitch class profiles)
    - Mel Spectrogram
    - Spectral Contrast
    - Tonnetz (tonal centroid features)
    - Zero Crossing Rate
    - Spectral Centroid
    - Spectral Rolloff
    """
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Apply augmentation if requested
        if augment:
            aug_choice = np.random.choice(['noise', 'stretch', 'pitch', 'shift'])
            if aug_choice == 'noise':
                audio = add_noise(audio)
            elif aug_choice == 'stretch':
                rate = np.random.uniform(0.8, 1.2)
                audio = time_stretch(audio, rate=rate)
            elif aug_choice == 'pitch':
                n_steps = np.random.randint(-3, 4)
                audio = pitch_shift(audio, sr, n_steps=n_steps)
            elif aug_choice == 'shift':
                audio = time_shift(audio)

        # Extract multiple feature sets
        features = []

        # 1. MFCC features (40 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)

        # 2. Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)
        features.extend(chroma_mean)
        features.extend(chroma_std)

        # 3. Mel Spectrogram (128 bands)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_mean = np.mean(mel.T, axis=0)
        mel_std = np.std(mel.T, axis=0)
        features.extend(mel_mean)
        features.extend(mel_std)

        # 4. Spectral Contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std = np.std(contrast.T, axis=0)
        features.extend(contrast_mean)
        features.extend(contrast_std)

        # 5. Tonnetz (tonal centroid features) (6 features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        tonnetz_std = np.std(tonnetz.T, axis=0)
        features.extend(tonnetz_mean)
        features.extend(tonnetz_std)

        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        features.extend([zcr_mean, zcr_std])

        # 7. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])

        # 8. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

        return np.array(features)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_dataset(augment=True):
    """
    Process the RAVDESS dataset with enhanced features and augmentation.

    RAVDESS filename format: 03-01-06-01-02-01-12.wav
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
    - Vocal channel (01 = speech, 02 = song)
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad,
               05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    - Emotional intensity (01 = normal, 02 = strong)
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
    - Repetition (01 = 1st repetition, 02 = 2nd repetition)
    - Actor (01 to 24, odd = male, even = female)
    """

    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    data = []
    labels = []

    print("Processing RAVDESS dataset with enhanced features...")
    print(f"Augmentation: {'ENABLED' if augment else 'DISABLED'}")
    print("=" * 60)

    # Find all audio files
    audio_files = list(Path(DATASET_PATH).rglob("*.wav"))
    total_files = len(audio_files)

    if total_files == 0:
        print(f"ERROR: No audio files found in {DATASET_PATH}")
        print("Please ensure RAVDESS dataset is extracted to data/raw/")
        return

    print(f"Found {total_files} audio files")

    processed = 0
    for file_path in audio_files:
        filename = file_path.name

        # Parse RAVDESS filename
        parts = filename.split('.')[0].split('-')

        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]

                # Extract features from original audio
                features = extract_enhanced_features(str(file_path), augment=False)
                if features is not None:
                    data.append(features)
                    labels.append(emotion)
                    processed += 1

                    # Generate augmented versions
                    if augment:
                        for i in range(AUGMENTATION_FACTOR):
                            aug_features = extract_enhanced_features(str(file_path), augment=True)
                            if aug_features is not None:
                                data.append(aug_features)
                                labels.append(emotion)
                                processed += 1

                if processed % 100 == 0:
                    print(f"Processed: {processed} samples...")

    print(f"\nTotal samples generated: {len(data)}")
    print(f"Feature dimensionality: {len(data[0]) if data else 0}")

    # Create DataFrame
    df = pd.DataFrame(data)
    df['emotion'] = labels

    # Display class distribution
    print("\nClass distribution:")
    print(df['emotion'].value_counts().sort_index())

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nEnhanced features saved to: {OUTPUT_PATH}")
    print(f"Dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    print("Enhanced Feature Extraction with Data Augmentation")
    print("=" * 60)
    print("\nThis will:")
    print("1. Extract rich audio features (MFCC, Chroma, Mel, Contrast, Tonnetz, etc.)")
    print("2. Apply data augmentation (noise, time/pitch shifting)")
    print(f"3. Generate {AUGMENTATION_FACTOR}x more training samples")
    print("\nThis may take several minutes...\n")

    df = process_dataset(augment=True)

    if df is not None:
        print("\n" + "=" * 60)
        print("SUCCESS! Enhanced dataset ready for training.")
        print("=" * 60)
        print(f"\nNext step: Run 'python train_enhanced.py' to train with improved model")
