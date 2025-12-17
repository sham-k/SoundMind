# -*- coding: utf-8 -*-
import numpy as np
import librosa
import pickle
from tensorflow import keras


class EmotionPredictor:
    """Loads trained model and makes emotion predictions on audio files."""

    def __init__(self, model_path, encoder_path, scaler_path):
        """
        Initialize the predictor with model artifacts.

        Args:
            model_path: Path to the trained Keras model (.h5)
            encoder_path: Path to the label encoder (.pkl)
            scaler_path: Path to the feature scaler (.pkl)
        """
        self.model = keras.models.load_model(model_path)

        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def extract_features(self, audio_path, duration=3, offset=0.5):
        """
        Extract audio features from an audio file.
        Automatically detects if model needs basic (40) or enhanced (392) features.

        Args:
            audio_path: Path to the audio file (.wav)
            duration: Duration of audio to analyze (seconds)
            offset: Starting point in the audio (seconds)

        Returns:
            numpy array of features (40 or 392 dimensions)
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=duration, offset=offset, sr=22050)

            # Determine feature type based on model input shape
            expected_features = self.model.input_shape[1]

            if expected_features == 40:
                # Basic MFCC features (for baseline model)
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
                return mfccs

            elif expected_features == 392:
                # Enhanced features (for optimized model)
                return self._extract_enhanced_features(y, sr)

            else:
                raise ValueError(f"Unexpected model input shape: {expected_features}")

        except Exception as e:
            raise ValueError(f"Error extracting features from {audio_path}: {e}")

    def _extract_enhanced_features(self, audio, sr):
        """
        Extract comprehensive audio features (392 dimensions).
        Same as preprocessing_enhanced.py
        """
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

    def predict(self, audio_path):
        """
        Predict emotion from an audio file.

        Args:
            audio_path: Path to the audio file (.wav)

        Returns:
            dict: {
                'emotion': predicted emotion label,
                'confidence': confidence score (0-1),
                'probabilities': dict of all emotion probabilities
            }
        """
        # Extract features
        features = self.extract_features(audio_path)

        # Scale features (same as training)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Make prediction
        predictions = self.model.predict(features_scaled, verbose=0)[0]

        # Get predicted emotion
        predicted_idx = np.argmax(predictions)
        emotion = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(predictions[predicted_idx])

        # Get all probabilities
        probabilities = {
            self.label_encoder.inverse_transform([i])[0]: float(predictions[i])
            for i in range(len(predictions))
        }

        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        }

    def predict_batch(self, audio_paths):
        """
        Predict emotions for multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            list of prediction dictionaries
        """
        return [self.predict(path) for path in audio_paths]
