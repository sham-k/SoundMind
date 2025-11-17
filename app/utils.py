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
        Extract MFCC features from an audio file.

        Args:
            audio_path: Path to the audio file (.wav)
            duration: Duration of audio to analyze (seconds)
            offset: Starting point in the audio (seconds)

        Returns:
            numpy array of MFCC features (40 coefficients)
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=duration, offset=offset)

            # Extract MFCCs (same as training)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

            return mfccs
        except Exception as e:
            raise ValueError(f"Error extracting features from {audio_path}: {e}")

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
