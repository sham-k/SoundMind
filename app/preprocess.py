# -*- coding: utf-8 -*-
import os
import librosa
import numpy as np
import pandas as pd

DATA_PATH = "data/raw/"
OUTPUT_CSV = "data/processed/features.csv"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Skipped {file_path}: {e}")
        return None

rows = []
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                file_path = os.path.join(root, file)
                features = extract_features(file_path)
                if features is not None:
                    rows.append([*features, emotion])

df = pd.DataFrame(rows)
os.makedirs("data/processed", exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Extracted {len(df)} samples → saved to {OUTPUT_CSV}")