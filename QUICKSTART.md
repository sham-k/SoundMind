# SoundMind - Quick Start Guide

## Prerequisites
- Python 3.8 or higher
- Virtual environment activated

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Web Application (Recommended)

Launch the interactive Streamlit web app:

```bash
cd app
streamlit run main.py
```

This will open the app in your browser at `http://localhost:8501`

**Features:**
- Upload WAV audio files
- Get instant emotion predictions
- View confidence scores and probability distributions
- Beautiful visualizations with charts

### Option 2: Command Line Testing

Test predictions directly from Python:

```bash
python test_prediction.py
```

### Option 3: Use in Your Own Code

```python
from app.utils import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor(
    model_path="models/emotion_model.h5",
    encoder_path="models/label_encoder.pkl",
    scaler_path="models/scaler.pkl"
)

# Make prediction
result = predictor.predict("path/to/audio.wav")

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Training Your Own Model

If you want to retrain the model:

```bash
# 1. Prepare your data in data/raw/ directory (RAVDESS format)

# 2. Preprocess the data
python app/preprocess.py

# 3. Train the model
python train.py
```

## Project Structure

```
SoundMind/
├── app/
│   ├── main.py          # Streamlit web app
│   ├── utils.py         # Prediction utilities
│   └── preprocess.py    # Data preprocessing
├── models/              # Trained models
│   ├── emotion_model.h5
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── data/
│   ├── raw/            # Raw audio files
│   └── processed/      # Extracted features
├── train.py            # Model training script
└── requirements.txt    # Dependencies
```

## Supported Emotions

- 😠 Angry
- 😌 Calm
- 🤢 Disgust
- 😨 Fearful
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprised

## Tips for Best Results

1. **Audio Quality**: Use clear recordings with minimal background noise
2. **Duration**: 3-5 seconds of speech works best
3. **Format**: WAV files are required (16-bit PCM recommended)
4. **Content**: Speech with clear emotional expression yields better results

## Troubleshooting

**Issue**: Model files not found
- **Solution**: Make sure you've run `python train.py` first to generate the model files

**Issue**: Streamlit not found
- **Solution**: Run `pip install streamlit` or reinstall requirements

**Issue**: Audio file not loading
- **Solution**: Ensure the file is in WAV format. Convert using:
  ```bash
  ffmpeg -i input.mp3 -ar 22050 output.wav
  ```
