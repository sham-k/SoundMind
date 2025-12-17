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
# From project root (recommended):
./run_app.sh

# Or manually:
streamlit run app/main.py
```

This will open the app in your browser at `http://localhost:8501`

The app automatically uses the best available model (prioritizing optimized → enhanced → baseline).

**Features:**
- Upload WAV audio files
- Get instant emotion predictions with 85% accuracy
- View confidence scores and probability distributions
- Beautiful visualizations with charts
- Automatically uses the best available trained model

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

**Good news!** The optimized model (85.07% accuracy) is already trained and ready to use!

If you want to retrain or improve further:

```bash
# Retrain optimized model (recommended - 85% accuracy)
python train_optimized.py

# Or train enhanced model (80% accuracy)
python train_enhanced.py

# Or train ensemble (85-90% accuracy, takes ~90 min)
python train_ensemble.py

# Or baseline model (65% accuracy)
python train.py
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training options and [MODEL_IMPROVEMENTS.md](MODEL_IMPROVEMENTS.md) for technical details.

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
