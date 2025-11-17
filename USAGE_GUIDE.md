# 📖 SoundMind Usage Guide

## 3 Ways to Use SoundMind

---

## Method 1: Command Line (Easiest!) ⚡

This is the **fastest way** to test audio files:

### Test with sample files:
```bash
# Test an angry voice
python predict_file.py data/raw/Actor_16/03-01-05-01-02-01-16.wav

# Test a happy voice
python predict_file.py data/raw/Actor_01/03-01-03-01-01-01-01.wav

# Test a sad voice
python predict_file.py data/raw/Actor_01/03-01-04-01-01-01-01.wav
```

### Test with your own file:
```bash
python predict_file.py /path/to/your/audio.wav
```

**Example Output:**
```
============================================================
🎧 SOUNDMIND - Emotion Prediction
============================================================

📦 Loading model...
🎵 Analyzing: data/raw/Actor_16/03-01-05-01-02-01-16.wav

============================================================
🎯 RESULTS
============================================================

  😠  Emotion: ANGRY
  📊 Confidence: 51.5%

All Probabilities:
------------------------------------------------------------
  😠 angry        ██████████  51.5%
  🤢 disgust      ██████  30.3%
  😲 surprised    █   7.7%
  ...
```

---

## Method 2: Web Interface (Most Visual!) 🌐

### Start the web app:
```bash
cd app
streamlit run main.py
```

### Then:
1. **Open your browser** → Go to `http://localhost:8501`

2. **You'll see:**
   - Title: "🎧 SoundMind"
   - File upload button
   - Sidebar with info

3. **Upload a file:**
   - Click **"Browse files"** button
   - Select a `.wav` file:
     - From dataset: Navigate to `Soundmind/data/raw/Actor_XX/`
     - Your own recording
   - Click "Open"

4. **Analyze:**
   - Click the **"🔮 Analyze Emotion"** button
   - Wait 2-3 seconds

5. **See results:**
   - Predicted emotion with emoji
   - Confidence percentage
   - Interactive probability chart
   - All emotion probabilities

### Web App Screenshot Guide:
```
┌─────────────────────────────────────────┐
│  🎧 SoundMind                           │
│  AI-Powered Emotion Recognition         │
├─────────────────────────────────────────┤
│                                         │
│  📤 Upload Audio File                   │
│  ┌───────────────────────────────────┐ │
│  │ Drag and drop file here           │ │
│  │      [Browse files]               │ │
│  └───────────────────────────────────┘ │
│                                         │
│  [Selected: my_audio.wav]              │
│  🎵 ▶━━━━━━━━━━━━━━━━━ 00:03         │
│                                         │
│     [🔮 Analyze Emotion]               │
│                                         │
│  ─────────────────────────────────────  │
│  🎯 Results                             │
│                                         │
│  Detected Emotion    Confidence        │
│  😊 Happy            78.5%             │
│                                         │
│  [Interactive Chart Showing All %]     │
│                                         │
└─────────────────────────────────────────┘
```

---

## Method 3: Python Code Integration 🐍

Use SoundMind in your own Python scripts:

```python
from app.utils import EmotionPredictor

# Initialize once
predictor = EmotionPredictor(
    model_path="models/emotion_model.h5",
    encoder_path="models/label_encoder.pkl",
    scaler_path="models/scaler.pkl"
)

# Predict for a single file
result = predictor.predict("my_audio.wav")

print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")

# Access all probabilities
for emotion, prob in result['probabilities'].items():
    print(f"{emotion}: {prob:.2%}")

# Batch prediction
results = predictor.predict_batch([
    "file1.wav",
    "file2.wav",
    "file3.wav"
])
```

---

## Where to Get Audio Files

### Option 1: Use Existing Dataset
You already have 1,440 files in `data/raw/`:
```bash
# List available actors
ls data/raw/ | grep Actor

# Pick any file
ls data/raw/Actor_01/

# Test it
python predict_file.py data/raw/Actor_01/03-01-03-01-01-01-01.wav
```

### Option 2: Record Your Own

**On macOS (QuickTime):**
1. Open QuickTime Player
2. File → New Audio Recording
3. Click the red record button
4. Speak with emotion for 3-5 seconds
5. Stop and save as `.wav`

**On macOS (Terminal):**
```bash
# Install sox if needed
brew install sox

# Record 5 seconds
sox -d -r 22050 -c 1 my_recording.wav trim 0 5
```

**On any platform (Audacity - Free):**
1. Download: https://www.audacityteam.org/
2. Click record, speak for 3-5 seconds
3. File → Export → Export Audio
4. Format: WAV (Microsoft), 16-bit PCM

### Option 3: Convert Other Formats

If you have MP3, M4A, etc:
```bash
# Install ffmpeg
brew install ffmpeg

# Convert to WAV
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav
```

---

## Quick Test Commands

Try these ready-to-use commands:

```bash
# Run the demo with 5 samples
python demo.py

# Predict a single file
python predict_file.py data/raw/Actor_16/03-01-05-01-02-01-16.wav

# Start the web app
cd app && streamlit run main.py
```

---

## Tips for Best Results

✅ **Do:**
- Use WAV files (16-bit PCM recommended)
- 3-5 seconds of clear speech
- Speak with clear emotional expression
- Minimize background noise

❌ **Avoid:**
- Music files
- Silence or very quiet audio
- Background noise/static
- Very short clips (< 1 second)

---

## Troubleshooting

**Q: "FileNotFoundError: No such file"**
- Check the file path is correct
- Use tab-completion in terminal
- Or use absolute paths: `/Users/sha/Soundmind/data/raw/...`

**Q: "Error extracting features"**
- Make sure file is actually WAV format
- Try converting: `ffmpeg -i input.mp3 output.wav`

**Q: Web app won't start**
- Check if port 8501 is free
- Try: `streamlit run main.py --server.port 8502`

**Q: Predictions seem random**
- Model works best on emotional speech
- Try files from the dataset first to verify it works
- Record yourself speaking with clear emotion

---

## Next Steps

1. ✅ Try the command line: `python predict_file.py data/raw/Actor_01/03-01-03-01-01-01-01.wav`
2. ✅ Run the demo: `python demo.py`
3. ✅ Start the web app: `cd app && streamlit run main.py`
4. 🎤 Record your own voice and test it!

Enjoy using SoundMind! 🎧✨
