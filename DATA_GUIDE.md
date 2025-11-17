# Audio Data Guide for SoundMind

## Option 1: Use Your Own Audio Files (Quick Testing)

You don't need the full RAVDESS dataset to **use** the app! The trained model is already saved in `models/` directory.

### Using the Web App:
1. Launch the app: `cd app && streamlit run main.py`
2. Upload ANY WAV file with speech
3. Get instant emotion predictions!

### Recording Your Own Audio:

**On macOS:**
```bash
# Record 5 seconds of audio
rec -r 22050 -c 1 my_audio.wav trim 0 5
```

**Using Python:**
```python
import sounddevice as sd
import scipy.io.wavfile as wavfile

# Record 5 seconds
fs = 22050
duration = 5
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
wavfile.write('my_recording.wav', fs, recording)
```

**Using Audacity (Free Software):**
1. Download from: https://www.audacityteam.org/
2. Record audio
3. Export as WAV (File → Export → Export Audio)
4. Choose: WAV (Microsoft), 16-bit PCM

### Converting Other Formats to WAV:

If you have MP3, M4A, or other formats:

```bash
# Install ffmpeg first: brew install ffmpeg

# Convert to WAV
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav
```

---

## Option 2: Get the RAVDESS Dataset (For Training)

If you want to **retrain** the model or work with the original dataset:

### Download RAVDESS:

**Official Source:**
- Website: https://zenodo.org/record/1188976
- Direct Download: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
- Size: ~25 GB
- Contains: 1,440 audio files from 24 actors

### Download Steps:

1. **Download the dataset:**
```bash
# Create data directory
mkdir -p data/raw

# Download (this will take a while - 25GB!)
cd data/raw
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

# Or use curl
curl -O https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
```

2. **Extract the files:**
```bash
unzip Audio_Speech_Actors_01-24.zip
```

3. **Verify the structure:**
```bash
ls data/raw/
# Should show: Actor_01, Actor_02, ..., Actor_24
```

### Dataset File Naming Convention:

Files are named: `03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav`

**Emotion codes:**
- `01` = neutral
- `02` = calm
- `03` = happy
- `04` = sad
- `05` = angry
- `06` = fearful
- `07` = disgust
- `08` = surprised

Example: `03-01-05-01-02-01-16.wav`
- Emotion: `05` (angry)
- Actor: `16`

---

## Option 3: Alternative Datasets

If RAVDESS is too large, try these smaller alternatives:

### 1. TESS (Toronto Emotional Speech Set)
- Size: ~2.5 GB
- URL: https://tspace.library.utoronto.ca/handle/1807/24487
- Emotions: 7 (same as RAVDESS minus calm)

### 2. CREMA-D
- Size: ~9 GB
- URL: https://github.com/CheyneyComputerScience/CREMA-D
- Emotions: 6 basic emotions

### 3. SAVEE (Surrey Audio-Visual Expressed Emotion)
- Size: ~500 MB
- URL: http://kahlan.eps.surrey.ac.uk/savee/
- Emotions: 7 emotions

---

## Quick Start Without Downloading

The project **already includes a trained model**, so you can:

1. **Use the demo:**
```bash
python demo.py
```
This uses sample files already in your `data/raw/` directory.

2. **Test with any WAV file:**
```bash
python test_prediction.py
# Edit the file to point to your own audio file
```

3. **Use the web app:**
```bash
cd app && streamlit run main.py
# Upload any WAV file through the browser
```

---

## Creating Sample Audio Files

Want to create test files quickly? Here's a Python script:

```python
# create_samples.py
import numpy as np
from scipy.io.wavfile import write

# Generate a simple tone
fs = 22050
duration = 3
t = np.linspace(0, duration, int(fs * duration))

# Create different tones for testing
frequencies = [440, 523, 659, 784]  # A, C, E, G
for i, freq in enumerate(frequencies):
    tone = np.sin(2 * np.pi * freq * t)
    tone = (tone * 32767).astype(np.int16)
    write(f'sample_{i}.wav', fs, tone)
```

---

## Troubleshooting

**Q: The app says "No audio file found"**
- Ensure your file is in WAV format
- Check the file isn't corrupted
- Try converting with ffmpeg

**Q: Prediction seems random**
- Make sure audio contains **speech** (not music or silence)
- Best results with emotional speech (3-5 seconds)
- Clear audio without background noise works best

**Q: How do I record on Windows?**
- Use Windows Voice Recorder (built-in)
- Or download Audacity (free)

**Q: Can I use this with real-time audio?**
- Not currently, but you can extend [app/main.py](app/main.py) to add microphone recording using `streamlit-audiorecorder`

---

## Next Steps

1. ✅ **You already have the dataset** in `data/raw/`
2. ✅ **Model is already trained** in `models/`
3. 🎯 **Just run the app:** `cd app && streamlit run main.py`
4. 🎤 **Upload any WAV file** and test it out!

No additional downloads needed unless you want to retrain the model!
