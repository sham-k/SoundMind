# 🎧 SoundMind — AI Emotion Recognition from Voice

SoundMind is an AI-powered system that analyzes short voice clips and predicts the speaker’s emotional state using deep learning and audio feature extraction.  
The project demonstrates machine learning, signal processing, and model deployment skills in a real-world, human-centered application.

---

## 🚀 Features

- 🎙️ **Upload voice clips** (.wav) through a simple Streamlit UI  
- 🧠 **Emotion classification** using MFCC audio features  
- 🤖 **Deep learning model** (Keras + TensorFlow)  
- 📈 **Training pipeline** for preprocessing and feature extraction  
- 🔍 **Supports all 8 RAVDESS emotions:**  
  - Neutral  
  - Calm  
  - Happy  
  - Sad  
  - Angry  
  - Fearful  
  - Disgust  
  - Surprised  

---

## 🗂 Project Structure

SoundMind/
│
├── app/
│   ├── main.py           # Streamlit UI
│   ├── preprocess.py     # Audio → MFCC feature extraction
│   └── utils.py          # Model loading & prediction helpers
│
├── data/
│   ├── raw/              # RAVDESS audio dataset (ignored by Git)
│   └── processed/        # Generated features.csv
│
├── models/               # Saved emotion_model.h5 (ignored by Git)
│
├── notebooks/            # Future experimentation notebooks
│
├── train.py              # Model training script
├── requirements.txt
└── README.md

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sham-k/SoundMind.git
cd SoundMind
```
### 2️⃣ Create & activate virtual environment (Python 3.11)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```
### 3️⃣  Install dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt || pip install tensorflow-macos
```
## Dataset (RAVDESS)

SoundMind uses the RAVDESS Speech Audio Dataset:
Download: https://zenodo.org/record/1188976

Required ZIP file:
Audio_Speech_Actors_01-24.zip

Place it here:
```bash
SoundMind/data/raw/
```
## 🧠 Preprocessing (Audio → MFCC)
Convert .wav audio files to MFCC feature vectors:
```bash
python app/preprocess.py
```
Generates:
```bash
data/processed/features.csv
```
Each row contains:
* 40 MFCC audio features
* Emotion label

## 🏋🏾‍♂️ Train the Model
Train your deep learning emotion classifier:
```bash
python train.py
```
Outputs:
```bash
models/emotion_model.h5
```
Model Architecture:
* Dense (256) + Dropout
* Dense (128) + Dropout
* Softmax output (8 classes

##  Run the Streamlit App
Start the web UI:
```bash
streamlit run app/main.py
```
 Then visit
  http://localhost:8501
Upload a .wav file to get:
*  Predicted emotion
*  Confidence score

  ## Tech Stack
  * Python
  * TensorFlow / TensorFlow-macOS
  * Librosa for audio processing
  * NumPy / Pandas
  * Streamlit for UI

Machine Learning Concepts:
* MFCC feature extraction
* Deep neural networks
* Audio signal processing
* Emotion inference

## Roadmap

* Real-time microphone emotion recognition
*  Probability bar chart visualization
*  Upgrade to Wav2Vec2, HuBERT, or YAMNet
*  Multi-language emotion detection
*   Build a React / React Native UI
*    Deploy on Hugging Face Spaces or Render

