# Model Training Guide - Improving Accuracy

## Quick Summary

| Model | Accuracy | Training Time | Complexity | Recommended For |
|-------|----------|---------------|------------|-----------------|
| **Baseline** (train.py) | 65.69% | ~5 min | Low | Testing only |
| **Enhanced** (train_enhanced.py) | 80.21% ✅ | ~15 min | Medium | **Production use** |
| **Advanced** (train_advanced.py) | 14.47% ❌ | ~25 min | High | **FAILED - Do not use** |
| **Optimized** (train_optimized.py) | **Target: 85%+** | ~20 min | Medium-High | **Best single model** |
| **Ensemble** (train_ensemble.py) | **Target: 85-90%** | ~90 min | High | **Maximum accuracy** |

---

## Recommended Approach

### Option 1: Quick Improvement (Recommended)
**Use the Optimized Model**

```bash
python train_optimized.py
```

**Why:**
- Builds on proven enhanced architecture (80.21%)
- Adds cutting-edge techniques that are stable:
  - Multi-scale CNNs (capture patterns at different time scales)
  - Bidirectional LSTM (temporal context)
  - Squeeze-Excitation blocks (attention mechanism)
  - Mixup augmentation (smoother decision boundaries)
- Target: **85%+ accuracy**
- Realistic training time: ~20 minutes
- Single model - easy to deploy

### Option 2: Maximum Accuracy
**Use the Ensemble Model**

```bash
python train_ensemble.py
```

**Why:**
- Trains 5 diverse models (CNN, LSTM, Hybrid)
- Combines predictions via soft voting
- Typically adds 2-5% over best single model
- Target: **85-90% accuracy**
- Longer training time: ~90 minutes
- More complex deployment (need to load 5 models)

---

## What Went Wrong with the Advanced Model?

The advanced model (14.47% accuracy) failed due to:

1. **Focal Loss Instability**
   - Focal loss is powerful but sensitive to hyperparameters
   - Without careful tuning, it can cause training collapse
   - Model converged to predicting almost everything as "surprised"

2. **Over-regularization**
   - Too many dropout layers (0.3-0.5) combined with heavy L2 regularization
   - Model couldn't learn complex patterns
   - Gradients became too small

3. **Learning Rate Schedule Issues**
   - Cosine decay might have reduced LR too quickly
   - Model got stuck in poor local minimum early in training

**Lesson:** More complex != better. Stick with proven, stable techniques.

---

## Architecture Comparison

### Enhanced Model (80.21% - Currently Best)
```
Input → Reshape →
CNN Blocks (128→256→512→256 filters) →
Global Pooling →
Dense (512→256→128) →
Output
```

**Strengths:**
- Solid CNN architecture for audio
- Good data augmentation (4x samples)
- Rich features (392 dimensions)
- Stable training

**Weaknesses:**
- No temporal modeling (LSTM)
- Single-scale convolutions
- No attention mechanism

### Optimized Model (Target: 85%+)
```
Input → Reshape →
Multi-scale CNN Blocks (parallel 3,5,7 kernels) →
Squeeze-Excitation (attention) →
Bidirectional LSTM (128→64) →
Dense (256→128) →
Output
```

**Improvements:**
- Multi-scale feature extraction
- Temporal context via LSTM
- Channel attention via SE blocks
- Mixup augmentation
- Better regularization balance

### Ensemble Model (Target: 85-90%)
```
5 Models:
- 2x CNN (different hyperparameters)
- 2x LSTM
- 1x Hybrid CNN-LSTM

Soft Voting: Average probabilities → Final prediction
```

**Improvements:**
- Model diversity reduces overfitting
- Ensemble typically beats best single model
- More robust to individual model failures

---

## Step-by-Step: Training the Optimized Model

### 1. Ensure Data is Ready
```bash
# Check if enhanced features exist
ls data/processed/features_enhanced.csv

# If not, run preprocessing
python preprocessing_enhanced.py
```

### 2. Train the Optimized Model
```bash
python train_optimized.py
```

Expected output:
```
OPTIMIZED EMOTION RECOGNITION TRAINING
======================================================================
Optimizations:
  1. Multi-scale CNN (capture patterns at different time scales)
  2. Bidirectional LSTM (temporal context in both directions)
  3. Squeeze-Excitation blocks (channel attention)
  4. Mixup augmentation (smoother decision boundaries)
  ...

Training...
Epoch 1/200
...
Test Accuracy: 85.xx%  ← Target!
```

### 3. Evaluate Results
The script will output:
- Test accuracy
- Per-class performance
- Confusion matrix
- Training history plot

### 4. Update Your App (if satisfied)
```bash
# Backup current model
cp models/emotion_model.h5 models/emotion_model_backup.h5

# Use the new model
cp models/emotion_model_optimized.h5 models/emotion_model.h5
cp models/label_encoder_optimized.pkl models/label_encoder.pkl
cp models/scaler_optimized.pkl models/scaler.pkl

# Test with your app
streamlit run app/main.py
```

---

## Understanding the Improvements

### 1. Multi-Scale CNNs
**Problem:** Audio emotions manifest at different time scales
- Short-term: Voice pitch variations (100-200ms)
- Medium-term: Word-level prosody (500ms-1s)
- Long-term: Sentence-level intonation (2-3s)

**Solution:** Use parallel convolutions with different kernel sizes (3, 5, 7)
- Captures patterns at multiple time scales simultaneously
- Similar to Inception architecture

### 2. Squeeze-Excitation Blocks
**Problem:** Not all features are equally important
- Some MFCC coefficients are more discriminative than others
- Model should focus on relevant channels

**Solution:** Channel attention mechanism
- Learn to weight feature channels by importance
- Improves feature representation

### 3. Bidirectional LSTM
**Problem:** Emotions evolve over time
- Context from both past and future helps
- Current frame + surrounding context = better prediction

**Solution:** BiLSTM layers
- Process sequence forward and backward
- Capture temporal dependencies

### 4. Mixup Augmentation
**Problem:** Decision boundaries might be too rigid
- Model might overfit to specific training examples
- Need smoother, more generalizable boundaries

**Solution:** Blend pairs of training samples
- Creates virtual training samples
- Encourages smooth predictions
- Improves generalization

---

## Troubleshooting

### If Optimized Model < 80% Accuracy

**Check 1: Data Quality**
```bash
# Verify features file
python -c "import pandas as pd; df = pd.read_csv('data/processed/features_enhanced.csv'); print(f'Shape: {df.shape}'); print(df['emotion'].value_counts())"
```

Expected: 5,760 samples, 392 features

**Check 2: Training Stability**
Look at the training history plot (`models/training_history_optimized.png`)
- Val accuracy should increase steadily
- If erratic or decreasing → reduce learning rate

**Check 3: Overfitting**
- If train accuracy >> test accuracy → increase dropout
- If both are low → reduce dropout, increase model capacity

### If Ensemble Model Takes Too Long

Reduce `N_MODELS` in [train_ensemble.py](train_ensemble.py):
```python
N_MODELS = 3  # Instead of 5
```

---

## Advanced: Further Improvements

If you achieve 85%+ but want even more:

### 1. More Augmentation
Edit [preprocessing_enhanced.py](preprocessing_enhanced.py):
```python
AUGMENTATION_FACTOR = 5  # Instead of 3
```

Pros: More diverse training data
Cons: Longer preprocessing (10-15 min)

### 2. Additional Datasets
Combine RAVDESS with other emotion datasets:
- **TESS**: Toronto Emotional Speech Set
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset
- **EmoDB**: Berlin Database of Emotional Speech

Pros: More training data, better generalization
Cons: Need to download and preprocess additional datasets

### 3. Transfer Learning
Use pre-trained audio models:
- **VGGish**: Audio event classification
- **YAMNet**: Yamaha audio model
- **Wav2Vec 2.0**: Self-supervised speech representation

Pros: Leverage large-scale pre-training
Cons: More complex implementation, larger models

### 4. Cross-Modal Learning
Combine audio with:
- Text transcriptions (sentiment analysis)
- Facial expressions (if video available)
- Physiological signals (if available)

Pros: Richer information
Cons: Need additional modalities

---

## Production Deployment

### Using the Optimized Model
```python
from app.utils import EmotionPredictor

# Load optimized model
predictor = EmotionPredictor(
    model_path="models/emotion_model_optimized.h5",
    encoder_path="models/label_encoder_optimized.pkl",
    scaler_path="models/scaler_optimized.pkl"
)

# Make prediction
result = predictor.predict("path/to/audio.wav")
print(f"Emotion: {result['emotion']} ({result['confidence']:.1%})")
```

### Using the Ensemble
```python
import numpy as np
from tensorflow import keras
import pickle

# Load all ensemble models
models = []
for i in range(5):
    model = keras.models.load_model(f"models/ensemble/model_{i}.h5")
    models.append(model)

# Load scaler and encoder
with open("models/scaler_ensemble.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/label_encoder_ensemble.pkl", "rb") as f:
    encoder = pickle.load(f)

# Extract features (using your preprocessing code)
features = extract_features("path/to/audio.wav")
features = scaler.transform([features])

# Get predictions from all models
predictions = []
for model in models:
    pred = model.predict(features, verbose=0)
    predictions.append(pred)

# Soft voting
ensemble_pred = np.mean(predictions, axis=0)
emotion_idx = np.argmax(ensemble_pred)
emotion = encoder.classes_[emotion_idx]
confidence = ensemble_pred[0][emotion_idx]

print(f"Emotion: {emotion} ({confidence:.1%})")
```

---

## Summary

**For quick, reliable improvement:**
```bash
python train_optimized.py  # Target: 85%+, ~20 min
```

**For maximum accuracy:**
```bash
python train_ensemble.py   # Target: 85-90%, ~90 min
```

**Current best working model:**
- Enhanced CNN: 80.21% (stable, production-ready)

**Failed model to avoid:**
- Advanced model: 14.47% (training collapsed)

Good luck! The optimized model should get you to 85%+ with stable, proven techniques.
