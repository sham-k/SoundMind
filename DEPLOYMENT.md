# Deployment Guide for SoundMind

## Quick Deployment to Streamlit Cloud

### Prerequisites
- GitHub repository is already set up and pushed ✅
- All model files are committed ✅
- Requirements.txt is ready ✅

### Step-by-Step Deployment

#### 1. Go to Streamlit Cloud
Visit: https://share.streamlit.io/

#### 2. Sign in with GitHub
- Click "Sign in with GitHub"
- Authorize Streamlit Cloud

#### 3. Deploy New App
1. Click "New app"
2. Fill in the details:
   - **Repository**: `sham-k/SoundMind`
   - **Branch**: `main`
   - **Main file path**: `app/main.py`
   - **App URL**: Choose your preferred URL (e.g., `soundmind-emotion-ai`)

#### 4. Advanced Settings (Optional)
- **Python version**: 3.11 (recommended)
- **Secrets**: Not needed for this app

#### 5. Deploy!
Click "Deploy!" and wait 2-3 minutes for the app to build.

---

## What Gets Deployed

### ✅ Included in Deployment
- Optimized model (85.07% accuracy) - 10.1 MB
- Enhanced model (80.21% accuracy) - 16.8 MB
- Baseline model (65.69% accuracy) - 2.3 MB
- All preprocessing and prediction code
- Streamlit web interface

### 📊 Model Priority
The app will automatically load models in this order:
1. **emotion_model_optimized.h5** (85.07%) ← Primary
2. emotion_model_enhanced.h5(80.21%) ← Fallback
3. emotion_model.h5 (65.69%) ← Last resort

### 🎯 Expected Performance
- **Model**: Optimized (85.07% accuracy)
- **Load time**: ~5-10 seconds (first load)
- **Prediction time**: ~1-2 seconds per audio file
- **Supported format**: WAV files only

---

## Troubleshooting

### Issue: "Models directory not found"
**Solution**: Ensure model files are committed to git:
```bash
git add models/*.h5 models/*.pkl
git commit -m "Add trained models"
git push
```

### Issue: "Module not found" errors
**Solution**: Check requirements.txt includes:
```
numpy
pandas
scikit-learn
librosa
soundfile
streamlit
plotly
tensorflow
```

### Issue: Memory limit exceeded
**Cause**: Streamlit Cloud has 1GB memory limit
**Solution**: The optimized model (10MB) fits comfortably. If needed:
1. Use only the optimized model (remove enhanced and baseline)
2. Or upgrade to Streamlit Cloud Teams

### Issue: App is slow
**Solutions**:
- First load is slow (model loading)
- Subsequent predictions are fast (model is cached)
- Consider reducing model size if needed

---

## Post-Deployment

### Verify Deployment
1. Visit your app URL
2. Upload a test WAV file
3. Check sidebar shows: "Test Accuracy: 85.07%"
4. Verify emotion prediction works

### Share Your App
- App URL: `https://[your-app-name].streamlit.app`
- Share with users!
- No authentication needed - fully public

### Monitor Usage
- View metrics at: https://share.streamlit.io/
- See number of visitors, errors, etc.

### Update the App
To deploy updates:
```bash
# Make changes locally
git add .
git commit -m "Update: description"
git push

# Streamlit Cloud auto-deploys on push!
```

---

## Alternative: Local Deployment

### Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t soundmind .
docker run -p 8501:8501 soundmind
```

### Heroku Deployment (Alternative)

Not recommended due to:
- Slug size limits (500MB)
- Complex buildpacks needed for audio libraries
- Use Streamlit Cloud instead (easier)

---

## Performance Optimization

### For Production Use

1. **Enable caching** (already implemented):
   ```python
   @st.cache_resource
   def load_predictor():
       # Model is loaded once and cached
   ```

2. **Use smaller model** if needed:
   - Keep only optimized model (10MB)
   - Remove enhanced (16MB) and baseline (2MB)

3. **Add rate limiting** (if needed):
   - Limit predictions per user
   - Add cooldown between predictions

4. **Monitor performance**:
   - Use Streamlit analytics
   - Track prediction times
   - Monitor error rates

---

## Security Notes

### Data Privacy
- Audio files are processed in-memory
- Files are deleted after prediction
- No data is stored permanently

### Model Security
- Models are public (in git repo)
- No sensitive data in models
- Training data (RAVDESS) is public research dataset

---

## Estimated Costs

### Streamlit Cloud (Recommended)
- **Free tier**: Perfect for this app
  - 1GB RAM (sufficient)
  - 1 CPU core
  - Unlimited bandwidth
  - Public apps

- **If needed - Teams tier**: $250/month
  - 8GB RAM
  - 4 CPU cores
  - Private apps
  - Custom domain

### Our App Requirements
- **RAM**: ~300MB (well under 1GB limit)
- **CPU**: Low (inference is fast)
- **Storage**: ~30MB (models + code)
- **Bandwidth**: Minimal (audio files are small)

**Verdict**: ✅ Free tier is perfect for SoundMind!

---

## Next Steps

1. ✅ Code is pushed to GitHub
2. 🚀 Deploy to Streamlit Cloud (follow steps above)
3. 🧪 Test the deployed app
4. 📢 Share the URL!

Your app is production-ready with 85.07% accuracy! 🎉
