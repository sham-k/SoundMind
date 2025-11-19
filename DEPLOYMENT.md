# 🚀 SoundMind Deployment Guide

This guide will help you deploy SoundMind online for free using **Streamlit Cloud**.

---

## Option 1: Streamlit Cloud (Recommended - FREE & Easy!)

Streamlit Cloud is the easiest way to deploy. It's **100% free** for public apps!

### Prerequisites
- GitHub account
- Your SoundMind code pushed to GitHub

### Step-by-Step Instructions

#### 1. Push Your Code to GitHub

If you haven't already:

```bash
# Check what files will be committed
git status

# Add all new files
git add .

# Commit the changes
git commit -m "Prepare SoundMind for deployment

- Add trained emotion recognition model
- Create Streamlit web interface
- Add deployment configuration files
- Update documentation

Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to GitHub
git push origin main
```

#### 2. Sign Up for Streamlit Cloud

1. Go to: **https://streamlit.io/cloud**
2. Click **"Sign up"**
3. Sign in with your **GitHub account**
4. Authorize Streamlit to access your repositories

#### 3. Deploy Your App

1. Click **"New app"** button
2. Fill in the details:
   - **Repository**: Select `your-username/Soundmind`
   - **Branch**: `main`
   - **Main file path**: `app/main.py`
   - **App URL** (optional): Choose a custom subdomain like `soundmind-emotion`

3. Click **"Deploy!"**

4. Wait 2-3 minutes for deployment

5. **Done!** Your app will be live at:
   ```
   https://your-app-name.streamlit.app
   ```

### Troubleshooting

**Issue: "ModuleNotFoundError"**
- Solution: Make sure `requirements.txt` is in the root directory
- Check all packages are listed correctly

**Issue: "Model file not found"**
- Solution: Make sure `models/` directory is NOT in `.gitignore`
- Push the models to GitHub: `git add models/ && git commit -m "Add models" && git push`

**Issue: "App keeps crashing"**
- Solution: Check the logs in Streamlit Cloud dashboard
- Look for missing dependencies in `requirements.txt` or `packages.txt`

---

## Option 2: Hugging Face Spaces (Alternative - FREE)

Another great free option with GPU support!

### Steps:

1. **Create account** at https://huggingface.co/join

2. **Create new Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose **Streamlit** as SDK
   - Name it (e.g., "soundmind-emotion")

3. **Upload files**:
   - Clone the Space repository locally
   - Copy your SoundMind files into it
   - Push to Hugging Face

```bash
git clone https://huggingface.co/spaces/your-username/soundmind-emotion
cd soundmind-emotion

# Copy your files
cp -r /path/to/Soundmind/* .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

4. **App will be live** at:
   ```
   https://huggingface.co/spaces/your-username/soundmind-emotion
   ```

---

## Option 3: Railway.app (Paid but Easy)

Railway offers easy deployment with a free tier ($5 credit/month).

### Steps:

1. **Sign up** at https://railway.app

2. **Click "New Project"** → "Deploy from GitHub repo"

3. **Select** your SoundMind repository

4. **Add variables** (if needed):
   - None required for basic setup

5. **Deploy!**

Railway will auto-detect it's a Python app and deploy it.

---

## Option 4: Render (Free Tier Available)

Render offers 750 hours/month free.

### Steps:

1. **Sign up** at https://render.com

2. **Create** "New Web Service"

3. **Connect** your GitHub repository

4. **Configure**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0`

5. **Deploy!**

---

## Files Required for Deployment

Make sure these files are in your repository:

✅ `app/main.py` - Main Streamlit app
✅ `app/utils.py` - Prediction utilities
✅ `models/emotion_model.h5` - Trained model
✅ `models/label_encoder.pkl` - Label encoder
✅ `models/scaler.pkl` - Feature scaler
✅ `requirements.txt` - Python dependencies
✅ `packages.txt` - System dependencies (for Streamlit Cloud)
✅ `.streamlit/config.toml` - Streamlit configuration

---

## Post-Deployment

### Share Your App

Once deployed, share your app:
- Add the URL to your GitHub README
- Share on LinkedIn/Twitter
- Add to your portfolio

### Monitor Usage

Most platforms provide:
- Usage statistics
- Error logs
- Performance metrics

### Update Your App

To update after deployment:
```bash
# Make changes locally
git add .
git commit -m "Update: description of changes"
git push

# Streamlit Cloud will auto-redeploy!
```

---

## Cost Comparison

| Platform | Free Tier | Best For |
|----------|-----------|----------|
| **Streamlit Cloud** | Unlimited public apps | Quick demos, portfolios |
| **Hugging Face** | Unlimited Spaces | ML/AI projects |
| **Railway** | $5/month credit | Production apps |
| **Render** | 750 hrs/month | Side projects |

---

## Recommended: Streamlit Cloud

**Why Streamlit Cloud?**
✅ Completely FREE
✅ Easiest setup (3 clicks!)
✅ Auto-deploys from GitHub
✅ Built specifically for Streamlit apps
✅ No credit card required

**Your app will be live in under 5 minutes!**

---

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Sign up for Streamlit Cloud
3. ✅ Deploy your app
4. ✅ Share your live URL!

Good luck! 🚀
