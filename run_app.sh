#!/bin/bash
# Launch script for SoundMind Streamlit app
# This ensures the app runs from the correct directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project root
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if models exist
if [ ! -d "models" ] || [ ! -f "models/emotion_model_optimized.h5" ]; then
    echo "WARNING: Optimized model not found!"
    echo "The app will try to use available models in priority order:"
    echo "  1. emotion_model_optimized.h5 (85.07%) - Not found"
    echo "  2. emotion_model_enhanced.h5 (80.21%)"
    echo "  3. emotion_model.h5 (65.69%)"
    echo ""
fi

# Run Streamlit from project root
# This ensures paths resolve correctly
echo "Starting SoundMind Emotion Recognition App..."
echo "The app will automatically use the best available model."
echo "Opening in browser at http://localhost:8501"
echo ""
streamlit run app/main.py
