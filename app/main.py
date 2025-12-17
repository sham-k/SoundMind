# -*- coding: utf-8 -*-
import streamlit as st
import os
import tempfile
from utils import EmotionPredictor
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="SoundMind - Emotion Recognition",
    page_icon=":headphones:",
    layout="centered"
)

# Emotion to color mapping for visualization
EMOTION_COLORS = {
    'angry': '#FF4444',
    'calm': '#88D8B0',
    'disgust': '#9B59B6',
    'fearful': '#FF8C00',
    'happy': '#FFD700',
    'neutral': '#808080',
    'sad': '#4169E1',
    'surprised': '#FF69B4'
}


@st.cache_resource
def load_predictor():
    """Load the emotion predictor model (cached)."""
    import os
    from pathlib import Path

    # Get the directory of this file (app/main.py)
    current_file = Path(__file__).resolve()  # Get absolute path
    app_dir = current_file.parent
    project_root = app_dir.parent

    # Try multiple possible locations for models directory
    possible_model_dirs = [
        project_root / "models",           # ../models from app/main.py (most reliable)
        Path.cwd().parent / "models",      # When running from app/ directory
        Path.cwd() / "models",             # When running from project root
    ]

    models_dir = None
    for dir_path in possible_model_dirs:
        resolved_path = dir_path.resolve()  # Get absolute path
        if resolved_path.exists() and resolved_path.is_dir():
            # Verify it actually contains model files
            model_files = list(resolved_path.glob("*.h5"))
            if model_files:
                models_dir = resolved_path
                break

    if models_dir is None:
        raise FileNotFoundError(
            f"Models directory not found or empty.\n"
            f"Searched locations:\n" +
            "\n".join([f"  - {p.resolve()}" for p in possible_model_dirs]) +
            f"\n\nCurrent working directory: {Path.cwd()}\n"
            f"Script location: {app_dir}\n"
            f"Project root: {project_root}\n\n"
            f"Please ensure models are in: {project_root / 'models'}"
        )

    # Try to load optimized model first, fall back to others
    model_options = [
        ("emotion_model_optimized.h5", "label_encoder_optimized.pkl", "scaler_optimized.pkl", "85.07%"),
        ("emotion_model_enhanced.h5", "label_encoder_enhanced.pkl", "scaler_enhanced.pkl", "80.21%"),
        ("emotion_model.h5", "label_encoder.pkl", "scaler.pkl", "65.69%"),
    ]

    for model_file, encoder_file, scaler_file, accuracy in model_options:
        model_path = str(models_dir / model_file)
        encoder_path = str(models_dir / encoder_file)
        scaler_path = str(models_dir / scaler_file)

        if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(scaler_path):
            predictor = EmotionPredictor(model_path, encoder_path, scaler_path)
            predictor.model_name = model_file
            predictor.model_accuracy = accuracy
            return predictor

    # If no model found, provide helpful error message
    raise FileNotFoundError(
        f"No trained model found in {models_dir}.\n"
        f"Please ensure you have trained a model by running:\n"
        f"  python train_optimized.py  (recommended)\n"
        f"or:\n"
        f"  python train_enhanced.py\n"
        f"  python train.py"
    )


def create_probability_chart(probabilities):
    """Create a horizontal bar chart of emotion probabilities."""
    # Sort by probability
    sorted_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions = [e[0].capitalize() for e in sorted_emotions]
    probs = [e[1] * 100 for e in sorted_emotions]
    colors = [EMOTION_COLORS[e[0]] for e in sorted_emotions]

    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p:.1f}%' for p in probs],
        textposition='auto',
    ))

    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Confidence (%)",
        yaxis_title="Emotion",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def main():
    # Header
    st.title(":headphones: SoundMind")
    st.subheader("AI-Powered Emotion Recognition from Voice")
    st.markdown("Upload a voice recording and discover the emotional state detected by our deep learning model.")

    st.markdown("---")

    # Load model
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Make sure the model has been trained and saved in the `models/` directory.")
        return

    # File upload
    st.markdown("### :arrow_up: Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=['wav'],
        help="Upload a .wav file containing speech (3-5 seconds works best)"
    )

    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Predict button
        if st.button(":crystal_ball: Analyze Emotion", type="primary"):
            with st.spinner("Analyzing audio..."):
                try:
                    # Make prediction
                    result = predictor.predict(tmp_path)

                    # Display results
                    st.markdown("---")
                    st.markdown("### :dart: Results")

                    # Main emotion result
                    emotion = result['emotion']
                    confidence = result['confidence'] * 100

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Detected Emotion",
                            value=emotion.upper()
                        )
                    with col2:
                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.1f}%"
                        )

                    # Probability distribution chart
                    st.plotly_chart(
                        create_probability_chart(result['probabilities']),
                        use_container_width=True
                    )

                    # Detailed probabilities
                    with st.expander(":bar_chart: View Detailed Probabilities"):
                        sorted_probs = sorted(
                            result['probabilities'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        for emo, prob in sorted_probs:
                            st.write(f"**{emo.capitalize()}**: {prob*100:.2f}%")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.info("Please ensure the audio file is a valid WAV format with speech content.")

        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Sidebar info
    st.sidebar.title(":information_source: About")

    # Show model information if available
    model_name = getattr(predictor, 'model_name', 'emotion_model.h5')
    model_accuracy = getattr(predictor, 'model_accuracy', 'Unknown')

    # Determine architecture based on model name
    if 'optimized' in model_name:
        architecture = "CNN + BiLSTM + Attention"
        features = "MFCC, Chroma, Mel, Contrast, Tonnetz, ZCR, Spectral (392 dims)"
    elif 'enhanced' in model_name:
        architecture = "Hybrid CNN"
        features = "MFCC, Chroma, Mel, Contrast, Tonnetz, ZCR, Spectral (392 dims)"
    else:
        architecture = "Dense Neural Network"
        features = "MFCC (40 coefficients)"

    st.sidebar.info(
        f"""
        **SoundMind** uses deep learning to analyze voice recordings
        and predict the speaker's emotional state.

        **Supported Emotions:**
        - Angry
        - Calm
        - Disgust
        - Fearful
        - Happy
        - Neutral
        - Sad
        - Surprised

        **Active Model:**
        - Model: {model_name}
        - Architecture: {architecture}
        - Features: {features}
        - Training Data: RAVDESS Dataset
        - Test Accuracy: {model_accuracy}
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**How to use:**")
    st.sidebar.markdown(
        """
        1. Upload a WAV audio file
        2. Click 'Analyze Emotion'
        3. View the predicted emotion and confidence scores
        """
    )


if __name__ == "__main__":
    main()
