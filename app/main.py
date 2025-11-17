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
    model_path = "../models/emotion_model.h5"
    encoder_path = "../models/label_encoder.pkl"
    scaler_path = "../models/scaler.pkl"

    return EmotionPredictor(model_path, encoder_path, scaler_path)


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
    st.sidebar.info(
        """
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

        **Model Details:**
        - Architecture: Deep Neural Network
        - Features: MFCC (Mel-Frequency Cepstral Coefficients)
        - Training Data: RAVDESS Dataset
        - Accuracy: ~66%
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
