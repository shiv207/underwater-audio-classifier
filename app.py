#!/usr/bin/env python3
"""
Streamlit App for Underwater Acoustic Classification

Usage:
    streamlit run app.py
"""

import os
import sys
import torch
import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core import UnderwaterAcousticClassifier, AudioPreprocessor

# Page config
st.set_page_config(
    page_title="Underwater Acoustic Classifier",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 3rem;
        font-weight: bold;
    }
    .class-name {
        font-size: 2rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load model with caching."""
    device = torch.device('cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get('num_classes', 4)
    
    model = UnderwaterAcousticClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model, device

def plot_spectrogram(log_mel_spec):
    """Create spectrogram plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        log_mel_spec,
        x_axis='time',
        y_axis='mel',
        sr=16000,
        fmax=8000,
        ax=ax,
        cmap='viridis'
    )
    ax.set_title('Log-Mel Spectrogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_probabilities(probs_dict):
    """Create probability bar chart."""
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0].replace('_', ' ').title() for item in sorted_probs]
    probs = [item[1] * 100 for item in sorted_probs]
    
    colors = ['#1f77b4' if i == 0 else '#7f7f7f' for i in range(len(classes))]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(classes, probs, color=colors)
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 2, i, f'{prob:.2f}%', 
               va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🌊 Underwater Acoustic Classifier</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        model_path = st.text_input(
            "Model Path",
            value="models/best_model.pth",
            help="Path to the trained model checkpoint"
        )
        
        st.markdown("---")
        st.subheader("About")
        st.info(
            "**Underwater Acoustic Classifier**\n\n"
            "Deep learning system for classifying underwater sounds:\n\n"
            "- Vessels\n"
            "- Marine Animals\n"
            "- Natural Sounds\n"
            "- Other Anthropogenic"
        )
        
        st.markdown("---")
        st.subheader("Model Info")
        if os.path.exists(model_path):
            st.success("Model found")
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                st.write(f"**Classes**: {checkpoint.get('num_classes', 4)}")
                if 'balanced_accuracy' in checkpoint:
                    st.write(f"**Balanced Acc**: {checkpoint['balanced_accuracy']:.2f}%")
            except:
                pass
        else:
            st.error("Model not found")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=['wav'],
            help="Upload an underwater acoustic recording (.wav format)"
        )
        
        # Sample files selector
        st.markdown("**Or select a sample file:**")
        sample_dir = "data/training"
        sample_path = None
        
        if os.path.exists(sample_dir):
            categories = [d for d in os.listdir(sample_dir) 
                         if os.path.isdir(os.path.join(sample_dir, d)) and not d.startswith('.')]
            
            if categories:
                selected_category = st.selectbox("Category", categories)
                category_path = os.path.join(sample_dir, selected_category)
                
                audio_files = [f for f in os.listdir(category_path) 
                              if f.endswith('.wav')]
                
                if audio_files:
                    selected_file = st.selectbox("File", audio_files)
                    sample_path = os.path.join(category_path, selected_file)
                    
                    if st.button("Use Sample File"):
                        uploaded_file = sample_path
    
    with col2:
        st.subheader("📋 Instructions")
        st.markdown(
            """
            1. Upload a WAV audio file or select a sample
            2. Click 'Classify Audio' to analyze
            3. View prediction results and visualizations
            
            **Supported formats**: WAV (mono/stereo, any sample rate)
            """
        )
    
    st.markdown("---")
    
    # Process audio
    if uploaded_file is not None:
        if not os.path.exists(model_path):
            st.error(f"Model not found: {model_path}")
            st.info("Please train a model first using: `python train.py --data-dir data/training --epochs 50`")
            return
        
        # Display audio player
        st.subheader("🔊 Audio Player")
        
        if isinstance(uploaded_file, str):
            audio_path = uploaded_file
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
        else:
            audio_bytes = uploaded_file.read()
            audio_path = f"/tmp/{uploaded_file.name}"
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
        
        st.audio(audio_bytes, format='audio/wav')
        
        # Classify button
        if st.button("🚀 Classify Audio", type="primary", use_container_width=True):
            with st.spinner("Processing audio..."):
                try:
                    # Load model
                    model, device = load_model(model_path)
                    
                    # Preprocess
                    preprocessor = AudioPreprocessor()
                    audio, log_mel_spec, metadata = preprocessor.process_audio_file(audio_path)
                    
                    # Classify
                    spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(spec_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    # Get class names
                    class_names = model.class_names
                    
                    # Get prediction
                    pred_idx = probabilities.argmax().item()
                    confidence = probabilities[pred_idx].item()
                    predicted_class = class_names[pred_idx]
                    
                    # Create probabilities dict
                    probs_dict = {class_names[i]: probabilities[i].item() 
                                 for i in range(len(probabilities))}
                    
                    st.success("Classification complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Classification Results")
                    
                    # Prediction box
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <div class="confidence-score">{confidence*100:.1f}%</div>
                            <div class="class-name">{predicted_class.replace('_', ' ').title()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Two columns for visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        st.pyplot(plot_spectrogram(log_mel_spec))
                    
                    with viz_col2:
                        st.pyplot(plot_probabilities(probs_dict))
                    
                    # Detailed results
                    st.markdown("---")
                    st.subheader("📊 Detailed Results")
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.markdown("**Audio Info**")
                        st.write(f"Duration: {metadata.get('duration', 0):.2f}s")
                        st.write(f"Sample Rate: {metadata.get('sample_rate', 0)} Hz")
                    
                    with result_col2:
                        st.markdown("**All Probabilities**")
                        for cls, prob in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"{cls.replace('_', ' ').title()}: {prob*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error during classification: {e}")
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == '__main__':
    main()
