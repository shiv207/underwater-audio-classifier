import streamlit as st
import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import json
from datetime import datetime
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import AudioPreprocessor
from classifier import AcousticClassifier
from detector import SimpleEnergyDetector

st.set_page_config(
    page_title="Underwater Acoustic Classifier",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = AudioPreprocessor()
if 'detector' not in st.session_state:
    st.session_state.detector = SimpleEnergyDetector()

CATEGORIES = {
    1: {
        'name': 'Vessels',
        'description': 'Ships, boats, and other watercraft',
        'examples': 'Cargo ships, submarines, fishing vessels',
        'color': '#667eea'
    },
    2: {
        'name': 'Marine Animals',
        'description': 'Biological sounds from marine life',
        'examples': 'Whales, dolphins, fish vocalizations',
        'color': '#48c6ef'
    },
    3: {
        'name': 'Natural Sounds',
        'description': 'Non-biological natural phenomena',
        'examples': 'Earthquakes, underwater landslides, ice cracking',
        'color': '#0ba360'
    },
    4: {
        'name': 'Other Anthropogenic',
        'description': 'Man-made sounds (non-vessel)',
        'examples': 'Sonar, torpedoes, underwater construction, turbines',
        'color': '#f093fb'
    }
}

def load_model():
    
    model_path = 'models/best_classifier.pth'
    
    if os.path.exists(model_path):
        try:
            classifier = AcousticClassifier(model_path=model_path)
            return classifier, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, False
    else:
        st.warning("⚠️ No trained model found. Please train the model first using `python retrain_model.py`")
        return None, False

def create_spectrogram_plot(audio, sr, log_mel_spec):
    
    fig = go.Figure()
    
    times = np.linspace(0, len(audio) / sr, log_mel_spec.shape[1])
    frequencies = librosa.mel_frequencies(n_mels=log_mel_spec.shape[0], fmin=0, fmax=sr/2)
    
    fig.add_trace(go.Heatmap(
        z=log_mel_spec,
        x=times,
        y=frequencies,
        colorscale='Viridis',
        colorbar=dict(title='dB'),
        hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.0f}Hz<br>Amplitude: %{z:.2f}dB<extra></extra>'
    ))
    
    fig.update_layout(
        title='Mel Spectrogram',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        height=400,
        hovermode='closest'
    )
    
    return fig

def create_waveform_plot(audio, sr):
    
    times = np.linspace(0, len(audio) / sr, len(audio))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=audio,
        mode='lines',
        line=dict(color='#48c6ef', width=1),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=250,
        showlegend=False
    )
    
    return fig

def create_confidence_chart(probabilities):
    
    categories = []
    confidences = []
    colors = []
    
    for class_name, prob in probabilities.items():
        if 'vessel' in class_name.lower():
            cat_id = 1
        elif 'marine' in class_name.lower() or 'animal' in class_name.lower():
            cat_id = 2
        elif 'natural' in class_name.lower():
            cat_id = 3
        else:
            cat_id = 4
        
        categories.append(CATEGORIES[cat_id]['name'])
        confidences.append(prob * 100)
        colors.append(CATEGORIES[cat_id]['color'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=categories,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{c:.1f}%' for c in confidences],
            textposition='auto',
            hovertemplate='%{y}<br>Confidence: %{x:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Classification Confidence',
        xaxis_title='Confidence (%)',
        yaxis_title='Category',
        height=300,
        showlegend=False,
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def process_audio_file(audio_file):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name
    
    try:
        audio, log_mel_spec, metadata = st.session_state.preprocessor.process_audio_file(tmp_path)
        
        if len(audio) == 0:
            st.error("Failed to process audio file")
            return None
        
        events = st.session_state.detector.detect_events(audio)
        
        if st.session_state.classifier:
            classification_result = st.session_state.classifier.classify_spectrogram(log_mel_spec)
        else:
            classification_result = {
                'category_id': 3,
                'confidence': 0.0,
                'probabilities': {}
            }
        
        return {
            'audio': audio,
            'log_mel_spec': log_mel_spec,
            'metadata': metadata,
            'events': events,
            'classification': classification_result
        }
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

st.markdown('<p class="main-header">🌊 Underwater Acoustic Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Military-Grade Underwater Sound Classification System</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model Status")
    if st.button("🔄 Load/Reload Model"):
        with st.spinner("Loading model..."):
            classifier, success = load_model()
            if success:
                st.session_state.classifier = classifier
                st.success("✅ Model loaded successfully!")
            else:
                st.session_state.classifier = None
    
    model_status = "🟢 Loaded" if st.session_state.classifier else "🔴 Not Loaded"
    st.info(f"**Status:** {model_status}")
    
    st.divider()
    
    st.subheader("📚 Categories")
    for cat_id, cat_info in CATEGORIES.items():
        with st.expander(f"{cat_info['name']}", expanded=False):
            st.write(f"**Description:** {cat_info['description']}")
            st.write(f"**Examples:** {cat_info['examples']}")
    
    st.divider()
    
    st.subheader("📖 Instructions")
    st.markdown("""
    1. **Load Model** - Click the button in the sidebar
    2. **Upload Audio** - Use the file uploader below
    3. **Classify** - Click the classify button to get results
    4. **Export** - Download results as JSON for further analysis
    """)

tab1, tab2, tab3 = st.tabs(["🎵 Upload & Classify", "📊 Batch Processing", "ℹ️ About"])

with tab1:
    st.header("Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload an underwater audio recording for classification"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("File Info")
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🔍 Classify Audio", type="primary", use_container_width=True):
            if not st.session_state.classifier:
                st.error("⚠️ Please load the model first!")
            else:
                with st.spinner("Processing audio..."):
                    result = process_audio_file(uploaded_file)
                    
                    if result:
                        st.success("✅ Classification complete!")
                        
                        st.divider()
                        st.header("📊 Results")
                        
                        cat_id = result['classification']['category_id']
                        confidence = result['classification']['confidence']
                        category_info = CATEGORIES[cat_id]
                        
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, {category_info['color']} 0%, #667eea 100%); 
                        border-radius: 10px; color: white; text-align: center;">
                            <h2 style="margin: 0; color: white;">🎯 {category_info['name']}</h2>
                            <p style="margin: 5px 0; font-size: 1.2em;">Confidence: {confidence*100:.1f}%</p>
                            <p style="margin: 5px 0; opacity: 0.9;">{category_info['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Duration", f"{result['metadata']['duration']:.2f}s")
                        with col2:
                            st.metric("Sample Rate", f"{result['metadata']['sample_rate']} Hz")
                        with col3:
                            st.metric("Events Detected", len(result['events']))
                        with col4:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        st.divider()
                        st.subheader("📈 Visualizations")
                        
                        if result['classification']['probabilities']:
                            st.plotly_chart(
                                create_confidence_chart(result['classification']['probabilities']),
                                use_container_width=True
                            )
                        
                        st.plotly_chart(
                            create_waveform_plot(result['audio'], result['metadata']['sample_rate']),
                            use_container_width=True
                        )
                        
                        st.plotly_chart(
                            create_spectrogram_plot(
                                result['audio'],
                                result['metadata']['sample_rate'],
                                result['log_mel_spec']
                            ),
                            use_container_width=True
                        )
                        
                        with st.expander("🔍 Detailed Classification Probabilities"):
                            prob_df = pd.DataFrame([
                                {'Category': k, 'Probability': f"{v*100:.2f}%"}
                                for k, v in result['classification']['probabilities'].items()
                            ])
                            st.dataframe(prob_df, use_container_width=True)
                        
                        if result['events']:
                            with st.expander(f"📍 Detected Events ({len(result['events'])})"):
                                events_df = pd.DataFrame(result['events'])
                                st.dataframe(events_df, use_container_width=True)
                        
                        st.divider()
                        export_data = {
                            'info': {
                                'file_name': uploaded_file.name,
                                'classification_date': datetime.now().isoformat(),
                                'version': '1.0'
                            },
                            'audio_info': {
                                'duration': result['metadata']['duration'],
                                'sample_rate': result['metadata']['sample_rate']
                            },
                            'classification': {
                                'category_id': int(cat_id),
                                'category_name': category_info['name'],
                                'confidence': float(confidence),
                                'probabilities': {k: float(v) for k, v in result['classification']['probabilities'].items()}
                            },
                            'events': result['events']
                        }
                        
                        st.download_button(
                            label="📥 Download Results (JSON)",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"classification_{uploaded_file.name.split('.')[0]}.json",
                            mime="application/json",
                            use_container_width=True
                        )

with tab2:
    st.header("📊 Batch Processing")
    st.info("🚧 Batch processing feature coming soon! Upload multiple files for classification.")
    
    uploaded_files = st.file_uploader(
        "Upload multiple audio files",
        type=['wav', 'mp3', 'flac'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files selected")
        
        if st.button("🔍 Process All Files", type="primary"):
            st.warning("Batch processing will be implemented in the next version.")

with tab3:
    st.header("ℹ️ About")
    
    for cat_id, cat_info in CATEGORIES.items():
        st.write(f"**{cat_id}. {cat_info['name']}** - {cat_info['description']}")
    
    st.divider()
