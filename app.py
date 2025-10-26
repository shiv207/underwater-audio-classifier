#!/usr/bin/env python3
import os
import json
import glob
import torch
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import warnings
import zipfile
import io

warnings.filterwarnings("ignore")

from simple_model import create_model
from simple_data import process_audio

st.set_page_config(
    page_title="Underwater Sound Classifier",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Increase upload size limit
os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '500'

@st.cache_resource
def load_model(model_path):
    """Load model from checkpoint"""
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_to_id = checkpoint["class_to_id"]
    num_classes = len(class_to_id)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    id_to_class = {v: k for k, v in class_to_id.items()}
    class_names = [id_to_class[i] for i in range(num_classes)]

    return model, device, class_names

def compress_audio_if_needed(audio_path, max_size_mb=200, target_sr=16000):
    """Compress audio file if it exceeds size limit"""
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    
    if file_size_mb <= max_size_mb:
        return audio_path, False
    
    # Compress the audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Save compressed version
    compressed_path = audio_path.replace('.wav', '_compressed.wav')
    sf.write(compressed_path, audio, target_sr, subtype='PCM_16')
    
    compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
    
    return compressed_path, True

def plot_spectrogram(log_mel_spec):
    """Create spectrogram visualization"""
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(
        log_mel_spec,
        x_axis="time",
        y_axis="mel",
        sr=16000,
        fmax=8000,
        ax=ax,
        cmap="viridis",
    )
    ax.set_title("Audio Spectrogram", fontsize=12, pad=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig


def plot_probabilities(probs_dict):
    """Create probability bar chart"""
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0].replace("_", " ").title() for item in sorted_probs]
    probs = [item[1] * 100 for item in sorted_probs]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#1f77b4" if i == 0 else "#d3d3d3" for i in range(len(probs))]
    bars = ax.barh(classes, probs, color=colors)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_xlim(0, 100)

    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 2, i, f"{prob:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    return fig

def main():
    # Header
    st.title("🌊 Underwater Sound Classifier")
    st.markdown("Identify underwater sounds using AI")

    # Sidebar - Model Selection
    with st.sidebar:
        st.header("⚙️ Settings")

        # Check available models
        available_models = {}
        if os.path.exists("models/best_model_finetuned.pth"):
            available_models["Fine-tuned (98% acc)"] = "models/best_model_finetuned.pth"
        if os.path.exists("models/best_model_simple.pth"):
            available_models["Simple CNN (93% acc)"] = "models/best_model_simple.pth"

        if not available_models:
            st.error("❌ No trained models found!")
            st.info("Run `python train_minimal.py` first")
            st.stop()

        # Model selection
        selected_model_name = st.selectbox(
            "Model", options=list(available_models.keys()), index=0
        )
        model_path = available_models[selected_model_name]

        st.markdown("---")
        
        # Detection mode
        st.subheader("🎯 Detection Mode")
        detection_mode = st.radio(
            "Mode",
            ["Full Audio", "Event Detection"],
            help="Full Audio: Classify entire file\nEvent Detection: Find anomalies with timestamps"
        )
        
        if detection_mode == "Event Detection":
            st.slider("Window Size (s)", 5.0, 30.0, 10.0, 5.0, key="window_size")
            st.slider("Hop Size (s)", 1.0, 10.0, 5.0, 1.0, key="hop_size")
            st.slider("Threshold", 0.3, 0.9, 0.5, 0.1, key="threshold")

        st.markdown("---")

        # Classes info
        st.subheader("📋 Categories")
        st.markdown(
            """
        - 🐋 Marine Animals
        - 🚢 Vessels
        - 🌊 Natural Sounds
        - 🔧 Other Sounds
        """
        )

        st.markdown("---")
        st.caption("Grand Challenge UDA Compatible")
    
    # File input options
    st.subheader("📁 Select Audio Files")
    
    input_method = st.radio(
        "Input Method",
        ["📂 Local Directory (for large files)", "⬆️ Upload Files", "📚 Training Samples"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    uploaded_files = None
    
    if input_method == "📂 Local Directory (for large files)":
        st.info("💡 Use this for files larger than 200MB")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            local_dir = st.text_input(
                "Directory path with audio files",
                placeholder="/path/to/I4_Test_files",
                help="Enter the full path to directory containing WAV files"
            )
        with col2:
            st.write("")
            st.write("")
            load_local = st.button("📂 Load", use_container_width=True, type="primary")
        
        if load_local and local_dir:
            if os.path.exists(local_dir):
                audio_files = glob.glob(os.path.join(local_dir, '*.wav'))
                if audio_files:
                    uploaded_files = sorted(audio_files)
                    st.success(f"✅ Found {len(uploaded_files)} audio files")
                else:
                    st.error("❌ No WAV files found in directory")
            else:
                st.error("❌ Directory not found")
    
    elif input_method == "⬆️ Upload Files":
        st.warning("⚠️ Upload limited to 200MB per file. Use 'Local Directory' for larger files.")
        uploaded_files = st.file_uploader(
            "Choose WAV files (drag & drop multiple files)", 
            type=["wav", "mp3", "flac"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
    
    elif input_method == "📚 Training Samples":
        # Sample selection
        with st.container():
            sample_dir = "data/training"
            sample_path = None

            if os.path.exists(sample_dir):
                categories = [
                    d
                    for d in os.listdir(sample_dir)
                    if os.path.isdir(os.path.join(sample_dir, d)) and not d.startswith(".")
                ]

                if categories:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        selected_category = st.selectbox("Category", categories)
                    with col2:
                        category_path = os.path.join(sample_dir, selected_category)
                        audio_files = [
                            f
                            for f in os.listdir(category_path)
                            if f.endswith((".wav", ".mp3", ".flac"))
                        ]
                        if audio_files:
                            selected_file = st.selectbox("File", audio_files)
                            sample_path = os.path.join(category_path, selected_file)

                    if st.button("📥 Load Sample", use_container_width=True):
                        uploaded_files = [sample_path]
            else:
                st.info("No training data found")
    
    # Process uploaded files
    if uploaded_files:
        st.info(f"📦 {len(uploaded_files)} file(s) uploaded")
        
        # Process files with compression if needed
        audio_paths = []
        compressed_files = []
        
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                # Handle file path
                if isinstance(uploaded_file, str):
                    audio_path = uploaded_file
                else:
                    audio_bytes = uploaded_file.read()
                    audio_path = f"/tmp/{uploaded_file.name}"
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                
                # Check and compress if needed
                file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                if file_size_mb > 200:
                    st.warning(f"⚠️ {os.path.basename(audio_path)} ({file_size_mb:.1f}MB) - Compressing...")
                    compressed_path, was_compressed = compress_audio_if_needed(audio_path, max_size_mb=200)
                    audio_paths.append(compressed_path)
                    compressed_files.append(os.path.basename(audio_path))
                else:
                    audio_paths.append(audio_path)
        
        if compressed_files:
            st.success(f"✅ Compressed {len(compressed_files)} large file(s) to 16kHz")
        
        # Show file selector if multiple files
        if len(audio_paths) > 1:
            selected_file_idx = st.selectbox(
                "Preview file:",
                range(len(audio_paths)),
                format_func=lambda i: os.path.basename(audio_paths[i])
            )
            preview_path = audio_paths[selected_file_idx]
        else:
            preview_path = audio_paths[0]
        
        # Audio player for preview
        with open(preview_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            classify_button = st.button(
                f"🎯 Classify All ({len(audio_paths)})", type="primary", use_container_width=True
            )
        with col2:
            compare_button = (
                st.button("⚖️ Compare Models", use_container_width=True)
                if len(available_models) > 1
                else None
            )
        
        # Classify audio
        if classify_button:
            mode = detection_mode if 'detection_mode' in locals() else "Full Audio"
            
            if mode == "Full Audio":
                # Full audio classification for all files
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load model once
                model, device, class_names = load_model(model_path)
                
                category_map = {
                    'vessels': 1,
                    'marine_animals': 2,
                    'natural_sounds': 3,
                    'other_anthropogenic': 4
                }
                category_names = {1: "vessel", 2: "marine_animal", 3: "natural_sound", 4: "other_anthropogenic"}
                
                # Initialize single submission structure
                final_submission = {
                    "info": {
                        "description": "Grand Challenge UDA",
                        "version": "1.0",
                        "year": 2025
                    },
                    "audios": [],
                    "categories": [
                        {"id": 1, "name": "vessel"},
                        {"id": 2, "name": "marine_animal"},
                        {"id": 3, "name": "natural_sound"},
                        {"id": 4, "name": "other_anthropogenic"}
                    ],
                    "annotations": []
                }
                
                all_results = []
                audio_id = 1
                annotation_id = 1
                
                for idx, audio_path in enumerate(audio_paths):
                    status_text.text(f"🔄 Processing {idx+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")
                    progress_bar.progress((idx) / len(audio_paths))
                    
                    try:
                        filename = os.path.basename(audio_path)
                        
                        # Load audio and get duration
                        audio_data, sr = librosa.load(audio_path, sr=None)
                        duration = float(len(audio_data) / sr)
                        
                        # Process and predict
                        log_mel_spec = process_audio(audio_path)
                        spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = model(spec_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                        
                        pred_idx = probabilities.argmax().item()
                        confidence = probabilities[pred_idx].item()
                        predicted_class = class_names[pred_idx]
                        category_id = category_map.get(predicted_class, 1)
                        
                        # Add to final submission
                        final_submission['audios'].append({
                            "id": audio_id,
                            "file_name": filename,
                            "file_path": "to be mentioned by the participants",
                            "duration": round(duration, 1)
                        })
                        
                        final_submission['annotations'].append({
                            "id": annotation_id,
                            "audio_id": audio_id,
                            "category_id": category_id,
                            "start_time": 0.0,
                            "end_time": round(duration, 1),
                            "score": round(confidence, 4)
                        })
                        
                        all_results.append({
                            'filename': filename,
                            'class': category_names[category_id],
                            'confidence': confidence,
                            'duration': duration
                        })
                        
                        audio_id += 1
                        annotation_id += 1
                    
                    except Exception as e:
                        st.error(f"❌ Error processing {os.path.basename(audio_path)}: {str(e)}")
                
                progress_bar.progress(1.0)
                status_text.text("✅ All files processed!")
                
                # Display results
                st.markdown("---")
                st.success(f"### 🎯 Processed {len(all_results)} Files")
                
                # Results table
                for result in all_results:
                    with st.expander(f"📄 {result['filename']} - {result['class'].replace('_', ' ').title()} ({result['confidence']*100:.1f}%)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                        with col2:
                            st.metric("Duration", f"{result['duration']:.1f}s")
                
                # Download single consolidated JSON
                st.markdown("---")
                st.subheader("📥 Download Submission")
                
                final_json_str = json.dumps(final_submission, indent=2)
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="📄 Download Final Submission JSON",
                        data=final_json_str,
                        file_name="final_submission.json",
                        mime="application/json",
                        use_container_width=True,
                        type="primary"
                    )
                
                with col_dl2:
                    # Show preview
                    if st.button("👁️ Preview JSON", use_container_width=True):
                        st.json(final_submission)
                
                st.info(f"✅ Single JSON file with {len(final_submission['audios'])} audios and {len(final_submission['annotations'])} annotations")
            
            else:
                # Event detection mode for all files
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load model once
                model, device, class_names = load_model(model_path)
                
                window_size = st.session_state.get('window_size', 10.0)
                hop_size = st.session_state.get('hop_size', 5.0)
                threshold = st.session_state.get('threshold', 0.5)
                
                category_map = {
                    'vessels': 1,
                    'marine_animals': 2,
                    'natural_sounds': 3,
                    'other_anthropogenic': 4
                }
                
                # Initialize single submission structure
                final_submission = {
                    "info": {
                        "description": "Grand Challenge UDA",
                        "version": "1.0",
                        "year": 2025
                    },
                    "audios": [],
                    "categories": [
                        {"id": 1, "name": "vessel"},
                        {"id": 2, "name": "marine_animal"},
                        {"id": 3, "name": "natural_sound"},
                        {"id": 4, "name": "other_anthropogenic"}
                    ],
                    "annotations": []
                }
                
                all_results = []
                audio_id = 1
                annotation_id = 1
                
                for idx, audio_path in enumerate(audio_paths):
                    status_text.text(f"🔍 Detecting events {idx+1}/{len(audio_paths)}: {os.path.basename(audio_path)}")
                    progress_bar.progress((idx) / len(audio_paths))
                    
                    try:
                        filename = os.path.basename(audio_path)
                        
                        # Load audio
                        audio_data, sr = librosa.load(audio_path, sr=None)
                        duration = float(len(audio_data) / sr)
                        
                        # Add audio entry
                        final_submission['audios'].append({
                            "id": audio_id,
                            "file_name": filename,
                            "file_path": "to be mentioned by the participants",
                            "duration": round(duration, 1)
                        })
                        
                        # Detect events using sliding window
                        events_found = 0
                        window_samples = int(window_size * sr)
                        hop_samples = int(hop_size * sr)
                        
                        for start_sample in range(0, len(audio_data) - window_samples, hop_samples):
                            end_sample = start_sample + window_samples
                            window_audio = audio_data[start_sample:end_sample]
                            
                            # Save temp window
                            temp_path = f"/tmp/temp_window_{idx}.wav"
                            sf.write(temp_path, window_audio, sr)
                            
                            # Process window
                            log_mel_spec = process_audio(temp_path)
                            spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                outputs = model(spec_tensor)
                                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                            
                            pred_idx = probabilities.argmax().item()
                            confidence = probabilities[pred_idx].item()
                            
                            if confidence >= threshold:
                                predicted_class = class_names[pred_idx]
                                category_id = category_map.get(predicted_class, 1)
                                
                                start_time = start_sample / sr
                                end_time = end_sample / sr
                                
                                final_submission['annotations'].append({
                                    "id": annotation_id,
                                    "audio_id": audio_id,
                                    "category_id": category_id,
                                    "start_time": round(start_time, 1),
                                    "end_time": round(end_time, 1),
                                    "score": round(confidence, 4)
                                })
                                
                                annotation_id += 1
                                events_found += 1
                        
                        all_results.append({
                            'filename': filename,
                            'events': events_found,
                            'duration': duration
                        })
                        
                        audio_id += 1
                    
                    except Exception as e:
                        st.error(f"❌ Error processing {os.path.basename(audio_path)}: {str(e)}")
                
                progress_bar.progress(1.0)
                status_text.text("✅ All files processed!")
                
                # Display results
                st.markdown("---")
                st.success(f"### 🔍 Processed {len(all_results)} Files")
                
                category_names = {1: "vessel", 2: "marine_animal", 3: "natural_sound", 4: "other_anthropogenic"}
                
                # Results for each file
                for result in all_results:
                    with st.expander(f"📄 {result['filename']} - {result['events']} events detected"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Events", result['events'])
                        with col2:
                            st.metric("Duration", f"{result['duration']:.1f}s")
                        
                        # Show events for this file
                        file_annotations = [a for a in final_submission['annotations'] 
                                          if final_submission['audios'][a['audio_id']-1]['file_name'] == result['filename']]
                        for event in file_annotations:
                            cat_name = category_names[event['category_id']]
                            st.write(f"**[{event['start_time']:.1f}s - {event['end_time']:.1f}s]** {cat_name.replace('_', ' ').title()} (score: {event['score']:.2f})")
                
                # Download single consolidated JSON
                st.markdown("---")
                st.subheader("📥 Download Submission")
                
                final_json_str = json.dumps(final_submission, indent=2)
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label="📄 Download Final Submission JSON",
                        data=final_json_str,
                        file_name="final_submission.json",
                        mime="application/json",
                        use_container_width=True,
                        type="primary"
                    )
                
                with col_dl2:
                    # Show preview
                    if st.button("👁️ Preview JSON", use_container_width=True):
                        st.json(final_submission)
                
                total_events = sum(r['events'] for r in all_results)
                st.info(f"✅ Single JSON file with {len(final_submission['audios'])} audios and {total_events} event annotations")
        
        # Compare models (only for preview file)
        if compare_button:
            with st.spinner("⚖️ Comparing models..."):
                try:
                    log_mel_spec = process_audio(preview_path)
                    spec_tensor = (
                        torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0)
                    )

                    st.markdown("---")
                    st.subheader(f"⚖️ Model Comparison - {os.path.basename(preview_path)}")

                    results = {}
                    for model_name, model_path_compare in available_models.items():
                        model, device, class_names = load_model(model_path_compare)

                        with torch.no_grad():
                            outputs = model(spec_tensor.to(device))
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)[
                                0
                            ]

                        pred_idx = probabilities.argmax().item()
                        confidence = probabilities[pred_idx].item()
                        predicted_class = class_names[pred_idx]

                        results[model_name] = {
                            "prediction": predicted_class,
                            "confidence": confidence,
                        }

                    # Display comparison
                    cols = st.columns(len(results))
                    for idx, (model_name, result) in enumerate(results.items()):
                        with cols[idx]:
                            st.markdown(f"**{model_name}**")
                            st.info(
                                f"**{result['prediction'].replace('_', ' ').title()}**"
                            )
                            st.metric("Confidence", f"{result['confidence']*100:.1f}%")

                    # Spectrogram
                    with st.expander("🎵 View Spectrogram"):
                        st.pyplot(plot_spectrogram(log_mel_spec))

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
