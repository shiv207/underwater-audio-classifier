#!/usr/bin/env python3
"""Batch process audio files and generate single submission JSON"""
import os
import json
import argparse
import glob
import torch
import librosa
import soundfile as sf
from simple_model import create_model
from simple_data import process_audio


def batch_process(audio_dir, model_path='models/best_model_finetuned.pth', 
                  output_json='final_submission.json', compress_large=True):
    """Process all audio files in directory and generate submission JSON"""
    
    print("="*60)
    print("Batch Audio Processing for Grand Challenge UDA")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_to_id = checkpoint['class_to_id']
    num_classes = len(class_to_id)
    
    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    id_to_class = {v: k for k, v in class_to_id.items()}
    class_names = [id_to_class[i] for i in range(num_classes)]
    
    category_map = {
        'vessels': 1,
        'marine_animals': 2,
        'natural_sounds': 3,
        'other_anthropogenic': 4
    }
    
    # Find audio files
    audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
    
    if not audio_files:
        print(f"❌ No WAV files found in {audio_dir}")
        return
    
    print(f"✅ Found {len(audio_files)} audio files\n")
    
    # Initialize submission
    submission = {
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
    
    audio_id = 1
    annotation_id = 1
    
    # Process each file
    for idx, audio_path in enumerate(audio_files, 1):
        try:
            filename = os.path.basename(audio_path)
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            
            print(f"[{idx}/{len(audio_files)}] Processing {filename} ({file_size_mb:.1f}MB)...")
            
            # Compress if needed
            if compress_large and file_size_mb > 200:
                print(f"    ⚠️  Large file detected, compressing...")
                audio, sr = librosa.load(audio_path, sr=None)
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                temp_path = f"/tmp/temp_compressed_{idx}.wav"
                sf.write(temp_path, audio, 16000, subtype='PCM_16')
                audio_path = temp_path
                print(f"    ✅ Compressed to 16kHz")
            
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
            
            # Add to submission
            submission['audios'].append({
                "id": audio_id,
                "file_name": filename,
                "file_path": "to be mentioned by the participants",
                "duration": round(duration, 1)
            })
            
            submission['annotations'].append({
                "id": annotation_id,
                "audio_id": audio_id,
                "category_id": category_id,
                "start_time": 0.0,
                "end_time": round(duration, 1),
                "score": round(confidence, 4)
            })
            
            print(f"    ✅ {predicted_class} (confidence: {confidence*100:.1f}%)")
            
            audio_id += 1
            annotation_id += 1
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue
    
    # Save submission
    with open(output_json, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print("\n" + "="*60)
    print(f"✅ Processing Complete!")
    print(f"   Total files: {len(submission['audios'])}")
    print(f"   Total annotations: {len(submission['annotations'])}")
    print(f"   Output: {output_json}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch process audio files for Grand Challenge UDA')
    parser.add_argument('--audio_dir', required=True, help='Directory with audio files')
    parser.add_argument('--model', default='models/best_model_finetuned.pth', help='Model path')
    parser.add_argument('--output', default='final_submission.json', help='Output JSON file')
    parser.add_argument('--no-compress', action='store_true', help='Disable auto-compression')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_dir):
        print(f"❌ Directory not found: {args.audio_dir}")
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        exit(1)
    
    batch_process(args.audio_dir, args.model, args.output, not args.no_compress)
