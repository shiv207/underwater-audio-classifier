#!/usr/bin/env python3
"""Generate single Grand Challenge UDA JSON submission for all test files"""
import os
import json
import argparse
import glob
import torch
import librosa
from simple_model import create_model
from simple_data import process_audio


def predict_all_and_generate_json(audio_dir, model_path='models/best_model_finetuned.pth', output_path='final_submission.json'):
    """Process all audio files and generate single submission JSON"""
    
    # Load model once
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
    
    # Initialize submission structure
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
    
    # Find all audio files
    audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
    
    if not audio_files:
        print(f"❌ No .wav files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print("Processing...")
    
    audio_id = 1
    annotation_id = 1
    
    # Process each audio file
    for audio_path in audio_files:
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
            
            # Add audio entry
            submission['audios'].append({
                "id": audio_id,
                "file_name": filename,
                "file_path": "to be mentioned by the participants",
                "duration": round(duration, 1)
            })
            
            # Add annotation entry
            submission['annotations'].append({
                "id": annotation_id,
                "audio_id": audio_id,
                "category_id": category_id,
                "start_time": 0.0,
                "end_time": round(duration, 1),
                "score": round(confidence, 4)
            })
            
            print(f"✓ [{audio_id}/{len(audio_files)}] {filename} - {predicted_class} ({confidence*100:.1f}%)")
            
            audio_id += 1
            annotation_id += 1
            
        except Exception as e:
            print(f"⚠ Error processing {audio_path}: {e}")
            continue
    
    # Save final submission
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\n✓ Processed {len(submission['audios'])} audio files")
    print(f"✓ Total annotations: {len(submission['annotations'])}")
    print(f"✓ Final submission saved to: {output_path}")
    
    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate single Grand Challenge UDA JSON submission')
    parser.add_argument('--audio_dir', required=True, help='Directory containing test audio files')
    parser.add_argument('--model', default='models/best_model_finetuned.pth', help='Path to model')
    parser.add_argument('--output', default='final_submission.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_dir):
        print(f"❌ Error: Audio directory not found: {args.audio_dir}")
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        exit(1)
    
    predict_all_and_generate_json(args.audio_dir, args.model, args.output)
