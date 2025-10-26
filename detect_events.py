#!/usr/bin/env python3
"""Event Detection for Grand Challenge UDA"""
import os
import json
import argparse
import torch
import librosa
import numpy as np
from simple_model import create_model
from simple_data import process_audio


def sliding_window_detection(audio_path, model, device, class_names, 
                             window_size=10.0, hop_size=5.0, threshold=0.5):
    """Detect events using sliding window approach"""
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    events = []
    event_id = 1
    
    for start_sample in range(0, len(audio) - window_samples + 1, hop_samples):
        end_sample = start_sample + window_samples
        audio_window = audio[start_sample:end_sample]
        
        temp_path = '/tmp/temp_window.wav'
        librosa.output.write_wav(temp_path, audio_window, sr)
        
        try:
            log_mel_spec = process_audio(temp_path)
            spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(spec_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            pred_idx = probabilities.argmax().item()
            confidence = probabilities[pred_idx].item()
            predicted_class = class_names[pred_idx]
            
            if confidence >= threshold:
                start_time = start_sample / sr
                end_time = end_sample / sr
                
                category_map = {
                    'vessels': 1,
                    'marine_animals': 2,
                    'natural_sounds': 3,
                    'other_anthropogenic': 4
                }
                category_id = category_map.get(predicted_class, 1)
                
                events.append({
                    'id': event_id,
                    'category_id': category_id,
                    'start_time': round(start_time, 1),
                    'end_time': round(end_time, 1),
                    'score': round(confidence, 4)
                })
                event_id += 1
        
        except Exception as e:
            print(f"Warning: Error processing window at {start_sample/sr:.1f}s: {e}")
            continue
    
    merged_events = merge_events(events)
    return merged_events, duration


def merge_events(events):
    """Merge overlapping events of the same category"""
    if not events:
        return []
    
    events = sorted(events, key=lambda x: x['start_time'])
    
    merged = []
    current = events[0].copy()
    
    for event in events[1:]:
        if (event['category_id'] == current['category_id'] and 
            event['start_time'] <= current['end_time'] + 1.0):
            current['end_time'] = max(current['end_time'], event['end_time'])
            current['score'] = max(current['score'], event['score'])
        else:
            merged.append(current)
            current = event.copy()
    
    merged.append(current)
    
    for i, event in enumerate(merged, 1):
        event['id'] = i
    
    return merged


def generate_uda_json(audio_path, model_path='models/best_model_finetuned.pth', 
                      output_path='result.json', window_size=10.0, hop_size=5.0,
                      threshold=0.5):
    """Generate Grand Challenge UDA JSON with event detection"""
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
    
    print(f"🔍 Detecting events in {os.path.basename(audio_path)}...")
    print(f"   Window: {window_size}s, Hop: {hop_size}s, Threshold: {threshold}")
    
    events, duration = sliding_window_detection(
        audio_path, model, device, class_names,
        window_size, hop_size, threshold
    )
    
    filename = os.path.basename(audio_path)
    
    result = {
        "info": {
            "description": "Grand Challenge UDA",
            "version": "1.0",
            "year": 2025
        },
        "audios": [
            {
                "id": 1,
                "file_name": filename,
                "duration": round(duration, 1)
            }
        ],
        "categories": [
            {"id": 1, "name": "vessel"},
            {"id": 2, "name": "marine_animal"},
            {"id": 3, "name": "natural_sound"},
            {"id": 4, "name": "other_anthropogenic"}
        ],
        "annotations": [
            {
                "id": event['id'],
                "audio_id": 1,
                "category_id": event['category_id'],
                "start_time": event['start_time'],
                "end_time": event['end_time'],
                "score": event['score']
            }
            for event in events
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Detected {len(events)} events")
    print(f"✓ Audio duration: {duration:.1f}s")
    print(f"✓ JSON saved to: {output_path}")
    
    category_names = {1: "vessel", 2: "marine_animal", 3: "natural_sound", 4: "other_anthropogenic"}
    for event in events:
        cat_name = category_names[event['category_id']]
        print(f"   [{event['start_time']:.1f}s - {event['end_time']:.1f}s] {cat_name} (score: {event['score']:.2f})")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Event Detection for Grand Challenge UDA')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--model', default='models/best_model_finetuned.pth', help='Path to model')
    parser.add_argument('--output', default='result.json', help='Output JSON file')
    parser.add_argument('--window', type=float, default=10.0, help='Window size in seconds')
    parser.add_argument('--hop', type=float, default=5.0, help='Hop size in seconds')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"❌ Error: Audio file not found: {args.audio}")
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        exit(1)
    
    generate_uda_json(args.audio, args.model, args.output, 
                     args.window, args.hop, args.threshold)
