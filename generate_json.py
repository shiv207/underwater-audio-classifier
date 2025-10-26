#!/usr/bin/env python3
"""Generate Grand Challenge UDA JSON format predictions"""
import os
import json
import argparse
import torch
import librosa
from simple_model import create_model
from simple_data import process_audio


def predict_and_generate_json(audio_path, model_path='models/best_model_finetuned.pth', output_path='result.json'):
    """Predict audio classification and generate Grand Challenge UDA JSON"""
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
    
    audio_data, sr = librosa.load(audio_path, sr=None)
    duration = float(len(audio_data) / sr)
    filename = os.path.basename(audio_path)
    
    log_mel_spec = process_audio(audio_path)
    spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(spec_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    pred_idx = probabilities.argmax().item()
    confidence = probabilities[pred_idx].item()
    predicted_class = class_names[pred_idx]
    
    category_map = {
        'vessels': 1,
        'marine_animals': 2,
        'natural_sounds': 3,
        'other_anthropogenic': 4
    }
    category_id = category_map.get(predicted_class, 1)
    
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
                "id": 1,
                "audio_id": 1,
                "category_id": category_id,
                "start_time": 0.0,
                "end_time": round(duration, 1),
                "score": round(confidence, 4)
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Prediction: {predicted_class}")
    print(f"✓ Confidence: {confidence*100:.1f}%")
    print(f"✓ Duration: {duration:.1f}s")
    print(f"✓ JSON saved to: {output_path}")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Grand Challenge UDA JSON predictions')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--model', default='models/best_model_finetuned.pth', help='Path to model')
    parser.add_argument('--output', default='result.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"❌ Error: Audio file not found: {args.audio}")
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        exit(1)
    
    predict_and_generate_json(args.audio, args.model, args.output)
