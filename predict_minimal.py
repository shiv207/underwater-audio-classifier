#!/usr/bin/env python3
import torch
import argparse
from simple_model import create_model
from simple_data import process_audio


def predict(audio_path, model_path='models/best_model_finetuned.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(model_path, map_location=device)
    class_to_id = checkpoint['class_to_id']
    id_to_class = {v: k for k, v in class_to_id.items()}
    
    model = create_model(num_classes=len(class_to_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    spec = process_audio(audio_path)
    spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'forward') and 'return_confidence' in model.forward.__code__.co_varnames:
            output, model_confidence = model(spec_tensor, return_confidence=True)
            model_conf = model_confidence[0].item()
        else:
            output = model(spec_tensor)
            model_conf = None
        
        probs = torch.softmax(output, dim=1)[0]
    
    pred_id = probs.argmax().item()
    pred_class = id_to_class[pred_id]
    confidence = probs[pred_id].item()
    
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    if model_conf is not None:
        print(f"Model Confidence: {model_conf*100:.2f}%")
    print("All probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {id_to_class[i]}: {prob.item()*100:.2f}%")
    
    return pred_class, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True, help='Audio file path')
    parser.add_argument('--model', default='models/best_model_finetuned.pth', help='Model path')
    args = parser.parse_args()
    
    predict(args.audio, args.model)
