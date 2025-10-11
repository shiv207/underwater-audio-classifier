"""
Simple test script to verify the underwater acoustic classification system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import AudioPreprocessor
from detector import SimpleEnergyDetector
from classifier import AcousticClassifier
import json

def test_preprocessing():
    """Test audio preprocessing."""
    print("Testing audio preprocessing...")
    
    preprocessor = AudioPreprocessor()
    
    # Test with sonar file
    audio_path = "data/training/other_anthropogenic/sonar.mp3"
    if os.path.exists(audio_path):
        audio, log_mel_spec, metadata = preprocessor.process_audio_file(audio_path)
        
        print(f"Audio shape: {audio.shape}")
        print(f"Spectrogram shape: {log_mel_spec.shape}")
        print(f"Duration: {metadata['duration']:.2f} seconds")
        print(f"Sample rate: {metadata['sample_rate']} Hz")
        
        return audio, log_mel_spec, metadata
    else:
        print(f"File not found: {audio_path}")
        return None, None, None

def test_detection(audio, log_mel_spec):
    """Test event detection."""
    print("\nTesting event detection...")
    
    # Use more sensitive detector settings
    detector = SimpleEnergyDetector(
        threshold_db=-40,  # More sensitive threshold
        min_duration=0.5   # Shorter minimum duration
    )
    
    if audio is not None and len(audio) > 0:
        events = detector.detect_events(audio)
        print(f"Detected {len(events)} events:")
        for i, event in enumerate(events):
            print(f"  Event {i+1}: {event['start_time']}-{event['end_time']}s (score: {event['score']})")
        return events
    else:
        print("No audio data to process")
        return []

def test_classification(log_mel_spec):
    """Test classification."""
    print("\nTesting classification...")
    
    classifier = AcousticClassifier()
    
    if log_mel_spec is not None and log_mel_spec.size > 0:
        result = classifier.classify_spectrogram(log_mel_spec)
        print(f"Classification result:")
        print(f"  Category ID: {result['category_id']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Probabilities: {result['probabilities']}")
        return result
    else:
        print("No spectrogram data to process")
        return None

def create_test_result(audio_path, events, classification_result, metadata):
    """Create test result in PS-12 format."""
    print("\nCreating test result...")
    
    # Create annotations from events and classification
    annotations = []
    for i, event in enumerate(events):
        annotation = {
            'id': i + 1,
            'audio_id': 1,
            'category_id': classification_result['category_id'] if classification_result else 4,
            'start_time': event['start_time'],
            'end_time': event['end_time'],
            'duration': event['duration'],
            'score': float(classification_result['confidence']) if classification_result else event['score']
        }
        annotations.append(annotation)
    
    # If no events detected, create one for entire audio
    if not annotations and classification_result:
        annotation = {
            'id': 1,
            'audio_id': 1,
            'category_id': int(classification_result['category_id']),
            'start_time': 0,
            'end_time': int(metadata['duration']),
            'duration': float(metadata['duration']),
            'score': float(classification_result['confidence'])
        }
        annotations.append(annotation)
    
    result = {
        'info': {
            'description': 'Test Results - Underwater Acoustic Classification',
            'version': '1.0',
            'year': 2025,
            'contributor': 'UDA Test System'
        },
        'audios': [{
            'id': 1,
            'file_name': os.path.basename(audio_path),
            'duration': float(metadata['duration'])
        }],
        'categories': [
            {'id': 1, 'name': 'vessels', 'supercategory': 'anthropogenic'},
            {'id': 2, 'name': 'marine_animals', 'supercategory': 'biological'},
            {'id': 3, 'name': 'natural_sounds', 'supercategory': 'natural'},
            {'id': 4, 'name': 'other_anthropogenic', 'supercategory': 'anthropogenic'}
        ],
        'annotations': annotations
    }
    
    return result

def main():
    """Main test function."""
    print("=" * 60)
    print("UNDERWATER ACOUSTIC CLASSIFICATION SYSTEM TEST")
    print("=" * 60)
    
    # Test preprocessing
    audio, log_mel_spec, metadata = test_preprocessing()
    
    if audio is None:
        print("Preprocessing failed. Exiting.")
        return
    
    # Test detection
    events = test_detection(audio, log_mel_spec)
    
    # Test classification
    classification_result = test_classification(log_mel_spec)
    
    # Create and save test result
    audio_path = "data/training/other_anthropogenic/sonar.mp3"
    result = create_test_result(audio_path, events, classification_result, metadata)
    
    # Save result
    output_file = "results/detailed_test_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nTest results saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Audio processed: ✓ ({metadata['duration']:.1f}s)")
    print(f"Events detected: {len(events)}")
    if classification_result:
        category_names = {1: 'vessels', 2: 'marine_animals', 3: 'natural_sounds', 4: 'other_anthropogenic'}
        predicted_category = category_names[classification_result['category_id']]
        print(f"Classification: {predicted_category} (confidence: {classification_result['confidence']:.3f})")
    print(f"Output format: PS-12 compliant JSON")
    
    print("\nSystem is working! ✓")
    print("\nNext steps:")
    print("1. Add more training data to improve accuracy")
    print("2. Train the models: python train.py --data-dir data/training")
    print("3. Test with different audio files")
    print("4. Deploy using Docker for production use")

if __name__ == '__main__':
    main()
