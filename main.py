
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import AudioPreprocessor
from detector import EventDetector, SimpleEnergyDetector
from classifier import AcousticClassifier
from evaluate import UnderwaterAcousticEvaluator, validate_json_format

class UnderwaterAcousticSystem:
    
    def __init__(self, 
                 detector_model_path: str = None,
                 classifier_model_path: str = None,
                 use_energy_detector: bool = True):
        """
        Initialize the complete system.
        
        Args:
            detector_model_path: Path to trained detector model
            classifier_model_path: Path to trained classifier model
            use_energy_detector: Use simple energy detector as fallback
        """
        # Initialize components
        self.preprocessor = AudioPreprocessor()
        
        # Initialize detector
        if detector_model_path and os.path.exists(detector_model_path):
            self.detector = EventDetector(detector_model_path)
        else:
            print("Using simple energy-based detector")
            self.detector = SimpleEnergyDetector()
        
        # Initialize classifier
        if classifier_model_path and os.path.exists(classifier_model_path):
            self.classifier = AcousticClassifier(classifier_model_path)
        else:
            print("Using untrained classifier")
            self.classifier = AcousticClassifier()
        
        # Category definitions
        self.categories = [
            {"id": 1, "name": "vessels", "supercategory": "anthropogenic"},
            {"id": 2, "name": "marine_animals", "supercategory": "biological"},
            {"id": 3, "name": "natural_sounds", "supercategory": "natural"},
            {"id": 4, "name": "other_anthropogenic", "supercategory": "anthropogenic"}
        ]
    
    def process_audio_file(self, audio_path: str, audio_id: int = 1) -> Dict:
        """
        Process a single audio file and return results.
        
        Args:
            audio_path: Path to audio file
            audio_id: Unique audio ID
            
        Returns:
            Processing results dictionary
        """
        print(f"Processing: {audio_path}")
        
        # Preprocess audio
        audio, log_mel_spec, metadata = self.preprocessor.process_audio_file(audio_path)
        
        if len(audio) == 0:
            print(f"Failed to process {audio_path}")
            return {
                'audio_info': {
                    'id': audio_id,
                    'file_name': os.path.basename(audio_path),
                    'duration': 0.0
                },
                'events': []
            }
        
        # Detect events
        if hasattr(self.detector, 'detect_events'):
            if isinstance(self.detector, SimpleEnergyDetector):
                events = self.detector.detect_events(audio)
            else:
                events = self.detector.detect_events(log_mel_spec)
        else:
            # Fallback: treat entire audio as single event
            events = [{
                'start_time': 0,
                'end_time': int(metadata['duration']),
                'duration': metadata['duration'],
                'score': 0.8
            }]
        
        # Classify each detected event
        annotations = []
        annotation_id = 1
        
        for event in events:
            # Calculate frame indices for event
            start_frame = int(event['start_time'] * self.preprocessor.target_sr / self.preprocessor.hop_length)
            end_frame = int(event['end_time'] * self.preprocessor.target_sr / self.preprocessor.hop_length)
            
            # Ensure frames are within bounds
            start_frame = max(0, min(start_frame, log_mel_spec.shape[1] - 1))
            end_frame = max(start_frame + 1, min(end_frame, log_mel_spec.shape[1]))
            
            # Extract event spectrogram
            if log_mel_spec.size > 0:
                event_spec = log_mel_spec[:, start_frame:end_frame]
                classification_result = self.classifier.classify_spectrogram(event_spec)
            else:
                # Default classification if no spectrogram
                classification_result = {
                    'category_id': 3,  # natural_sounds as default
                    'confidence': 0.5
                }
            
            # Create annotation
            annotation = {
                'id': annotation_id,
                'audio_id': audio_id,
                'category_id': classification_result['category_id'],
                'start_time': event['start_time'],
                'end_time': event['end_time'],
                'duration': event['duration'],
                'score': float(classification_result['confidence'])
            }
            
            annotations.append(annotation)
            annotation_id += 1
        
        return {
            'audio_info': {
                'id': audio_id,
                'file_name': os.path.basename(audio_path),
                'duration': metadata['duration']
            },
            'events': annotations
        }
    
    def process_directory(self, input_dir: str) -> Dict:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            
        Returns:
            Complete results in PS-12 format
        """
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        # Find all audio files
        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).glob(f'*{ext}'))
            audio_files.extend(Path(input_dir).glob(f'**/*{ext}'))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return self._create_empty_result()
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process each file
        all_audios = []
        all_annotations = []
        
        for i, audio_file in enumerate(audio_files, 1):
            result = self.process_audio_file(str(audio_file), i)
            all_audios.append(result['audio_info'])
            all_annotations.extend(result['events'])
        
        # Create final result
        result = {
            'info': {
                'description': 'Underwater Acoustic Classification Results',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'UDA Classification System',
                'date_created': datetime.now().isoformat()
            },
            'audios': all_audios,
            'categories': self.categories,
            'annotations': all_annotations
        }
        
        return result
    
    def process_single_file(self, input_file: str) -> Dict:
        """
        Process a single audio file.
        
        Args:
            input_file: Path to audio file
            
        Returns:
            Results in PS-12 format
        """
        result = self.process_audio_file(input_file, 1)
        
        return {
            'info': {
                'description': 'Underwater Acoustic Classification Results',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'UDA Classification System',
                'date_created': datetime.now().isoformat()
            },
            'audios': [result['audio_info']],
            'categories': self.categories,
            'annotations': result['events']
        }
    
    def _create_empty_result(self) -> Dict:
        """Create empty result structure."""
        return {
            'info': {
                'description': 'Underwater Acoustic Classification Results',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'UDA Classification System',
                'date_created': datetime.now().isoformat()
            },
            'audios': [],
            'categories': self.categories,
            'annotations': []
        }

def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description='Underwater Acoustic Classification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/audio.wav --output results.json
  python main.py --input data/ --output results.json --detector-model models/detector.pth
  python main.py --evaluate --ground-truth gt.json --predictions pred.json
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, help='Input audio file or directory')
    parser.add_argument('--output', type=str, default='results.json', 
                       help='Output JSON file (default: results.json)')
    
    # Model arguments
    parser.add_argument('--detector-model', type=str, 
                       help='Path to trained detector model')
    parser.add_argument('--classifier-model', type=str,
                       help='Path to trained classifier model')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation mode')
    parser.add_argument('--ground-truth', type=str,
                       help='Ground truth JSON file for evaluation')
    parser.add_argument('--predictions', type=str,
                       help='Predictions JSON file for evaluation')
    
    # Other arguments
    parser.add_argument('--validate', action='store_true',
                       help='Validate output JSON format')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Evaluation mode
    if args.evaluate:
        if not args.ground_truth or not args.predictions:
            print("Error: --ground-truth and --predictions required for evaluation")
            return 1
        
        evaluator = UnderwaterAcousticEvaluator()
        results = evaluator.evaluate_system(args.ground_truth, args.predictions)
        evaluator.print_evaluation_report(results)
        return 0
    
    # Inference mode
    if not args.input:
        print("Error: --input required for inference")
        return 1
    
    # Initialize system
    system = UnderwaterAcousticSystem(
        detector_model_path=args.detector_model,
        classifier_model_path=args.classifier_model
    )
    
    # Process input
    if os.path.isfile(args.input):
        results = system.process_single_file(args.input)
    elif os.path.isdir(args.input):
        results = system.process_directory(args.input)
    else:
        print(f"Error: Input path {args.input} does not exist")
        return 1
    
    # Save results
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Validate output format if requested
    if args.validate:
        if validate_json_format(args.output):
            print("Output format validation passed")
        else:
            print("Output format validation failed")
            return 1
    
    # Print summary
    num_files = len(results['audios'])
    num_events = len(results['annotations'])
    print(f"Processed {num_files} audio files, detected {num_events} events")
    
    if args.verbose:
        print("\nDetected events:")
        for ann in results['annotations']:
            category_name = next(
                cat['name'] for cat in results['categories'] 
                if cat['id'] == ann['category_id']
            )
            print(f"  Audio {ann['audio_id']}: {category_name} "
                  f"({ann['start_time']}-{ann['end_time']}s, "
                  f"score={ann['score']:.3f})")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
