"""
Evaluation module for underwater acoustic classification system.
Computes Identification Error Rate (IER) and other metrics.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pyannote.metrics.identification import IdentificationErrorRate
from pyannote.core import Annotation, Segment
import sklearn.metrics as metrics

class UnderwaterAcousticEvaluator:
    """
    Evaluator for underwater acoustic classification system.
    """
    
    def __init__(self, 
                 false_alarm_weight: float = 0.25,
                 miss_weight: float = 1.0,
                 confusion_weight: float = 0.75):
        """
        Initialize evaluator with IER weights.
        
        Args:
            false_alarm_weight: Weight for false alarm errors
            miss_weight: Weight for miss errors  
            confusion_weight: Weight for class confusion errors
        """
        self.false_alarm_weight = false_alarm_weight
        self.miss_weight = miss_weight
        self.confusion_weight = confusion_weight
        
        # Initialize IER metric
        self.ier_metric = IdentificationErrorRate()
        
        # Class mapping
        self.class_names = {
            1: 'vessels',
            2: 'marine_animals',
            3: 'natural_sounds',
            4: 'other_anthropogenic'
        }
    
    def load_annotations(self, annotation_file: str) -> Dict:
        """
        Load ground truth annotations from JSON file.
        
        Args:
            annotation_file: Path to annotation JSON file
            
        Returns:
            Loaded annotations dictionary
        """
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations
    
    def load_predictions(self, prediction_file: str) -> Dict:
        """
        Load predictions from JSON file.
        
        Args:
            prediction_file: Path to prediction JSON file
            
        Returns:
            Loaded predictions dictionary
        """
        with open(prediction_file, 'r') as f:
            predictions = json.load(f)
        return predictions
    
    def convert_to_pyannote_annotation(self, 
                                     annotations: List[Dict], 
                                     audio_duration: float) -> Annotation:
        """
        Convert annotations to pyannote Annotation format.
        
        Args:
            annotations: List of annotation dictionaries
            audio_duration: Total audio duration in seconds
            
        Returns:
            pyannote Annotation object
        """
        annotation = Annotation()
        
        for ann in annotations:
            start_time = ann['start_time']
            end_time = ann['end_time']
            category_id = ann['category_id']
            
            # Create segment
            segment = Segment(start_time, end_time)
            
            # Add to annotation with class label
            class_name = self.class_names.get(category_id, 'unknown')
            annotation[segment] = class_name
        
        return annotation
    
    def compute_ier(self, 
                   ground_truth: Dict, 
                   predictions: Dict) -> Dict:
        """
        Compute Identification Error Rate (IER).
        
        Args:
            ground_truth: Ground truth annotations
            predictions: Model predictions
            
        Returns:
            IER results dictionary
        """
        results = {}
        
        # Process each audio file
        for audio_info in ground_truth.get('audios', []):
            audio_id = audio_info['id']
            audio_duration = audio_info.get('duration', 60.0)  # Default duration
            
            # Get ground truth annotations for this audio
            gt_annotations = [
                ann for ann in ground_truth.get('annotations', [])
                if ann['audio_id'] == audio_id
            ]
            
            # Get predictions for this audio
            pred_annotations = [
                ann for ann in predictions.get('annotations', [])
                if ann['audio_id'] == audio_id
            ]
            
            # Convert to pyannote format
            gt_annotation = self.convert_to_pyannote_annotation(gt_annotations, audio_duration)
            pred_annotation = self.convert_to_pyannote_annotation(pred_annotations, audio_duration)
            
            # Compute IER for this file
            ier_value = self.ier_metric(gt_annotation, pred_annotation)
            results[audio_id] = ier_value
        
        # Compute overall IER
        if results:
            overall_ier = np.mean(list(results.values()))
        else:
            overall_ier = 1.0  # Maximum error if no results
        
        return {
            'overall_ier': overall_ier,
            'per_file_ier': results,
            'num_files': len(results)
        }
    
    def compute_classification_metrics(self, 
                                     ground_truth: Dict, 
                                     predictions: Dict) -> Dict:
        """
        Compute classification metrics (precision, recall, F1).
        
        Args:
            ground_truth: Ground truth annotations
            predictions: Model predictions
            
        Returns:
            Classification metrics dictionary
        """
        # Collect all ground truth and predicted labels
        gt_labels = []
        pred_labels = []
        
        # Create mapping from audio_id + time to annotations
        gt_events = {}
        for ann in ground_truth.get('annotations', []):
            key = (ann['audio_id'], ann['start_time'], ann['end_time'])
            gt_events[key] = ann['category_id']
        
        pred_events = {}
        for ann in predictions.get('annotations', []):
            key = (ann['audio_id'], ann['start_time'], ann['end_time'])
            pred_events[key] = ann['category_id']
        
        # Match events based on temporal overlap
        matched_pairs = self._match_events(
            ground_truth.get('annotations', []),
            predictions.get('annotations', [])
        )
        
        for gt_ann, pred_ann in matched_pairs:
            if gt_ann and pred_ann:
                gt_labels.append(gt_ann['category_id'])
                pred_labels.append(pred_ann['category_id'])
        
        if not gt_labels:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'confusion_matrix': [],
                'classification_report': {}
            }
        
        # Compute metrics
        precision = metrics.precision_score(gt_labels, pred_labels, average='weighted', zero_division=0)
        recall = metrics.recall_score(gt_labels, pred_labels, average='weighted', zero_division=0)
        f1 = metrics.f1_score(gt_labels, pred_labels, average='weighted', zero_division=0)
        accuracy = metrics.accuracy_score(gt_labels, pred_labels)
        
        # Confusion matrix
        cm = metrics.confusion_matrix(gt_labels, pred_labels, labels=[1, 2, 3, 4])
        
        # Classification report
        class_names_list = [self.class_names[i] for i in [1, 2, 3, 4]]
        report = metrics.classification_report(
            gt_labels, pred_labels, 
            labels=[1, 2, 3, 4],
            target_names=class_names_list,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def _match_events(self, 
                     gt_annotations: List[Dict], 
                     pred_annotations: List[Dict],
                     overlap_threshold: float = 0.5) -> List[Tuple]:
        """
        Match ground truth and predicted events based on temporal overlap.
        
        Args:
            gt_annotations: Ground truth annotations
            pred_annotations: Predicted annotations
            overlap_threshold: Minimum overlap ratio for matching
            
        Returns:
            List of matched (gt_ann, pred_ann) pairs
        """
        matched_pairs = []
        used_predictions = set()
        
        for gt_ann in gt_annotations:
            best_match = None
            best_overlap = 0.0
            best_idx = -1
            
            gt_start = gt_ann['start_time']
            gt_end = gt_ann['end_time']
            gt_duration = gt_end - gt_start
            
            for i, pred_ann in enumerate(pred_annotations):
                if i in used_predictions:
                    continue
                
                if pred_ann['audio_id'] != gt_ann['audio_id']:
                    continue
                
                pred_start = pred_ann['start_time']
                pred_end = pred_ann['end_time']
                
                # Calculate overlap
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    overlap_ratio = overlap_duration / max(gt_duration, 0.1)
                    
                    if overlap_ratio > best_overlap and overlap_ratio >= overlap_threshold:
                        best_match = pred_ann
                        best_overlap = overlap_ratio
                        best_idx = i
            
            if best_match:
                matched_pairs.append((gt_ann, best_match))
                used_predictions.add(best_idx)
            else:
                # No match found (miss)
                matched_pairs.append((gt_ann, None))
        
        # Add unmatched predictions (false alarms)
        for i, pred_ann in enumerate(pred_annotations):
            if i not in used_predictions:
                matched_pairs.append((None, pred_ann))
        
        return matched_pairs
    
    def compute_detection_metrics(self, 
                                ground_truth: Dict, 
                                predictions: Dict) -> Dict:
        """
        Compute detection metrics (precision, recall for event detection).
        
        Args:
            ground_truth: Ground truth annotations
            predictions: Model predictions
            
        Returns:
            Detection metrics dictionary
        """
        matched_pairs = self._match_events(
            ground_truth.get('annotations', []),
            predictions.get('annotations', [])
        )
        
        true_positives = sum(1 for gt, pred in matched_pairs if gt and pred)
        false_positives = sum(1 for gt, pred in matched_pairs if not gt and pred)
        false_negatives = sum(1 for gt, pred in matched_pairs if gt and not pred)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'detection_precision': precision,
            'detection_recall': recall,
            'detection_f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate_system(self, 
                       ground_truth_file: str, 
                       predictions_file: str) -> Dict:
        """
        Complete system evaluation.
        
        Args:
            ground_truth_file: Path to ground truth JSON
            predictions_file: Path to predictions JSON
            
        Returns:
            Complete evaluation results
        """
        # Load data
        ground_truth = self.load_annotations(ground_truth_file)
        predictions = self.load_predictions(predictions_file)
        
        # Compute all metrics
        ier_results = self.compute_ier(ground_truth, predictions)
        classification_results = self.compute_classification_metrics(ground_truth, predictions)
        detection_results = self.compute_detection_metrics(ground_truth, predictions)
        
        # Combine results
        results = {
            'ier': ier_results,
            'classification': classification_results,
            'detection': detection_results,
            'summary': {
                'overall_ier': ier_results['overall_ier'],
                'classification_f1': classification_results['f1_score'],
                'detection_f1': detection_results['detection_f1'],
                'accuracy': classification_results['accuracy']
            }
        }
        
        return results
    
    def print_evaluation_report(self, results: Dict):
        """
        Print formatted evaluation report.
        
        Args:
            results: Evaluation results dictionary
        """
        print("=" * 60)
        print("UNDERWATER ACOUSTIC CLASSIFICATION EVALUATION REPORT")
        print("=" * 60)
        
        # Summary metrics
        summary = results['summary']
        print(f"\nSUMMARY METRICS:")
        print(f"Overall IER: {summary['overall_ier']:.4f}")
        print(f"Classification F1: {summary['classification_f1']:.4f}")
        print(f"Detection F1: {summary['detection_f1']:.4f}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        
        # Classification metrics
        classification = results['classification']
        print(f"\nCLASSIFICATION METRICS:")
        print(f"Precision: {classification['precision']:.4f}")
        print(f"Recall: {classification['recall']:.4f}")
        print(f"F1-Score: {classification['f1_score']:.4f}")
        
        # Detection metrics
        detection = results['detection']
        print(f"\nDETECTION METRICS:")
        print(f"Precision: {detection['detection_precision']:.4f}")
        print(f"Recall: {detection['detection_recall']:.4f}")
        print(f"F1-Score: {detection['detection_f1']:.4f}")
        print(f"True Positives: {detection['true_positives']}")
        print(f"False Positives: {detection['false_positives']}")
        print(f"False Negatives: {detection['false_negatives']}")
        
        # Confusion matrix
        print(f"\nCONFUSION MATRIX:")
        cm = np.array(classification['confusion_matrix'])
        class_names = ['vessels', 'marine_animals', 'natural_sounds', 'other_anthropogenic']
        
        print("Predicted ->")
        print("Actual ↓   ", end="")
        for name in class_names:
            print(f"{name[:8]:>8}", end=" ")
        print()
        
        for i, name in enumerate(class_names):
            print(f"{name[:10]:<10}", end=" ")
            for j in range(len(class_names)):
                if i < cm.shape[0] and j < cm.shape[1]:
                    print(f"{cm[i,j]:>8}", end=" ")
                else:
                    print(f"{'0':>8}", end=" ")
            print()
        
        print("=" * 60)

def validate_json_format(json_file: str) -> bool:
    """
    Validate JSON file format according to Appendix-B specification.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        True if valid format, False otherwise
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ['info', 'audios', 'categories', 'annotations']
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False
        
        # Validate annotations format
        for ann in data.get('annotations', []):
            required_ann_fields = ['id', 'audio_id', 'category_id', 'start_time', 'end_time', 'duration', 'score']
            for field in required_ann_fields:
                if field not in ann:
                    print(f"Missing annotation field: {field}")
                    return False
        
        print("JSON format validation passed")
        return True
        
    except Exception as e:
        print(f"JSON validation error: {e}")
        return False
