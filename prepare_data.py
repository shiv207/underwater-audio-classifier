"""
Data preparation script for underwater acoustic classification.
Organizes existing audio files into proper training structure.
"""

import os
import shutil
import argparse
from pathlib import Path
import json
from typing import Dict, List

def organize_existing_data(source_dir: str, target_dir: str) -> Dict[str, int]:
    """
    Organize existing underwater acoustic data into training structure.
    
    Args:
        source_dir: Source directory with existing audio files
        target_dir: Target directory for organized data
        
    Returns:
        Dictionary with file counts per category
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory structure
    categories = ['vessels', 'marine_animals', 'natural_sounds', 'other_anthropogenic']
    for category in categories:
        (target_path / category).mkdir(parents=True, exist_ok=True)
    
    file_counts = {category: 0 for category in categories}
    
    # Map source directories to target categories
    directory_mapping = {
        'acquatic_mammels': 'marine_animals',
        'natural_sounds': 'natural_sounds', 
        'Other_anthropogenic': 'other_anthropogenic',
        'vessels': 'vessels'  # If it exists
    }
    
    # Copy files from source directories
    for source_subdir, target_category in directory_mapping.items():
        source_subpath = source_path / source_subdir
        
        if not source_subpath.exists():
            print(f"Warning: {source_subpath} does not exist, skipping...")
            continue
        
        target_category_path = target_path / target_category
        
        # Copy audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        for ext in audio_extensions:
            for audio_file in source_subpath.glob(f'*{ext}'):
                target_file = target_category_path / audio_file.name
                
                # Avoid overwriting existing files
                counter = 1
                original_target = target_file
                while target_file.exists():
                    stem = original_target.stem
                    suffix = original_target.suffix
                    target_file = target_category_path / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.copy2(audio_file, target_file)
                file_counts[target_category] += 1
                print(f"Copied: {audio_file.name} -> {target_category}/{target_file.name}")
    
    return file_counts

def create_sample_annotations(data_dir: str, output_file: str):
    """
    Create sample annotation file for training/evaluation.
    
    Args:
        data_dir: Directory containing organized audio files
        output_file: Output JSON annotation file
    """
    data_path = Path(data_dir)
    
    # Category mapping
    category_mapping = {
        'vessels': 1,
        'marine_animals': 2,
        'natural_sounds': 3,
        'other_anthropogenic': 4
    }
    
    # Create annotation structure
    annotations = {
        'info': {
            'description': 'Underwater Acoustic Training Data',
            'version': '1.0',
            'year': 2024,
            'contributor': 'UDA Training System',
            'date_created': '2024-01-01T00:00:00'
        },
        'audios': [],
        'categories': [
            {'id': 1, 'name': 'vessels', 'supercategory': 'anthropogenic'},
            {'id': 2, 'name': 'marine_animals', 'supercategory': 'biological'},
            {'id': 3, 'name': 'natural_sounds', 'supercategory': 'natural'},
            {'id': 4, 'name': 'other_anthropogenic', 'supercategory': 'anthropogenic'}
        ],
        'annotations': []
    }
    
    audio_id = 1
    annotation_id = 1
    
    # Process each category
    for category_name, category_id in category_mapping.items():
        category_path = data_path / category_name
        
        if not category_path.exists():
            continue
        
        # Find audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        for ext in audio_extensions:
            for audio_file in category_path.glob(f'*{ext}'):
                # Add audio info
                audio_info = {
                    'id': audio_id,
                    'file_name': str(audio_file.relative_to(data_path)),
                    'duration': 60.0  # Default duration, should be computed from actual file
                }
                annotations['audios'].append(audio_info)
                
                # Add annotation (assume entire file is one event)
                annotation = {
                    'id': annotation_id,
                    'audio_id': audio_id,
                    'category_id': category_id,
                    'start_time': 0,
                    'end_time': 60,
                    'duration': 60.0,
                    'score': 1.0
                }
                annotations['annotations'].append(annotation)
                
                audio_id += 1
                annotation_id += 1
    
    # Save annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created annotation file: {output_file}")
    print(f"Total audio files: {len(annotations['audios'])}")
    print(f"Total annotations: {len(annotations['annotations'])}")

def download_sample_data(target_dir: str):
    """
    Download sample underwater acoustic data for testing.
    This is a placeholder - in practice, you would download from actual sources.
    
    Args:
        target_dir: Directory to save sample data
    """
    target_path = Path(target_dir)
    
    print("Sample data download functionality:")
    print("In a real implementation, this would download from sources like:")
    print("- DOSITS (Discovery of Sound in the Sea)")
    print("- MBARI (Monterey Bay Aquarium Research Institute)")
    print("- DeepShip dataset")
    print("- VTUAD (Vessel Type Underwater Acoustic Dataset)")
    print("- Watkins Marine Mammal Sound Database")
    print("- OceanShip dataset")
    
    # Create placeholder structure
    categories = ['vessels', 'marine_animals', 'natural_sounds', 'other_anthropogenic']
    for category in categories:
        category_path = target_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder README
        readme_path = category_path / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write(f"Placeholder for {category} audio files.\n")
            f.write("Add your audio files here for training.\n")
    
    print(f"Created sample data structure in: {target_dir}")

def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description='Prepare underwater acoustic training data')
    
    parser.add_argument('--source-dir', type=str,
                       help='Source directory with existing audio files')
    parser.add_argument('--target-dir', type=str, default='data/training',
                       help='Target directory for organized training data')
    parser.add_argument('--create-annotations', action='store_true',
                       help='Create sample annotation file')
    parser.add_argument('--annotation-file', type=str, default='data/annotations.json',
                       help='Output annotation file')
    parser.add_argument('--download-samples', action='store_true',
                       help='Create sample data structure')
    
    args = parser.parse_args()
    
    # Create target directory
    os.makedirs(args.target_dir, exist_ok=True)
    
    if args.download_samples:
        download_sample_data(args.target_dir)
    
    if args.source_dir:
        print(f"Organizing data from {args.source_dir} to {args.target_dir}")
        file_counts = organize_existing_data(args.source_dir, args.target_dir)
        
        print("\nFile counts by category:")
        for category, count in file_counts.items():
            print(f"  {category}: {count} files")
        
        total_files = sum(file_counts.values())
        print(f"Total files organized: {total_files}")
    
    if args.create_annotations:
        create_sample_annotations(args.target_dir, args.annotation_file)
    
    print("\nData preparation completed!")
    print(f"Training data directory: {args.target_dir}")
    
    if args.create_annotations:
        print(f"Annotation file: {args.annotation_file}")
    
    print("\nNext steps:")
    print("1. Add more audio files to each category directory")
    print("2. Run training: python train.py --data-dir data/training")
    print("3. Test inference: python main.py --input data/test.wav --output results.json")

if __name__ == '__main__':
    main()
