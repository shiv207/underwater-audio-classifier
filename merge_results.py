#!/usr/bin/env python3
"""Merge individual result JSON files into a single submission file"""
import json
import glob
import os


def merge_results(input_pattern='result_*.json', output_file='final_submission.json'):
    """Merge all result_*.json files into a single submission"""
    
    # Find all result files
    result_files = sorted(glob.glob(input_pattern), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not result_files:
        print("❌ No result_*.json files found")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Initialize the merged structure
    merged = {
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
    
    audio_id_counter = 1
    annotation_id_counter = 1
    
    # Process each result file
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract audio info
            if data.get('audios') and len(data['audios']) > 0:
                audio = data['audios'][0].copy()
                audio['id'] = audio_id_counter
                audio['file_path'] = "to be mentioned by the participants"
                merged['audios'].append(audio)
                
                # Extract annotations and update IDs
                if data.get('annotations'):
                    for annotation in data['annotations']:
                        new_annotation = annotation.copy()
                        new_annotation['id'] = annotation_id_counter
                        new_annotation['audio_id'] = audio_id_counter
                        merged['annotations'].append(new_annotation)
                        annotation_id_counter += 1
                
                audio_id_counter += 1
                print(f"✓ Processed {result_file}")
        
        except Exception as e:
            print(f"⚠ Error processing {result_file}: {e}")
            continue
    
    # Save merged result
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n✓ Merged {len(merged['audios'])} audio files")
    print(f"✓ Total annotations: {len(merged['annotations'])}")
    print(f"✓ Saved to: {output_file}")
    
    return merged


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge result JSON files into single submission')
    parser.add_argument('--input', default='result_*.json', help='Input file pattern (default: result_*.json)')
    parser.add_argument('--output', default='final_submission.json', help='Output file (default: final_submission.json)')
    args = parser.parse_args()
    
    merge_results(args.input, args.output)
