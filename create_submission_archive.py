#!/usr/bin/env python3
"""Create compressed submission archive with audio files and JSON"""
import os
import argparse
import zipfile
import glob


def create_submission_archive(audio_dir, json_file, output_zip='submission.zip'):
    """Create ZIP archive with audio files and submission JSON"""
    
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        return
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        return
    
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    if not audio_files:
        print(f"❌ No WAV files found in {audio_dir}")
        return
    
    print(f"Creating submission archive...")
    print(f"Audio files: {len(audio_files)}")
    print(f"JSON file: {json_file}")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # Add JSON file
        zipf.write(json_file, os.path.basename(json_file))
        print(f"✓ Added {json_file}")
        
        # Add audio files
        for i, audio_path in enumerate(sorted(audio_files), 1):
            filename = os.path.basename(audio_path)
            zipf.write(audio_path, f"audio/{filename}")
            if i % 10 == 0:
                print(f"✓ Added {i}/{len(audio_files)} audio files...")
        
        print(f"✓ Added all {len(audio_files)} audio files")
    
    zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n✓ Archive created: {output_zip}")
    print(f"✓ Archive size: {zip_size_mb:.1f}MB")
    
    if zip_size_mb > 200:
        print(f"\n⚠ Warning: Archive is still larger than 200MB")
        print(f"   Consider compressing audio files first using compress_audio.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create submission archive')
    parser.add_argument('--audio_dir', required=True, help='Directory with audio files')
    parser.add_argument('--json', required=True, help='Submission JSON file')
    parser.add_argument('--output', default='submission.zip', help='Output ZIP file')
    
    args = parser.parse_args()
    
    create_submission_archive(args.audio_dir, args.json, args.output)
