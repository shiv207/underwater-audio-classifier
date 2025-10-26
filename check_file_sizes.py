#!/usr/bin/env python3
"""Check audio file sizes and identify files that need compression"""
import os
import glob
import argparse


def check_file_sizes(directory, max_size_mb=200):
    """Check all WAV files and report sizes"""
    
    audio_files = glob.glob(os.path.join(directory, '*.wav'))
    
    if not audio_files:
        print(f"❌ No WAV files found in {directory}")
        return
    
    print(f"Checking {len(audio_files)} audio files...\n")
    
    large_files = []
    ok_files = []
    total_size = 0
    
    for audio_path in sorted(audio_files):
        filename = os.path.basename(audio_path)
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        total_size += file_size_mb
        
        if file_size_mb > max_size_mb:
            print(f"⚠️  {filename}: {file_size_mb:.1f}MB (TOO LARGE)")
            large_files.append((filename, file_size_mb))
        else:
            print(f"✅ {filename}: {file_size_mb:.1f}MB")
            ok_files.append((filename, file_size_mb))
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total files: {len(audio_files)}")
    print(f"  OK files: {len(ok_files)}")
    print(f"  Large files (>{max_size_mb}MB): {len(large_files)}")
    print(f"  Total size: {total_size:.1f}MB")
    
    if large_files:
        print(f"\n⚠️  Files that need compression:")
        for filename, size in large_files:
            print(f"     {filename} ({size:.1f}MB)")
        print(f"\nRun: python compress_audio.py --input_dir {directory} --output_dir compressed_files")
    else:
        print(f"\n✅ All files are within size limit!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check audio file sizes')
    parser.add_argument('--dir', required=True, help='Directory with audio files')
    parser.add_argument('--max_size', type=float, default=200, help='Max size in MB (default: 200)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"❌ Directory not found: {args.dir}")
        exit(1)
    
    check_file_sizes(args.dir, args.max_size)
