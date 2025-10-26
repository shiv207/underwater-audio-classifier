#!/usr/bin/env python3
"""Compress large audio files to meet upload size limits"""
import os
import argparse
import glob
import librosa
import soundfile as sf


def compress_audio(input_path, output_path, target_sr=16000, target_bitdepth='PCM_16'):
    """Compress audio file by reducing sample rate and bit depth"""
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Save with compression
        sf.write(output_path, audio, target_sr, subtype=target_bitdepth)
        
        # Get file sizes
        original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        reduction = ((original_size - compressed_size) / original_size) * 100
        
        return original_size, compressed_size, reduction
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None, None, None


def compress_directory(input_dir, output_dir, target_sr=16000, max_size_mb=200):
    """Compress all WAV files in directory that exceed size limit"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = glob.glob(os.path.join(input_dir, '*.wav'))
    
    if not audio_files:
        print(f"No WAV files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Target: {target_sr}Hz, max size: {max_size_mb}MB\n")
    
    compressed_count = 0
    copied_count = 0
    
    for audio_path in sorted(audio_files):
        filename = os.path.basename(audio_path)
        output_path = os.path.join(output_dir, filename)
        
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            print(f"⚠ {filename} ({file_size_mb:.1f}MB) - Compressing...")
            orig_size, comp_size, reduction = compress_audio(audio_path, output_path, target_sr)
            
            if comp_size and comp_size <= max_size_mb:
                print(f"✓ Compressed: {orig_size:.1f}MB → {comp_size:.1f}MB ({reduction:.1f}% reduction)\n")
                compressed_count += 1
            elif comp_size:
                print(f"⚠ Still too large: {comp_size:.1f}MB - Try lower sample rate\n")
                compressed_count += 1
            else:
                print(f"❌ Failed to compress\n")
        else:
            # File is already small enough, just copy
            import shutil
            shutil.copy2(audio_path, output_path)
            print(f"✓ {filename} ({file_size_mb:.1f}MB) - OK")
            copied_count += 1
    
    print(f"\n✓ Compressed: {compressed_count} files")
    print(f"✓ Copied: {copied_count} files")
    print(f"✓ Output directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress audio files to meet size limits')
    parser.add_argument('--input_dir', required=True, help='Input directory with audio files')
    parser.add_argument('--output_dir', default='compressed_audio', help='Output directory')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Target sample rate (default: 16000)')
    parser.add_argument('--max_size', type=float, default=200, help='Max file size in MB (default: 200)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        exit(1)
    
    compress_directory(args.input_dir, args.output_dir, args.sample_rate, args.max_size)
