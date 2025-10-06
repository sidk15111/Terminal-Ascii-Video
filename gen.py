import cv2
import os
import json
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

# More detailed ASCII characters (densest to lightest)
ASCII_CHARS = "$@B%8&WM#*oahkbdpqwmZ0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
UNICODE_BLOCKS = "█▓▒░ "[::-1]

def brightness(r, g, b):
    """Calculate brightness using luminance formula"""
    return int(0.299 * r + 0.587 * g + 0.114 * b)

def map_to_char(brightness_val, charset):
    """Map brightness value to ASCII character"""
    index = brightness_val * (len(charset) - 1) // 255
    return charset[index]

def convert_frame_to_ascii_color_optimized(frame, width=80, use_color=True, unicode=False):
    """Optimized version with vectorized operations"""
    charset = np.array(list(UNICODE_BLOCKS if unicode else ASCII_CHARS))
    
    height = int(frame.shape[0] * width / frame.shape[1] / 2) 
    if height == 0:
        height = 1
        
    resized_frame = cv2.resize(frame, (width, height))

    # Convert BGR to RGB
    if len(resized_frame.shape) > 2:
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = resized_frame
        
    if use_color and len(rgb_frame.shape) == 3:
        # Vectorized brightness calculation
        brightness_vals = (0.299 * rgb_frame[:,:,0] + 
                          0.587 * rgb_frame[:,:,1] + 
                          0.114 * rgb_frame[:,:,2]).astype(int)
        
        # Map to character indices
        char_indices = (brightness_vals * (len(charset) - 1) // 255).astype(int)
        chars = charset[char_indices]
        
        # Build colored output
        ascii_frame = ""
        for i in range(height):
            line = ""
            for j in range(width):
                r, g, b = rgb_frame[i, j]
                char = chars[i, j]
                line += f"\033[38;2;{r};{g};{b}m{char}\033[0m"
            ascii_frame += line + "\n"
    else:
        # Grayscale fallback
        if len(rgb_frame.shape) == 3:
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        else:
            gray_frame = rgb_frame
            
        char_indices = (gray_frame * (len(charset) - 1) // 255).astype(int)
        chars = charset[char_indices]
        ascii_frame = '\n'.join([''.join(row) for row in chars])
    
    return ascii_frame

def pre_render_video_to_frames(video_path, output_dir, width=80, use_color=True, unicode=False):
    """Pre-render entire video to ASCII frames with progress tracking"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info:")
    print(f"  Original resolution: {width_orig}x{height_orig}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"  ASCII width: {width}")
    print(f"  Color mode: {'ON' if use_color else 'OFF'}")
    print(f"  Character set: {'Unicode blocks' if unicode else 'ASCII'}")
    print()
    
    # Save metadata
    metadata = {
        'total_frames': total_frames,
        'fps': fps,
        'original_resolution': [width_orig, height_orig],
        'ascii_width': width,
        'use_color': use_color,
        'unicode': unicode,
        'video_path': video_path,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Process frames with progress bar
    frame_count = 0
    failed_frames = 0
    
    try:
        with tqdm(total=total_frames, desc="Rendering frames", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                 ncols=100) as pbar:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Convert frame to ASCII
                    ascii_art = convert_frame_to_ascii_color_optimized(frame, width, use_color, unicode)
                    
                    # Save frame to file
                    frame_filename = output_path / f"frame_{frame_count:06d}.txt"
                    with open(frame_filename, 'w', encoding='utf-8') as f:
                        f.write(ascii_art)
                    
                    frame_count += 1
                    pbar.update(1)
                    
                    # Update progress bar with stats every 100 frames
                    if frame_count % 100 == 0:
                        pbar.set_postfix({
                            'Frame': frame_count,
                            'Failed': failed_frames,
                        })
                        
                except Exception as e:
                    failed_frames += 1
                    print(f"\nError processing frame {frame_count}: {e}")
                    pbar.update(1)
                    continue
        
        cap.release()
        
        print(f"\n✅ Pre-rendering complete!")
        print(f"   Processed: {frame_count} frames")
        print(f"   Failed: {failed_frames} frames") 
        print(f"   Output directory: {output_path.absolute()}")
        print(f"   Storage size: ~{estimate_storage_size(output_path)}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Pre-rendering interrupted at frame {frame_count}")
        cap.release()
        return False

def estimate_storage_size(output_path):
    """Estimate storage size of rendered frames"""
    try:
        # Check a few sample files to estimate average size
        sample_files = list(output_path.glob("frame_*.txt"))[:10]
        if not sample_files:
            return "Unknown"
            
        total_size = sum(f.stat().st_size for f in sample_files)
        avg_size = total_size / len(sample_files)
        
        # Estimate total size
        total_files = len(list(output_path.glob("frame_*.txt")))
        estimated_total = avg_size * total_files
        
        # Convert to readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if estimated_total < 1024:
                return f"{estimated_total:.1f} {unit}"
            estimated_total /= 1024
        return f"{estimated_total:.1f} TB"
        
    except Exception:
        return "Unknown"

if __name__ == "__main__":
    print("🎬 ASCII Video Pre-Renderer")
    print("=" * 50)
    
    video_path = input("Enter the path to the video file: ").strip()
    output_dir = input("Enter output directory (default: ascii_frames): ").strip() or "ascii_frames"
    
    try:
        width = int(input("Enter ASCII width (default 80): ") or "80")
    except ValueError:
        width = 80

    use_color = input("Enable color? (y/n, default y): ").strip().lower() != 'n'
    unicode_mode = input("Use Unicode blocks? (y/n, default n): ").strip().lower() == 'y'
    
    print("\nStarting pre-rendering...")
    success = pre_render_video_to_frames(video_path, output_dir, width, use_color, unicode_mode)
    
    if success:
        print("\nNext steps:")
        print(f"1. Run the playback script: python ascii_video_player.py")
        print(f"2. Select the directory: {output_dir}")
