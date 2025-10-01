import cv2
import os
import time
import numpy as np

# More detailed ASCII characters (densest to lightest)
ASCII_CHARS = "$@B%8&WM#*oahkbdpqwmZ0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
UNICODE_BLOCKS = "█▓▒░ "[::-1]  # Optional: Full blocks to quarter blocks

def brightness(r, g, b):
    """Calculate brightness using luminance formula"""
    return int(0.299 * r + 0.587 * g + 0.114 * b)

def map_to_char(brightness_val, charset):
    """Map brightness value to ASCII character"""
    index = brightness_val * (len(charset) - 1) // 255
    return charset[index]

def convert_frame_to_ascii_color(frame, width=80, use_color=True, unicode=False):
    """Convert frame to colored ASCII art"""
    charset = UNICODE_BLOCKS if unicode else ASCII_CHARS
    
    height = int(frame.shape[0] * width / frame.shape[1] / 2) 
    if height == 0:
        height = 1
        
    resized_frame = cv2.resize(frame, (width, height))

    # Convert BGR to RGB (OpenCV uses BGR by default)
    if len(resized_frame.shape) > 2:
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = resized_frame
        
    ascii_frame = ""
    
    for row in rgb_frame:
        line = ""
        for pixel in row:
            if len(pixel) >= 3:
                r, g, b = pixel[:3]
                bright = brightness(r, g, b)
                char = map_to_char(bright, charset)
                
                if use_color:
                    # Use ANSI RGB color codes
                    line += f"\033[38;2;{r};{g};{b}m{char}\033[0m"
                else:
                    line += char
            else:
                # Grayscale pixel
                bright = pixel if isinstance(pixel, (int, np.integer)) else int(pixel)
                char = map_to_char(bright, charset)
                line += char
        ascii_frame += line + "\n"
    
    return ascii_frame

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

def play_video_in_terminal_color(video_path, width=80, fps=30, use_color=True, unicode=False, optimized=True):
    """Play video as colored ASCII art in terminal"""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame_time = 1.0 / video_fps if video_fps > 0 else 1.0 / fps
    
    print(f"Video FPS: {video_fps}")
    print(f"Target frame time: {target_frame_time:.4f} seconds")
    print(f"Color mode: {'ON' if use_color else 'OFF'}")
    print(f"Character set: {'Unicode blocks' if unicode else 'ASCII'}")
    print(f"Optimization: {'ON' if optimized else 'OFF'}")
    print("Press Ctrl+C to stop...")
    time.sleep(2)  # Give user time to read info
    
    try:
        frame_count = 0
        total_processing_time = 0
        
        while True:
            # Start timing this frame
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            if optimized:
                ascii_art = convert_frame_to_ascii_color_optimized(frame, width, use_color, unicode)
            else:
                ascii_art = convert_frame_to_ascii_color(frame, width, use_color, unicode)
            
            # Clear screen and display
            os.system('cls' if os.name == 'nt' else 'clear')
            print(ascii_art, end='')
            
            # Calculate how long processing took
            processing_time = time.time() - frame_start_time
            total_processing_time += processing_time
            
            # Calculate remaining time to sleep
            sleep_time = target_frame_time - processing_time
            
            # Display timing info every 60 frames
            frame_count += 1
            if frame_count % 60 == 0:
                avg_processing = total_processing_time / frame_count
                print(f"\nFrame {frame_count}: Processing {processing_time:.4f}s, Sleep {sleep_time:.4f}s")
                print(f"Average processing time: {avg_processing:.4f}s")
                if sleep_time <= 0:
                    print("⚠️  Running behind schedule!")
            
            # Only sleep if we have time remaining
            if sleep_time > 0:
                time.sleep(sleep_time)
            # If sleep_time <= 0, we're running behind schedule
            
    except KeyboardInterrupt:
        print("\nVideo playback interrupted.")
        if frame_count > 0:
            avg_processing = total_processing_time / frame_count
            print(f"Final stats - Average processing time: {avg_processing:.4f}s per frame")
    
    finally:
        cap.release()

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ").strip()
    
    try:
        width = int(input("Enter terminal width (default 80): ") or "80")
    except ValueError:
        width = 80

    try:
        fps = int(input("Enter FPS (default: use video FPS): ") or "0")
    except ValueError:
        fps = 0
        
    use_color = input("Enable color? (y/n, default y): ").strip().lower() != 'n'
    unicode_mode = input("Use Unicode blocks? (y/n, default n): ").strip().lower() == 'y'
    optimized = input("Use optimized processing? (y/n, default y): ").strip().lower() != 'n'
    
    play_video_in_terminal_color(video_path, width, fps, use_color, unicode_mode, optimized)
