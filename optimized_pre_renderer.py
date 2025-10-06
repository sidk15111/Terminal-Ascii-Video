
#!/usr/bin/env python3
"""
optimized_pre_renderer.py

Optimized version of the ASCII pre-renderer:
- Uses numpy vectorization for brightness -> char mapping (already present).
- Adds multiprocessing to process frames in parallel.
- Optionally uses OpenCV CUDA (if available) for resizing/color conversion to speed up preprocessing.
- Measures per-frame processing times and prints mean ± std in milliseconds (matches user's requested formatting).
"""
import cv2
import os
import json
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing
import math

# Keep the user's ASCII sets (reversed order was in the original)
ASCII_CHARS = "$@B%8&WM#*oahkbdpqwmZ0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
UNICODE_BLOCKS = "█▓▒░ "[::-1]


def _brightness_vectorized(rgb_frame):
    """Return brightness image using vectorized luminance formula (uint8)."""
    return (0.299 * rgb_frame[:, :, 0] +
            0.587 * rgb_frame[:, :, 1] +
            0.114 * rgb_frame[:, :, 2]).astype(np.uint8)


def _map_brightness_to_chars(brightness_vals, charset):
    """Map 2D brightness array to 2D array of characters using vectorized indexing."""
    indices = (brightness_vals.astype(np.int32) * (len(charset) - 1) // 255).astype(np.int32)
    return charset[indices]


def _build_ascii_lines_from_chars_and_rgb(chars2d, rgb_frame, use_color):
    """Build a single string for the whole ascii frame. Joins rows for speed."""
    h, w = chars2d.shape
    lines = []
    if use_color:
        for i in range(h):
            row_chars = chars2d[i]
            r_row = rgb_frame[i, :, 0]
            g_row = rgb_frame[i, :, 1]
            b_row = rgb_frame[i, :, 2]
            line_parts = [f"\033[38;2;{r_row[j]};{g_row[j]};{b_row[j]}m{row_chars[j]}\033[0m"
                          for j in range(w)]
            lines.append(''.join(line_parts))
    else:
        for i in range(h):
            lines.append(''.join(chars2d[i]))
    return '\n'.join(lines)


def worker_process_frame(args):
    frame_index, frame_bytes, shape, dtype, width, use_color, unicode_mode = args
    start = time.time()
    try:
        arr = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)
        if arr.ndim == 3 and arr.shape[2] == 3:
            rgb = arr[:, :, ::-1]
        else:
            rgb = arr if arr.ndim == 3 else np.stack([arr]*3, axis=-1)

        height = int(arr.shape[0] * width / arr.shape[1] / 2)
        height = max(1, height)

        rgb_resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

        charset = np.array(list(UNICODE_BLOCKS if unicode_mode else ASCII_CHARS))

        brightness_vals = _brightness_vectorized(rgb_resized)

        chars2d = _map_brightness_to_chars(brightness_vals, charset)

        ascii_frame = _build_ascii_lines_from_chars_and_rgb(chars2d, rgb_resized, use_color)

        elapsed_ms = (time.time() - start) * 1000.0
        return (frame_index, ascii_frame, elapsed_ms)
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000.0
        return (frame_index, f"ERROR:{e}", elapsed_ms)


def try_gpu_preprocess(frame, width):
    if not hasattr(cv2, 'cuda'):
        raise RuntimeError("OpenCV CUDA module not available")
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(frame)
    height = int(frame.shape[0] * width / frame.shape[1] / 2)
    height = max(1, height)
    try:
        gpu_resized = cv2.cuda.resize(gpu_mat, (width, height), interpolation=cv2.INTER_AREA)
        resized = gpu_resized.download()
        if resized.ndim == 3 and resized.shape[2] == 3:
            resized = resized[:, :, ::-1]
        return resized
    except Exception as e:
        raise RuntimeError(f"CUDA preprocessing failed: {e}")


def pre_render_video_to_frames_optimized(video_path, output_dir, width=80, use_color=True, unicode=False,
                                         use_gpu=False, processes=None, write_files=True):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
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
    print(f"  Using GPU preprocessing: {'YES' if use_gpu else 'NO'}")
    print()

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

    frame_count = 0
    failed_frames = 0

    processing_times_ms = []
    total_processing_time_ms = 0.0
    total_processing_time_ms_sq = 0.0

    cpu_cores = cpu_count() if processes is None else max(1, processes)
    use_multiprocessing = not use_gpu and cpu_cores > 1

    pool = None
    try:
        if use_multiprocessing:
            proc_count = max(1, cpu_cores - 1)
            pool = Pool(processes=proc_count)
            pending = []
        else:
            pool = None

        args_iterable = []
        pbar = tqdm(total=total_frames, desc="Rendering frames",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                    ncols=100)

        frame_index = 0
        CHUNKSIZE = 16

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if use_gpu:
                    processed_rgb = try_gpu_preprocess(frame, width)
                    start = time.time()
                    charset = np.array(list(UNICODE_BLOCKS if unicode else ASCII_CHARS))
                    brightness_vals = _brightness_vectorized(processed_rgb)
                    chars2d = _map_brightness_to_chars(brightness_vals, charset)
                    ascii_frame = _build_ascii_lines_from_chars_and_rgb(chars2d, processed_rgb, use_color)
                    elapsed_ms = (time.time() - start) * 1000.0
                    processing_times_ms.append(elapsed_ms)
                    total_processing_time_ms += elapsed_ms
                    total_processing_time_ms_sq += elapsed_ms * elapsed_ms

                    if write_files:
                        fname = output_path / f"frame_{frame_index:06d}.txt"
                        with open(fname, 'w', encoding='utf-8') as outf:
                            outf.write(ascii_frame)

                    frame_index += 1
                    pbar.update(1)

                else:
                    frame_bytes = frame.tobytes()
                    args_iterable.append((frame_index, frame_bytes, frame.shape, frame.dtype.name, width,
                                          use_color, unicode))
                    frame_index += 1

                    if use_multiprocessing and len(args_iterable) >= CHUNKSIZE:
                        results = pool.map(worker_process_frame, args_iterable, chunksize=4)
                        for idx, ascii_frame, elapsed_ms in results:
                            processing_times_ms.append(elapsed_ms)
                            total_processing_time_ms += elapsed_ms
                            total_processing_time_ms_sq += elapsed_ms * elapsed_ms
                            if write_files:
                                fname = output_path / f"frame_{idx:06d}.txt"
                                with open(fname, 'w', encoding='utf-8') as outf:
                                    outf.write(ascii_frame)
                            pbar.update(1)
                        args_iterable = []

            except Exception as e:
                failed_frames += 1
                print(f"\nError processing frame {frame_index}: {e}")
                pbar.update(1)
                continue

        if use_multiprocessing and args_iterable:
            results = pool.map(worker_process_frame, args_iterable, chunksize=4)
            for idx, ascii_frame, elapsed_ms in results:
                processing_times_ms.append(elapsed_ms)
                total_processing_time_ms += elapsed_ms
                total_processing_time_ms_sq += elapsed_ms * elapsed_ms
                if write_files:
                    fname = output_path / f"frame_{idx:06d}.txt"
                    with open(fname, 'w', encoding='utf-8') as outf:
                        outf.write(ascii_frame)
                pbar.update(1)

        pbar.close()
        cap.release()

        frame_count = len(list(output_path.glob("frame_*.txt")))

        n = len(processing_times_ms)
        if n > 0:
            mean_ms = total_processing_time_ms / n
            variance = max(0.0, (total_processing_time_ms_sq / n) - (mean_ms * mean_ms))
            std_ms = math.sqrt(variance)
        else:
            mean_ms = 0.0
            std_ms = 0.0

        timing_summary = f"{mean_ms:.1f} ms (± {std_ms:.2f})"

        print(f"\n✅ Pre-rendering complete!")
        print(f"   Processed: {frame_count} frames")
        print(f"   Failed: {failed_frames} frames")
        print(f"   Output directory: {output_path.absolute()}")
        print(f"   Avg processing time per frame: {timing_summary}")
        print(f"   Storage size: ~{estimate_storage_size(output_path)}")

        return True

    except KeyboardInterrupt:
        print(f"\n⚠️ Pre-rendering interrupted")
        cap.release()
        if pool:
            pool.terminate()
        return False
    finally:
        if pool:
            pool.close()
            pool.join()


def estimate_storage_size(output_path):
    try:
        sample_files = list(output_path.glob("frame_*.txt"))[:10]
        if not sample_files:
            return "Unknown"
        total_size = sum(f.stat().st_size for f in sample_files)
        avg_size = total_size / len(sample_files)
        total_files = len(list(output_path.glob("frame_*.txt")))
        estimated_total = avg_size * total_files
        for unit in ['B', 'KB', 'MB', 'GB']:
            if estimated_total < 1024:
                return f"{estimated_total:.1f} {unit}"
            estimated_total /= 1024
        return f"{estimated_total:.1f} TB"
    except Exception:
        return "Unknown"


if __name__ == "__main__":
    print("🎬 Optimized ASCII Video Pre-Renderer")
    print("=" * 60)

    video_path = input("Enter the path to the video file: ").strip()
    output_dir = input("Enter output directory (default: ascii_frames): ").strip() or "ascii_frames"

    try:
        width = int(input("Enter ASCII width (default 80): ") or "80")
    except ValueError:
        width = 80

    use_color = input("Enable color? (y/n, default y): ").strip().lower() != 'n'
    unicode_mode = input("Use Unicode blocks? (y/n, default n): ").strip().lower() == 'y'
    use_gpu = input("Use GPU preprocessing if available? (y/n, default y): ").strip().lower() != 'n'
    processes = None
    try:
        pc = int(input(f"Number of processes to use (default: auto {cpu_count()}): ") or str(cpu_count()))
        processes = pc
    except ValueError:
        processes = None

    print("\nStarting pre-rendering...")
    success = pre_render_video_to_frames_optimized(video_path, output_dir, width, use_color, unicode_mode,
                                                   use_gpu, processes, write_files=True)

    if success:
        print("\nNext steps:")
        print(f"1. Run the playback script: python ascii_video_player.py")
        print(f"2. Select the directory: {output_dir}")
