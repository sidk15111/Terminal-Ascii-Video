# 🎥 Terminal Video ASCII Player

A simple command-line tool that converts video frames into ASCII art and plays them directly in the terminal.  

## ✨ Features
- Play videos as ASCII animation in real time.  
- Adjustable frame rate and resolution.  
- Supports color or grayscale ASCII output.  
- Works with most video formats (via `ffmpeg` or `opencv`).  
- Pause, resume, and quit controls.  

## 📦 Requirements
- Python 3.8+  
- `opencv-python`  
- `numpy`  
- `ffmpeg` (installed on system, if required for decoding)  

Install dependencies:  
```bash
pip install opencv-python numpy
```

## 🚀 Usage
Run the player with a video file:  
```bash
python ascii_player.py path/to/video.mp4
```

### Options
| Flag | Description | Example |
|------|-------------|---------|
| `-w`, `--width`  | Set output width (default: auto) | `-w 80` |
| `-f`, `--fps`    | Force playback FPS | `-f 24` |
| `-c`, `--color`  | Enable colored ASCII output | `-c` |
| `-r`, `--reverse`| Invert ASCII character mapping | `-r` |

### Controls
- **Space** → Pause / Resume  
- **Q** → Quit  

## 🖼 Example
Plain ASCII preview (short demo):  

```
@@@@@@@@@@@%%%%%%%%#######*******++++++==--:.
@@@@@@@@@@%%%%%%%%#######*******++++++==--:..
@@@@@@@@%%%%%%%%#######*******++++++==--::...
@@@@@@%%%%%%%%#######*******++++++==--::.....
```

## 📂 Project Structure
```
ascii-video-player/
│── ascii_player.py     # Main script
│── utils.py            # Helper functions (ASCII mapping, scaling, etc.)
│── README.md           # Project documentation
│── requirements.txt    # Dependencies
```

## 🔮 Future Improvements
- Audio playback sync with ASCII video.  
- Live webcam feed in ASCII.  
- Export ASCII video as `.txt` or `.gif`.  

## 📜 License
MIT License — feel free to use and modify.  
