# Uni-x Auto Edit

Video Editor tu dong voi Whisper AI va FFmpeg.

## Features

- Tu dong tao phu de bang Whisper AI
- Ho tro nhieu template
- Chuyen canh video tu dong
- Filter va effect phong phu
- Ho tro GPU (NVIDIA NVENC)
- Giao dien tieng Viet

## Quick Start

### Option 1: Run directly (requires Python installed)

```bash
pip install pillow faster-whisper fonttools
python unixautoedit.py
```

### Option 2: Build Portable Package (Windows)

```bash
python build_portable.py
```

This will create a `UnixAutoEdit_Portable` folder with everything needed.

## Portable Package Structure

```
UnixAutoEdit_Portable/
├── Run.bat              # Launch application
├── Run_Debug.bat        # Launch with console (for debugging)
├── python/              # Embedded Python 3.11
├── app/
│   ├── launcher.py      # Auto-update launcher
│   └── unixautoedit.py  # Main application
├── ffmpeg/              # FFmpeg binaries
├── templates/           # Video templates
├── fonts/               # Custom fonts
├── logos/               # Logo images
└── effects/             # Video overlay effects
```

## Auto-Update

The launcher automatically checks for updates from GitHub:
- Version URL: `https://raw.githubusercontent.com/nguyenvantinh2391994-blip/UnixAutoEdit/main/version.txt`
- Script URL: `https://raw.githubusercontent.com/nguyenvantinh2391994-blip/UnixAutoEdit/main/unixautoedit.py`

## Requirements

### For development:
- Python 3.8+
- pillow
- faster-whisper
- fonttools
- FFmpeg (in PATH)

### For portable version:
- Windows 10/11 (64-bit)
- No installation needed

## License

MIT License
# Test auto-merge - Fri Dec 12 05:23:12 UTC 2025

