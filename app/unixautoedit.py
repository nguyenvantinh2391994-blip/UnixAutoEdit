# -*- coding: utf-8 -*-
"""
Uni-x Auto Edit v2.3
- Hỗ trợ đóng gói Portable với Python Embedded
- Auto-update từ GitHub
- Sửa lỗi font tiếng Việt
- Ẩn cửa sổ CMD khi chạy FFmpeg
- Giao diện chuyên nghiệp, tiếng Việt
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import threading
import random
import time
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy

# GUI
TKINTER_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, colorchooser, simpledialog
    TKINTER_AVAILABLE = True
except ImportError as e:
    print("=" * 60)
    print("CRITICAL ERROR: tkinter is not available!")
    print("=" * 60)
    print(f"Error: {e}")
    print()
    print("This application requires tkinter to run.")
    print()
    print("Solutions:")
    print("1. If using portable version: Run build_portable.py again")
    print("2. If using installed Python: Reinstall Python with tkinter")
    print("   On Windows: Check 'tcl/tk and IDLE' during installation")
    print("   On Linux: sudo apt-get install python3-tk")
    print("   On Mac: brew install python-tk")
    print()
    print("Required tkinter files (for portable):")
    print("  - python/DLLs/_tkinter.pyd")
    print("  - python/tcl86t.dll, python/tk86t.dll")
    print("  - python/tcl/ folder")
    print("  - python/Lib/tkinter/ folder")
    print("=" * 60)
    print()
    print("Press Enter to exit...")
    try:
        input()
    except:
        pass
    import sys
    sys.exit(1)

# PIL
try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper_timestamped as whisper_ts
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import whisper as whisper_standard
    WHISPER_STANDARD_AVAILABLE = True
except ImportError:
    WHISPER_STANDARD_AVAILABLE = False

# ===== HIDE CMD WINDOW ON WINDOWS =====
SUBPROCESS_FLAGS = {}
if sys.platform == 'win32':
    SUBPROCESS_FLAGS = {
        'creationflags': subprocess.CREATE_NO_WINDOW,
    }
    # Startup info để ẩn hoàn toàn
    STARTUPINFO = subprocess.STARTUPINFO()
    STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    STARTUPINFO.wShowWindow = subprocess.SW_HIDE
    SUBPROCESS_FLAGS['startupinfo'] = STARTUPINFO

def run_cmd(cmd, capture=True, timeout=None):
    """Chạy command và ẩn cửa sổ CMD"""
    try:
        if capture:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                **SUBPROCESS_FLAGS
            )
        else:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                **SUBPROCESS_FLAGS
            )
        return result
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"Command error: {e}")
        return None

# ===== PATHS =====
# Detect if running in portable mode (from app/ subfolder)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)

# Check for portable structure (python/ and app/ folders in parent)
if (os.path.basename(_SCRIPT_DIR) == "app" and
    os.path.isdir(os.path.join(_PARENT_DIR, "python"))):
    # Portable mode - use parent as base
    BASE_DIR = _PARENT_DIR
    PORTABLE_MODE = True
else:
    # Normal mode - use script directory
    BASE_DIR = _SCRIPT_DIR
    PORTABLE_MODE = False

# Set paths relative to BASE_DIR
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
EFFECTS_DIR = os.path.join(BASE_DIR, "effects")
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
LOGOS_DIR = os.path.join(BASE_DIR, "logos")

# Create directories if they don't exist
for d in [TEMPLATES_DIR, EFFECTS_DIR, FONTS_DIR, LOGOS_DIR]:
    os.makedirs(d, exist_ok=True)

# Add FFmpeg to PATH in portable mode
if PORTABLE_MODE:
    FFMPEG_DIR = os.path.join(BASE_DIR, "ffmpeg")
    if os.path.isdir(FFMPEG_DIR):
        os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# ===== VERSION =====
def get_app_version():
    """Read version from version.txt"""
    version_file = os.path.join(BASE_DIR, "version.txt")
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return "2.0"

APP_VERSION = get_app_version()

# ===== VIETNAMESE FONT SUPPORT =====
VIETNAMESE_FONTS = [
    "Arial", "Tahoma", "Segoe UI", "Times New Roman", "Verdana",
    "Roboto", "Open Sans", "Noto Sans", "Source Sans Pro", "Be Vietnam Pro",
]

# ===== COLORS - PROFESSIONAL DARK THEME =====
COLORS = {
    "bg_main": "#0f0f1a",
    "bg_secondary": "#1a1a2e",
    "bg_card": "#252540",
    "bg_hover": "#2d2d4a",
    "accent": "#6366f1",
    "accent_light": "#818cf8",
    "accent_dark": "#4f46e5",
    "success": "#22c55e",
    "warning": "#eab308",
    "error": "#ef4444",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "text_dim": "#64748b",
    "border": "#374151",
    "input_bg": "#1e293b",
    "input_text": "#f1f5f9",
    "input_border": "#475569",
}

# ===== GPU DETECTION =====
GPU_INFO = {"nvenc": False, "name": "", "vram_mb": 0}

def detect_gpu():
    global GPU_INFO
    try:
        result = run_cmd(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            timeout=10
        )
        if result and result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            GPU_INFO["name"] = parts[0] if parts else ""
            GPU_INFO["vram_mb"] = int(parts[1]) if len(parts) > 1 else 0
    except:
        pass
    
    try:
        result = run_cmd(["ffmpeg", "-hide_banner", "-encoders"], timeout=10)
        if result and "h264_nvenc" in result.stdout:
            test = run_cmd(
                ["ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                timeout=10
            )
            GPU_INFO["nvenc"] = (test and test.returncode == 0)
    except:
        pass
    return GPU_INFO

# ===== EFFECTS =====
FILTER_EFFECTS = {
    "none": {"name": "Không có", "filter": None},
    "film_grain": {"name": "Film Grain", "filter": "noise=alls=20:allf=t+u"},
    "vintage": {"name": "Vintage", "filter": "colorbalance=rs=.3:gs=-.1:bs=-.3,noise=alls=15:allf=t"},
    "cinematic": {"name": "Cinematic", "filter": "eq=contrast=1.1:brightness=0.03:saturation=1.2,vignette=PI/5"},
    "teal_orange": {"name": "Teal & Orange", "filter": "colorbalance=rs=.2:gs=-.1:bs=-.2:rh=.1:bh=.2,eq=saturation=1.3"},
    "cold": {"name": "Lạnh (Cold)", "filter": "colorbalance=rs=-.3:gs=-.1:bs=.4,eq=brightness=0.02"},
    "warm": {"name": "Ấm (Warm)", "filter": "colorbalance=rs=.35:gs=.15:bs=-.25"},
    "bw": {"name": "Đen trắng", "filter": "hue=s=0"},
    "vibrant": {"name": "Tươi sáng", "filter": "eq=saturation=1.5:contrast=1.1"},
    "dreamy": {"name": "Mơ màng", "filter": "gblur=sigma=2,eq=brightness=0.08:saturation=0.85"},
    "vignette": {"name": "Vignette", "filter": "vignette=PI/4"},
    "sharpen": {"name": "Sắc nét", "filter": "unsharp=5:5:1.0"},
    "retro_vhs": {"name": "VHS Retro", "filter": "noise=alls=30:allf=t,eq=saturation=1.4,chromashift=cbh=3:crh=-3"},
}

VIDEO_OVERLAY_EFFECTS = {
    "": {"name": "Không có"},
    "snow": {"name": "Tuyết rơi"},
    "rain": {"name": "Mưa"},
    "bokeh": {"name": "Bokeh"},
    "dust": {"name": "Hạt bụi"},
    "light_leak": {"name": "Light Leak"},
}

# ===== DEFAULT TEMPLATE =====
DEFAULT_TEMPLATE = {
    "name": "Mặc định",
    "target_width": 1920,
    "target_height": 1080,
    "video_fps": 30,
    "font_file": "",
    "font_name": "Arial",
    "font_size": 72,
    "font_color": "#FFFFFF",
    "outline_color": "#000000",
    "outline_width": 3,
    "margin_bottom": 50,
    "logo_enabled": False,
    "logo_file": "",
    "logo_position": "top_right",
    "logo_size_percent": 15,
    "logo_opacity": 0.85,
    "logo_margin": 30,
    "filter_effect": "none",
    "video_overlay": "",
    "overlay_opacity": 0.5,
    "transition_enabled": True,
    "transition_duration": 0.8,
    "transition_type": "random",
    "whisper_model": "tiny",
    "whisper_language": "en",
    "use_gpu": True,
}

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

# ===== UTILITIES =====
def get_audio_duration(path):
    try:
        result = run_cmd([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", path
        ])
        if result and result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None

def get_video_duration(path):
    try:
        result = run_cmd([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", path
        ])
        if result and result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return 5.0

def list_media_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(extensions):
                files.append(os.path.join(folder, f))
    return sorted(files)

def escape_ffmpeg_path(path):
    """Escape path cho FFmpeg - FIX cho Windows"""
    path = path.replace("\\", "/")
    path = path.replace(":", "\\:")
    path = path.replace("'", "\\'")
    return path

def format_timestamp_srt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def hex_to_ass_color(hex_color):
    h = hex_color.lstrip("#")
    if len(h) < 6:
        h = "FFFFFF"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}&".upper()

def format_time(seconds):
    if seconds < 0:
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def get_font_name_from_file(font_path):
    """Đọc tên font thực từ file TTF/OTF"""
    if not font_path or not os.path.exists(font_path):
        return None
    
    try:
        from fontTools.ttLib import TTFont
        font = TTFont(font_path)
        for record in font['name'].names:
            if record.nameID == 1 and record.platformID in (1, 3):
                try:
                    name = record.toUnicode()
                    if name:
                        font.close()
                        return name.strip()
                except:
                    pass
        font.close()
    except ImportError:
        pass
    except:
        pass
    
    try:
        with open(font_path, 'rb') as f:
            data = f.read(12)
            if len(data) < 12:
                return None
            
            num_tables = int.from_bytes(data[4:6], 'big')
            f.seek(12)
            
            name_offset = None
            for _ in range(num_tables):
                table_data = f.read(16)
                if len(table_data) < 16:
                    break
                tag = table_data[0:4].decode('ascii', errors='ignore')
                if tag == 'name':
                    name_offset = int.from_bytes(table_data[8:12], 'big')
                    break
            
            if name_offset:
                f.seek(name_offset)
                name_header = f.read(6)
                count = int.from_bytes(name_header[2:4], 'big')
                string_offset = int.from_bytes(name_header[4:6], 'big')
                
                for i in range(count):
                    record = f.read(12)
                    if len(record) < 12:
                        break
                    
                    platform_id = int.from_bytes(record[0:2], 'big')
                    name_id = int.from_bytes(record[6:8], 'big')
                    length = int.from_bytes(record[8:10], 'big')
                    offset = int.from_bytes(record[10:12], 'big')
                    
                    if name_id == 1 and length > 0:
                        current_pos = f.tell()
                        f.seek(name_offset + string_offset + offset)
                        name_bytes = f.read(length)
                        f.seek(current_pos)
                        
                        try:
                            if platform_id == 3:
                                font_name = name_bytes.decode('utf-16-be')
                            else:
                                font_name = name_bytes.decode('utf-8', errors='ignore')
                            
                            if font_name and font_name.strip():
                                return font_name.strip()
                        except:
                            pass
    except:
        pass
    
    return os.path.splitext(os.path.basename(font_path))[0]

def find_system_font(font_name):
    """Tìm font trong thư mục fonts của app và hệ thống Windows"""
    if not font_name:
        return None

    # Tìm trong FONTS_DIR của app TRƯỚC, sau đó mới tìm trong hệ thống
    font_dirs = [
        FONTS_DIR,  # Thư mục fonts bên cạnh app
        os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts'),
    ]
    
    font_files = {
        'arial': 'arial.ttf',
        'arial bold': 'arialbd.ttf',
        'tahoma': 'tahoma.ttf',
        'tahoma bold': 'tahomabd.ttf',
        'times new roman': 'times.ttf',
        'verdana': 'verdana.ttf',
        'segoe ui': 'segoeui.ttf',
        'segoe ui bold': 'segoeuib.ttf',
        'calibri': 'calibri.ttf',
        'consolas': 'consola.ttf',
    }
    
    font_lower = font_name.lower()
    
    if font_lower in font_files:
        for font_dir in font_dirs:
            path = os.path.join(font_dir, font_files[font_lower])
            if os.path.exists(path):
                return path
    
    for font_dir in font_dirs:
        if not os.path.isdir(font_dir):
            continue
        for f in os.listdir(font_dir):
            if f.lower().endswith(('.ttf', '.otf')):
                if font_lower in f.lower():
                    return os.path.join(font_dir, f)
                fpath = os.path.join(font_dir, f)
                fname = get_font_name_from_file(fpath)
                if fname and font_lower == fname.lower():
                    return fpath
    
    return None

def get_vietnamese_font():
    """Tìm font hỗ trợ tiếng Việt tốt nhất có sẵn"""
    for font_name in VIETNAMESE_FONTS:
        font_path = find_system_font(font_name)
        if font_path:
            return font_path, font_name
    
    arial_path = find_system_font("Arial")
    if arial_path:
        return arial_path, "Arial"
    
    return None, "Arial"

def get_encoder_params(use_gpu=True, for_intermediate=False):
    if use_gpu and GPU_INFO.get("nvenc"):
        if for_intermediate:
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "constqp", "-qp", "23"]
        else:
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    else:
        if for_intermediate:
            return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
        else:
            return ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]

# ===== TEMPLATE MANAGER =====
class TemplateManager:
    def __init__(self):
        self.templates = {}
        self.load_all()
    
    def load_all(self):
        self.templates = {"Mặc định": deepcopy(DEFAULT_TEMPLATE)}
        if os.path.isdir(TEMPLATES_DIR):
            for f in os.listdir(TEMPLATES_DIR):
                if f.endswith(".json"):
                    try:
                        path = os.path.join(TEMPLATES_DIR, f)
                        with open(path, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                            name = data.get("name", os.path.splitext(f)[0])
                            self.templates[name] = data
                    except:
                        pass
    
    def save(self, name, data):
        data["name"] = name
        filename = f"{name.replace(' ', '_').replace('/', '_')}.json"
        path = os.path.join(TEMPLATES_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.templates[name] = data
        return path
    
    def delete(self, name):
        if name == "Mặc định":
            return False
        if name in self.templates:
            for f in os.listdir(TEMPLATES_DIR):
                if f.endswith(".json"):
                    try:
                        path = os.path.join(TEMPLATES_DIR, f)
                        with open(path, "r", encoding="utf-8") as fp:
                            data = json.load(fp)
                            if data.get("name") == name:
                                os.remove(path)
                                break
                    except:
                        pass
            del self.templates[name]
            return True
        return False
    
    def get_list(self):
        return list(self.templates.keys())
    
    def get(self, name):
        return self.templates.get(name, DEFAULT_TEMPLATE)

# ===== WHISPER =====
_WHISPER_MODEL = None
_WHISPER_TYPE = None

def get_whisper_model(model_name="tiny"):
    global _WHISPER_MODEL, _WHISPER_TYPE
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL, _WHISPER_TYPE
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except:
        device = "cpu"
    
    if FASTER_WHISPER_AVAILABLE:
        try:
            compute_type = "float16" if device == "cuda" else "int8"
            _WHISPER_MODEL = FasterWhisperModel(model_name, device=device, compute_type=compute_type)
            _WHISPER_TYPE = "faster"
            return _WHISPER_MODEL, _WHISPER_TYPE
        except:
            pass
    
    if WHISPER_AVAILABLE:
        try:
            _WHISPER_MODEL = whisper_ts.load_model(model_name, device=device)
            _WHISPER_TYPE = "timestamped"
            return _WHISPER_MODEL, _WHISPER_TYPE
        except:
            pass
    
    if WHISPER_STANDARD_AVAILABLE:
        try:
            _WHISPER_MODEL = whisper_standard.load_model(model_name, device=device)
            _WHISPER_TYPE = "standard"
            return _WHISPER_MODEL, _WHISPER_TYPE
        except:
            pass
    
    raise RuntimeError("Không tìm thấy Whisper model!")

def generate_srt(audio_path, output_dir, template):
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "audio.wav")
    run_cmd(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path])
    
    audio_file = wav_path if os.path.exists(wav_path) else audio_path
    model, whisper_type = get_whisper_model(template.get("whisper_model", "tiny"))
    lang = template.get("whisper_language", "en")
    
    result = None
    if whisper_type == "faster":
        segments, info = model.transcribe(audio_file, language=lang or None, word_timestamps=True)
        segments_list = []
        for seg in segments:
            seg_data = {"start": seg.start, "end": seg.end, "text": seg.text, "words": []}
            if seg.words:
                for w in seg.words:
                    seg_data["words"].append({"start": w.start, "end": w.end, "text": w.word})
            segments_list.append(seg_data)
        result = {"segments": segments_list}
    elif whisper_type == "timestamped":
        result = whisper_ts.transcribe(model, audio_file, language=lang or None, verbose=False)
    elif whisper_type == "standard":
        result = model.transcribe(audio_file, language=lang or None, verbose=False)
    
    if not result:
        raise RuntimeError("Whisper thất bại")
    
    raw_words = []
    for seg in result.get("segments", []):
        words = seg.get("words", [])
        if words:
            for w in words:
                txt = (w.get("text") or w.get("word") or "").strip()
                if txt and w.get("start") is not None:
                    raw_words.append({"t0": float(w["start"]), "t1": float(w["end"]), "txt": txt})
        else:
            txt = seg.get("text", "").strip()
            if txt:
                raw_words.append({"t0": float(seg["start"]), "t1": float(seg["end"]), "txt": txt})
    
    cues, buf = [], []
    min_dur, max_dur, max_chars = 0.8, 4.0, 40
    
    def flush():
        nonlocal buf, cues
        if buf:
            t0, t1 = buf[0]["t0"], buf[-1]["t1"]
            if cues:
                t0 = max(t0, cues[-1][1] + 0.05)
            text = " ".join(w["txt"] for w in buf).strip()
            if text:
                cues.append((t0, max(t1, t0 + min_dur), text))
            buf = []
    
    for w in raw_words:
        cand = buf + [w]
        dur = cand[-1]["t1"] - cand[0]["t0"]
        chars = sum(len(x["txt"]) for x in cand)
        if dur >= max_dur or chars >= max_chars:
            flush()
            buf = [w]
        else:
            buf = cand
            if any(c in w["txt"] for c in ".?!"):
                flush()
    flush()
    
    srt_path = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (t0, t1, txt) in enumerate(cues, 1):
            f.write(f"{i}\n{format_timestamp_srt(t0)} --> {format_timestamp_srt(t1)}\n{txt}\n\n")
    
    if os.path.exists(wav_path):
        os.remove(wav_path)
    return srt_path

def srt_to_ass(srt_path, ass_path, template, font_dir=None):
    """Convert SRT to ASS - FIX VIETNAMESE FONT"""
    
    font_file = template.get("font_file", "")
    font_name = template.get("font_name", "Arial")
    
    if font_file and os.path.exists(font_file):
        real_name = get_font_name_from_file(font_file)
        if real_name:
            font_name = real_name
    else:
        sys_font_path = find_system_font(font_name)
        if sys_font_path:
            font_file = sys_font_path
        else:
            vn_font_path, vn_font_name = get_vietnamese_font()
            if vn_font_path:
                font_file = vn_font_path
                font_name = vn_font_name
    
    font_size = template.get("font_size", 72)
    font_color = hex_to_ass_color(template.get("font_color", "#FFFFFF"))
    outline_color = hex_to_ass_color(template.get("outline_color", "#000000"))
    outline = template.get("outline_width", 3)
    margin_v = template.get("margin_bottom", 50)
    play_w = template.get("target_width", 1920)
    play_h = template.get("target_height", 1080)
    
    safe_font_name = font_name.replace(",", "").replace("'", "").replace('"', '')
    
    header = f"""[Script Info]
ScriptType: v4.00+
ScaledBorderAndShadow: yes
WrapStyle: 2
PlayResX: {play_w}
PlayResY: {play_h}
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{safe_font_name},{font_size},{font_color},&H000000FF&,{outline_color},&H80000000&,-1,0,0,0,100,100,0,0,1,{outline},2,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    def srt_to_ass_time(t):
        h, m, sms = t.split(":")
        s, ms = sms.split(",")
        return f"{int(h)}:{m}:{s}.{int(int(ms)/10):02d}"
    
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    
    events = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        i += 1
        if i >= len(lines):
            break
        time_line = lines[i].strip()
        i += 1
        if "-->" not in time_line:
            continue
        start, end = [x.strip() for x in time_line.split("-->")]
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i])
            i += 1
        text = "\\N".join(text_lines)
        events.append(f"Dialogue: 0,{srt_to_ass_time(start)},{srt_to_ass_time(end)},Default,,0,0,0,,{text}")
    
    with open(ass_path, "w", encoding="utf-8-sig") as f:
        f.write(header + "\n".join(events))
    
    return ass_path, font_file

# ===== VIDEO PROCESSING =====
def create_video_clip(path, output, template, duration=None, use_gpu=True):
    w = template.get("target_width", 1920)
    h = template.get("target_height", 1080)
    fps = template.get("video_fps", 30)
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},format=yuv420p"
    
    cmd = ["ffmpeg", "-y"]
    if use_gpu and GPU_INFO.get("nvenc"):
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", path, "-vf", vf]
    cmd += get_encoder_params(use_gpu, for_intermediate=True)
    cmd += ["-an"]
    if duration:
        cmd += ["-t", str(duration)]
    cmd.append(output)
    run_cmd(cmd)
    return output

def create_image_clip(path, output, duration, template, use_gpu=True):
    w = template.get("target_width", 1920)
    h = template.get("target_height", 1080)
    fps = template.get("video_fps", 30)
    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=yuv420p"
    
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", path, "-vf", vf,
           *get_encoder_params(use_gpu, for_intermediate=True),
           "-t", str(duration), "-r", str(fps), "-an", output]
    run_cmd(cmd)
    return output

def apply_transition(clip1, clip2, output, trans_dur=0.8, trans_type="random", use_gpu=True):
    dur1 = get_video_duration(clip1)
    offset = max(0, dur1 - trans_dur)
    if trans_type == "random":
        trans_type = random.choice(["fade", "fadeblack", "wipeleft", "wiperight", "slideleft", "slideright"])
    
    cmd = ["ffmpeg", "-y", "-i", clip1, "-i", clip2,
           "-filter_complex", f"[0:v][1:v]xfade=transition={trans_type}:duration={trans_dur}:offset={offset}[v]",
           "-map", "[v]", *get_encoder_params(use_gpu, for_intermediate=True), "-an", output]
    result = run_cmd(cmd)
    if not result or result.returncode != 0:
        with open(output + ".txt", "w") as f:
            f.write(f"file '{clip1}'\nfile '{clip2}'\n")
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", output + ".txt", "-c", "copy", output])
    return output

def merge_clips(clips, output, template, use_gpu=True):
    if len(clips) == 1:
        shutil.copy2(clips[0], output)
        return output

    trans_enabled = template.get("transition_enabled", True)
    trans_dur = template.get("transition_duration", 0.8)
    trans_type = template.get("transition_type", "random")

    if not trans_enabled:
        concat_file = output + ".txt"
        with open(concat_file, "w") as f:
            for c in clips:
                f.write(f"file '{c}'\n")
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", output])
        return output

    tmp_dir = os.path.dirname(clips[0])
    merged = clips[0]

    for i in range(1, len(clips)):
        out = os.path.join(tmp_dir, f"m_{i}.mp4")
        apply_transition(merged, clips[i], out, trans_dur, trans_type, use_gpu)
        if merged != clips[0] and os.path.exists(merged):
            os.remove(merged)
        merged = out

    shutil.move(merged, output)
    return output

def create_overlay_video(effect_type, output_path, duration=10, width=1920, height=1080):
    fps = 30
    if effect_type == "snow":
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
               "-vf", "noise=alls=100:allf=t,eq=brightness=-0.3:contrast=3,gblur=sigma=1.5,scroll=v=0.01:h=0",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", output_path]
    elif effect_type == "rain":
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
               "-vf", "noise=alls=80:allf=t,eq=brightness=-0.4:contrast=4,avgblur=sizeX=1:sizeY=10,scroll=v=0.05:h=0",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", output_path]
    elif effect_type == "bokeh":
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
               "-vf", "noise=alls=50:allf=t,eq=brightness=-0.5:contrast=5,gblur=sigma=25",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", output_path]
    elif effect_type == "dust":
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
               "-vf", "noise=alls=30:allf=t,eq=brightness=-0.6:contrast=6,gblur=sigma=2",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", output_path]
    elif effect_type == "light_leak":
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"gradients=s={width}x{height}:d={duration}:r={fps}:c0=orange:c1=red:c2=yellow:c3=orange",
               "-vf", "gblur=sigma=80,eq=brightness=-0.3:saturation=1.5,format=yuv420p",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", output_path]
    else:
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
               "-vf", "noise=alls=30:allf=t", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", output_path]
    run_cmd(cmd)
    return os.path.exists(output_path)

def render_final(visual_path, audio_path, output_path, template, ass_path=None, font_dir=None, duration=None, use_gpu=True):
    """Render final video - FIX VIETNAMESE SUBTITLES"""
    
    inputs = []
    if use_gpu and GPU_INFO.get("nvenc"):
        inputs += ["-hwaccel", "cuda"]
    inputs += ["-i", visual_path, "-i", audio_path]
    
    input_idx = 2
    
    video_overlay_key = template.get("video_overlay", "")
    overlay_idx = None
    if video_overlay_key and video_overlay_key in VIDEO_OVERLAY_EFFECTS:
        overlay_path = os.path.join(EFFECTS_DIR, f"{video_overlay_key}.mp4")
        if os.path.exists(overlay_path):
            inputs += ["-stream_loop", "-1", "-i", overlay_path]
            overlay_idx = input_idx
            input_idx += 1
    
    logo_idx = None
    logo_file = template.get("logo_file", "")
    if template.get("logo_enabled") and logo_file:
        if not os.path.isabs(logo_file):
            logo_file = os.path.join(LOGOS_DIR, logo_file)
        if os.path.exists(logo_file):
            inputs += ["-i", logo_file]
            logo_idx = input_idx
            input_idx += 1
    
    w = template.get("target_width", 1920)
    h = template.get("target_height", 1080)
    
    filter_parts = []
    label_idx = 0
    
    filter_parts.append(f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}[v{label_idx}]")
    current_label = f"v{label_idx}"
    label_idx += 1
    
    if overlay_idx is not None:
        opacity = template.get("overlay_opacity", 0.5)
        filter_parts.append(f"[{overlay_idx}:v]scale={w}:{h},format=rgba,colorchannelmixer=aa={opacity}[ov]")
        filter_parts.append(f"[{current_label}][ov]blend=all_mode=screen:all_opacity=1[v{label_idx}]")
        current_label = f"v{label_idx}"
        label_idx += 1
    
    effect_key = template.get("filter_effect", "none")
    if effect_key != "none" and effect_key in FILTER_EFFECTS:
        effect_filter = FILTER_EFFECTS[effect_key].get("filter")
        if effect_filter:
            filter_parts.append(f"[{current_label}]{effect_filter}[v{label_idx}]")
            current_label = f"v{label_idx}"
            label_idx += 1
    
    if logo_idx is not None:
        position = template.get("logo_position", "top_right")
        size_pct = template.get("logo_size_percent", 15) / 100.0
        opacity = template.get("logo_opacity", 0.85)
        margin = template.get("logo_margin", 30)
        logo_w = int(w * size_pct)
        pos_map = {"top_left": (f"{margin}", f"{margin}"), "top_right": (f"W-w-{margin}", f"{margin}"),
                   "bottom_left": (f"{margin}", f"H-h-{margin}"), "bottom_right": (f"W-w-{margin}", f"H-h-{margin}")}
        x, y = pos_map.get(position, pos_map["top_right"])
        filter_parts.append(f"[{logo_idx}:v]scale={logo_w}:-1,format=rgba,colorchannelmixer=aa={opacity}[logo]")
        filter_parts.append(f"[{current_label}][logo]overlay={x}:{y}[v{label_idx}]")
        current_label = f"v{label_idx}"
        label_idx += 1
    
    if ass_path and os.path.exists(ass_path):
        ass_escaped = escape_ffmpeg_path(ass_path)
        sub_filter = f"subtitles=filename='{ass_escaped}'"
        
        if font_dir and os.path.isdir(font_dir):
            fontsdir_escaped = escape_ffmpeg_path(font_dir)
            sub_filter += f":fontsdir='{fontsdir_escaped}'"
        
        font_file = template.get("font_file", "")
        font_name = template.get("font_name", "Arial")
        
        if font_file and os.path.exists(font_file):
            real_name = get_font_name_from_file(font_file)
            if real_name:
                font_name = real_name
        
        safe_font_name = font_name.replace("'", "").replace(",", "").replace('"', '')
        font_size = template.get("font_size", 72)
        font_color = template.get("font_color", "#FFFFFF").lstrip("#")
        outline_color = template.get("outline_color", "#000000").lstrip("#")
        outline_width = template.get("outline_width", 3)
        
        if len(font_color) < 6: font_color = "FFFFFF"
        if len(outline_color) < 6: outline_color = "000000"
        
        primary_color = f"&H00{font_color[4:6]}{font_color[2:4]}{font_color[0:2]}&"
        out_color = f"&H00{outline_color[4:6]}{outline_color[2:4]}{outline_color[0:2]}&"
        
        force_style = f"FontName={safe_font_name},FontSize={font_size}"
        force_style += f",PrimaryColour={primary_color}"
        force_style += f",OutlineColour={out_color}"
        force_style += f",Outline={outline_width},Bold=-1,BorderStyle=1"
        
        sub_filter += f":force_style='{force_style}'"
        
        filter_parts.append(f"[{current_label}]{sub_filter}[v{label_idx}]")
        current_label = f"v{label_idx}"
        label_idx += 1
    
    filter_parts.append(f"[{current_label}]format=yuv420p[vout]")
    filter_complex = ";".join(filter_parts)
    
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[vout]", "-map", "1:a",
           *get_encoder_params(use_gpu, for_intermediate=False), "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-shortest"]
    if duration:
        cmd += ["-t", str(duration)]
    cmd.append(output_path)
    
    result = run_cmd(cmd)
    
    if not result or result.returncode != 0:
        simple_cmd = ["ffmpeg", "-y", "-i", visual_path, "-i", audio_path,
                      "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},format=yuv420p",
                      *get_encoder_params(use_gpu, for_intermediate=False), "-c:a", "aac", "-shortest"]
        if duration:
            simple_cmd += ["-t", str(duration)]
        simple_cmd.append(output_path)
        run_cmd(simple_cmd)
    
    return output_path

def process_folder(folder_path, output_folder, template, log_fn=None, progress_fn=None):
    """Process folder - FIX VIETNAMESE FONT"""

    folder_name = os.path.basename(folder_path)
    output_path = os.path.join(output_folder, f"{folder_name}.mp4")
    use_gpu = template.get("use_gpu", True) and GPU_INFO.get("nvenc", False)
    start_time = time.time()

    def log(msg):
        if log_fn:
            log_fn(f"[{folder_name}] {msg}")

    def report_progress(step_percent):
        """Report progress within this video (0-100)"""
        if progress_fn:
            progress_fn(step_percent)
    
    if os.path.exists(output_path):
        log("Đã tồn tại - Bỏ qua")
        return True, "skipped", output_path, 0
    
    audio_files = list_media_files(folder_path, (".mp3", ".wav", ".m4a", ".aac"))
    if not audio_files:
        log("LỖI: Không có audio")
        return False, "no_audio", None, 0
    
    audio_file = audio_files[0]
    audio_duration = get_audio_duration(audio_file)
    if not audio_duration:
        log("LỖI: Không đọc được audio")
        return False, "audio_error", None, 0
    
    video_files = list_media_files(folder_path, (".mp4", ".mov", ".mkv", ".avi"))
    image_files = list_media_files(folder_path, (".jpg", ".jpeg", ".png", ".webp"))
    
    if not video_files and not image_files:
        log("LỖI: Không có video/ảnh")
        return False, "no_visual", None, 0
    
    log(f"Bắt đầu: {audio_duration:.1f}s | {len(video_files)} video | {len(image_files)} ảnh")
    report_progress(5)  # Started

    tmp_dir = tempfile.mkdtemp(prefix="ve_")

    try:
        log("Đang tạo phụ đề...")
        report_progress(10)
        srt_path = generate_srt(audio_file, tmp_dir, template)
        report_progress(35)  # Subtitles done
        ass_path = os.path.join(tmp_dir, "subs.ass")
        
        font_tmp_dir = os.path.join(tmp_dir, "fonts")
        os.makedirs(font_tmp_dir, exist_ok=True)
        
        _, used_font_file = srt_to_ass(srt_path, ass_path, template, font_tmp_dir)
        
        font_file = template.get("font_file", "")
        font_name = template.get("font_name", "Arial")
        
        if font_file and os.path.exists(font_file):
            shutil.copy2(font_file, os.path.join(font_tmp_dir, os.path.basename(font_file)))
            real_name = get_font_name_from_file(font_file)
            if real_name:
                font_name = real_name
                template["font_name"] = real_name
            log(f"✓ Font: {font_name}")
        elif used_font_file and os.path.exists(used_font_file):
            shutil.copy2(used_font_file, os.path.join(font_tmp_dir, os.path.basename(used_font_file)))
            log(f"✓ Font (system): {font_name}")
        else:
            vn_font_path, vn_font_name = get_vietnamese_font()
            if vn_font_path:
                shutil.copy2(vn_font_path, os.path.join(font_tmp_dir, os.path.basename(vn_font_path)))
                template["font_name"] = vn_font_name
                template["font_file"] = vn_font_path
                log(f"✓ Font (VN): {vn_font_name}")
        
        if os.path.isdir(FONTS_DIR):
            for f in os.listdir(FONTS_DIR):
                if f.lower().endswith(('.ttf', '.otf')):
                    src = os.path.join(FONTS_DIR, f)
                    dst = os.path.join(font_tmp_dir, f)
                    try:
                        shutil.copy2(src, dst)
                    except:
                        pass
        
        log("Đang tạo clips...")
        report_progress(40)
        clips = []

        if video_files:
            for i, vf in enumerate(video_files):
                out = os.path.join(tmp_dir, f"clip_{len(clips):04d}.mp4")
                create_video_clip(vf, out, template, use_gpu=use_gpu)
                clips.append(out)
                clip_progress = 40 + int(15 * (i + 1) / len(video_files))
                report_progress(clip_progress)

        if image_files:
            remaining = audio_duration - sum(get_video_duration(c) for c in clips)
            if remaining > 0:
                dur_per_img = remaining / len(image_files)
                for i, img in enumerate(image_files):
                    out = os.path.join(tmp_dir, f"clip_{len(clips):04d}.mp4")
                    create_image_clip(img, out, dur_per_img, template, use_gpu=use_gpu)
                    clips.append(out)
                    clip_progress = 55 + int(10 * (i + 1) / len(image_files))
                    report_progress(clip_progress)

        log("Đang ghép clips...")
        report_progress(70)
        visual = os.path.join(tmp_dir, "visual.mp4")
        merge_clips(clips, visual, template, use_gpu)
        report_progress(75)

        log("Đang render...")
        report_progress(80)
        render_final(visual, audio_file, output_path, template, ass_path=ass_path, font_dir=font_tmp_dir, duration=audio_duration, use_gpu=use_gpu)
        report_progress(100)
        
        elapsed = time.time() - start_time
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log(f"✓ Hoàn thành: {size_mb:.1f} MB trong {format_time(elapsed)}")
        
        srt_out = os.path.join(output_folder, f"{folder_name}.srt")
        shutil.copy2(srt_path, srt_out)
        
        return True, "success", output_path, elapsed
        
    except Exception as e:
        log(f"✗ LỖI: {e}")
        return False, str(e), None, time.time() - start_time
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ===== MAIN APPLICATION =====
class UnixAutoEdit:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Uni-x Auto Edit v{APP_VERSION}")
        self.root.geometry("1400x900")
        self.root.minsize(1300, 800)
        self.root.configure(bg=COLORS["bg_main"])
        
        self.template_manager = TemplateManager()
        self.current_template = "Mặc định"
        self.config = self.load_config()
        
        self.processing = False
        self.stop_flag = False
        self.process_times = []
        self.folder_data = {}
        self.completed_outputs = {}  # Store output paths: {folder_name: output_path}

        self.preview_source_image = None
        
        detect_gpu()
        self.setup_styles()
        self.setup_ui()
        self.load_template_to_ui(self.template_manager.get(self.current_template))
        
        gpu_status = f"GPU: {GPU_INFO.get('name', 'N/A')[:20]}" if GPU_INFO.get('nvenc') else "CPU Mode"
        self.log(f"✓ Khởi động thành công | {gpu_status}")
        
        vn_font_path, vn_font_name = get_vietnamese_font()
        if vn_font_path:
            self.log(f"✓ Font tiếng Việt: {vn_font_name}")
    
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"input_folder": "", "output_folder": "", "max_workers": 2, "skip_existing": True, "use_gpu": True}
    
    def save_config(self):
        config = {"input_folder": self.input_var.get(), "output_folder": self.output_var.get(),
                  "max_workers": self.workers_var.get(), "skip_existing": self.skip_var.get(), "use_gpu": self.gpu_var.get()}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure(".", background=COLORS["bg_main"], foreground=COLORS["text_primary"])
        style.configure("TFrame", background=COLORS["bg_main"])
        style.configure("TLabel", background=COLORS["bg_main"], foreground=COLORS["text_primary"], font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=COLORS["accent_light"])
        
        style.configure("TNotebook", background=COLORS["bg_main"], borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab", background=COLORS["bg_secondary"], foreground=COLORS["text_secondary"],
                       padding=[25, 12], font=("Segoe UI", 11))
        style.map("TNotebook.Tab",
            background=[("selected", COLORS["accent"]), ("active", COLORS["bg_card"])],
            foreground=[("selected", "#FFFFFF"), ("active", COLORS["text_primary"])],
            padding=[("selected", [35, 18])],
            font=[("selected", ("Segoe UI", 12, "bold"))]
        )
        
        style.configure("TButton", background=COLORS["bg_card"], foreground=COLORS["text_primary"],
                       font=("Segoe UI", 10), padding=[15, 8], borderwidth=0)
        style.map("TButton", background=[("active", COLORS["accent"]), ("pressed", COLORS["accent_dark"])])
        
        style.configure("TCombobox", fieldbackground=COLORS["input_bg"], foreground=COLORS["input_text"],
                       background=COLORS["input_bg"], arrowcolor=COLORS["text_primary"], padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", COLORS["input_bg"])],
                 foreground=[("readonly", COLORS["input_text"])])
        
        style.configure("TCheckbutton", background=COLORS["bg_main"], foreground=COLORS["text_primary"], font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", COLORS["bg_main"])])
        
        style.configure("TRadiobutton", background=COLORS["bg_main"], foreground=COLORS["text_primary"], font=("Segoe UI", 10))
        
        style.configure("Treeview", background=COLORS["bg_secondary"], foreground=COLORS["text_primary"],
                       fieldbackground=COLORS["bg_secondary"], font=("Segoe UI", 10), rowheight=32, borderwidth=0)
        style.configure("Treeview.Heading", background=COLORS["bg_card"], foreground=COLORS["text_primary"],
                       font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Treeview", background=[("selected", COLORS["accent"])], foreground=[("selected", "#FFFFFF")])
        
        style.configure("TScale", background=COLORS["bg_main"], troughcolor=COLORS["bg_card"], sliderthickness=20)
        style.configure("TSpinbox", fieldbackground=COLORS["input_bg"], foreground=COLORS["input_text"],
                       background=COLORS["input_bg"], arrowcolor=COLORS["text_primary"])
    
    def setup_ui(self):
        header = tk.Frame(self.root, bg=COLORS["bg_secondary"], height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title_frame = tk.Frame(header, bg=COLORS["bg_secondary"])
        title_frame.pack(side=tk.LEFT, padx=25, pady=10)
        
        tk.Label(title_frame, text="UNI-X", font=("Segoe UI", 24, "bold"),
                bg=COLORS["bg_secondary"], fg=COLORS["accent"]).pack(side=tk.LEFT)
        tk.Label(title_frame, text=" AUTO EDIT", font=("Segoe UI", 24),
                bg=COLORS["bg_secondary"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        tk.Label(title_frame, text=f" v{APP_VERSION}", font=("Segoe UI", 12),
                bg=COLORS["bg_secondary"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=5)
        
        gpu_frame = tk.Frame(header, bg=COLORS["bg_secondary"])
        gpu_frame.pack(side=tk.RIGHT, padx=25)
        
        if GPU_INFO.get('nvenc'):
            gpu_text = f"⚡ {GPU_INFO.get('name', 'GPU')[:25]}"
            gpu_color = COLORS["success"]
        else:
            gpu_text = "💻 CPU Mode"
            gpu_color = COLORS["text_dim"]
        
        tk.Label(gpu_frame, text=gpu_text, font=("Segoe UI", 10),
                bg=COLORS["bg_secondary"], fg=gpu_color).pack()
        
        main = tk.Frame(self.root, bg=COLORS["bg_main"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 0))
        
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        tab1 = tk.Frame(self.notebook, bg=COLORS["bg_main"])
        self.notebook.add(tab1, text="   📁  DỰ ÁN   ")
        self.setup_project_tab(tab1)
        
        tab2 = tk.Frame(self.notebook, bg=COLORS["bg_main"])
        self.notebook.add(tab2, text="   ⚙️  CÀI ĐẶT   ")
        self.setup_settings_tab(tab2)
        
        bottom = tk.Frame(self.root, bg=COLORS["bg_main"])
        bottom.pack(fill=tk.X, padx=20, pady=10)
        self.setup_progress_panel(bottom)
    
    def setup_project_tab(self, parent):
        left = tk.Frame(parent, bg=COLORS["bg_main"], width=380)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15), pady=15)
        left.pack_propagate(False)
        
        folder_card = self.create_card(left, "📂 THƯ MỤC", 200)
        
        self.create_label(folder_card, "Thư mục nguồn:")
        in_row = tk.Frame(folder_card, bg=COLORS["bg_card"])
        in_row.pack(fill=tk.X, pady=(3, 12))
        
        self.input_var = tk.StringVar(value=self.config.get("input_folder", ""))
        self.create_entry(in_row, self.input_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.create_button(in_row, "📁", self.browse_input, width=3).pack(side=tk.RIGHT, padx=(8, 0))
        
        self.create_label(folder_card, "Thư mục xuất:")
        out_row = tk.Frame(folder_card, bg=COLORS["bg_card"])
        out_row.pack(fill=tk.X, pady=(3, 0))
        
        self.output_var = tk.StringVar(value=self.config.get("output_folder", ""))
        self.create_entry(out_row, self.output_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.create_button(out_row, "📁", self.browse_output, width=3).pack(side=tk.RIGHT, padx=(8, 0))
        
        opt_card = self.create_card(left, "⚡ TÙY CHỌN", 180)
        
        render_row = tk.Frame(opt_card, bg=COLORS["bg_card"])
        render_row.pack(fill=tk.X, pady=5)
        
        self.create_label(render_row, "Render:", pack=False).pack(side=tk.LEFT)
        self.gpu_var = tk.BooleanVar(value=self.config.get("use_gpu", True) and GPU_INFO.get("nvenc", False))
        
        gpu_state = tk.NORMAL if GPU_INFO.get("nvenc") else tk.DISABLED
        ttk.Radiobutton(render_row, text="🚀 GPU", variable=self.gpu_var, value=True, state=gpu_state).pack(side=tk.LEFT, padx=15)
        ttk.Radiobutton(render_row, text="💻 CPU", variable=self.gpu_var, value=False).pack(side=tk.LEFT)
        
        self.skip_var = tk.BooleanVar(value=self.config.get("skip_existing", True))
        ttk.Checkbutton(opt_card, text="Bỏ qua file đã có", variable=self.skip_var).pack(anchor=tk.W, pady=5)
        
        workers_row = tk.Frame(opt_card, bg=COLORS["bg_card"])
        workers_row.pack(fill=tk.X, pady=5)
        
        self.create_label(workers_row, "Xử lý song song:", pack=False).pack(side=tk.LEFT)
        self.workers_var = tk.IntVar(value=self.config.get("max_workers", 2))
        ttk.Spinbox(workers_row, from_=1, to=4, textvariable=self.workers_var, width=5).pack(side=tk.LEFT, padx=10)
        
        btn_frame = tk.Frame(left, bg=COLORS["bg_main"])
        btn_frame.pack(fill=tk.X, pady=15)

        self.start_btn = tk.Button(btn_frame, text="▶  BẮT ĐẦU XỬ LÝ", font=("Segoe UI", 13, "bold"),
                                  bg=COLORS["success"], fg="#FFFFFF", relief=tk.FLAT, cursor="hand2",
                                  activebackground="#16a34a", command=self.start_process)
        self.start_btn.pack(fill=tk.X, ipady=12)

        self.stop_btn = tk.Button(btn_frame, text="⏹  DỪNG LẠI", font=("Segoe UI", 11, "bold"),
                                 bg=COLORS["error"], fg="#FFFFFF", relief=tk.FLAT, cursor="hand2",
                                 activebackground="#dc2626", command=self.stop_process, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, ipady=8, pady=(10, 0))

        # ===== RIGHT PANEL - FILE LIST =====
        right = tk.Frame(parent, bg=COLORS["bg_main"])
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, pady=15)

        # Header with title and stats
        list_header = tk.Frame(right, bg=COLORS["bg_main"])
        list_header.pack(fill=tk.X, pady=(0, 8))

        tk.Label(list_header, text="📋 DANH SÁCH FILE", font=("Segoe UI", 12, "bold"),
                bg=COLORS["bg_main"], fg=COLORS["accent_light"]).pack(side=tk.LEFT)

        self.stats_label = tk.Label(list_header, text="Tổng: 0 | Chọn: 0 | Xong: 0",
                                   bg=COLORS["bg_main"], fg=COLORS["text_dim"], font=("Segoe UI", 10))
        self.stats_label.pack(side=tk.RIGHT)

        # Control panel - MOVED TO TOP (above file list)
        control_panel = tk.Frame(right, bg=COLORS["bg_card"], padx=12, pady=10)
        control_panel.pack(fill=tk.X, pady=(0, 10))

        btn_style = {"bg": COLORS["bg_hover"], "fg": COLORS["text_primary"], "relief": tk.FLAT,
                    "font": ("Segoe UI", 9), "cursor": "hand2", "padx": 10, "pady": 4}

        # Row 1: Selection buttons
        row1 = tk.Frame(control_panel, bg=COLORS["bg_card"])
        row1.pack(fill=tk.X, pady=(0, 6))

        tk.Button(row1, text="✓ Chọn tất cả", command=self.select_all, **btn_style).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(row1, text="✗ Bỏ chọn", command=self.deselect_all, **btn_style).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(row1, text="☑ Bật xử lý", command=self.enable_selected, **btn_style).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(row1, text="☐ Tắt xử lý", command=self.disable_selected, **btn_style).pack(side=tk.LEFT)

        # Selection info on right
        self.selection_label = tk.Label(row1, text="",
                                        bg=COLORS["bg_card"], fg=COLORS["text_secondary"], font=("Segoe UI", 9))
        self.selection_label.pack(side=tk.RIGHT)

        # Row 2: Template selection for selected items
        row2 = tk.Frame(control_panel, bg=COLORS["bg_card"])
        row2.pack(fill=tk.X)

        tk.Label(row2, text="Áp dụng template cho mục đã chọn:", bg=COLORS["bg_card"],
                fg=COLORS["text_secondary"], font=("Segoe UI", 9)).pack(side=tk.LEFT)

        self.quick_template_var = tk.StringVar(value="Mặc định")
        self.quick_template_combo = ttk.Combobox(row2, textvariable=self.quick_template_var,
                                                 values=self.template_manager.get_list(), width=15, state="readonly")
        self.quick_template_combo.pack(side=tk.LEFT, padx=(8, 0))
        self.quick_template_combo.bind("<<ComboboxSelected>>", self.apply_template_to_selected)

        # File list (Treeview)
        tree_frame = tk.Frame(right, bg=COLORS["bg_secondary"])
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("name", "template", "status", "open", "audio", "video", "img")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=12, selectmode="extended")

        self.tree.heading("name", text="Tên thư mục")
        self.tree.heading("template", text="Template")
        self.tree.heading("status", text="Trạng thái")
        self.tree.heading("open", text="Mở")
        self.tree.heading("audio", text="🎵")
        self.tree.heading("video", text="🎬")
        self.tree.heading("img", text="🖼")

        self.tree.column("name", width=180)
        self.tree.column("template", width=100)
        self.tree.column("status", width=80)
        self.tree.column("open", width=50, anchor=tk.CENTER)
        self.tree.column("audio", width=40, anchor=tk.CENTER)
        self.tree.column("video", width=40, anchor=tk.CENTER)
        self.tree.column("img", width=40, anchor=tk.CENTER)

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind events
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.on_tree_double_click)  # Double-click to change template
        self.tree.bind("<Button-1>", self.on_tree_click)  # Single click to open file

        # Start auto-refresh for file list
        self._last_folder_mtime = 0
        self._auto_refresh_enabled = True
        self.root.after(2000, self._auto_refresh_list)
    
    def setup_settings_tab(self, parent):
        container = tk.Frame(parent, bg=COLORS["bg_main"])
        container.pack(fill=tk.BOTH, expand=True, pady=15)
        
        left = tk.Frame(container, bg=COLORS["bg_main"], width=500)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left.pack_propagate(False)
        
        canvas = tk.Canvas(left, bg=COLORS["bg_main"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=COLORS["bg_main"])
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=480)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Template
        tpl_card = self.create_card(scroll_frame, "💾 TEMPLATE", 100, width=470)
        tpl_row = tk.Frame(tpl_card, bg=COLORS["bg_card"])
        tpl_row.pack(fill=tk.X)
        
        self.template_var = tk.StringVar(value=self.current_template)
        self.template_combo = ttk.Combobox(tpl_row, textvariable=self.template_var,
                                          values=self.template_manager.get_list(), width=20, state="readonly")
        self.template_combo.pack(side=tk.LEFT)
        self.template_combo.bind("<<ComboboxSelected>>", self.on_template_change)
        
        for text, cmd in [("💾 Lưu", self.save_template), ("➕ Mới", self.new_template), ("🗑 Xóa", self.delete_template)]:
            self.create_button(tpl_row, text, cmd, width=8).pack(side=tk.LEFT, padx=(10, 0))
        
        # Resolution
        res_card = self.create_card(scroll_frame, "📐 ĐỘ PHÂN GIẢI", 120, width=470)
        res_row = tk.Frame(res_card, bg=COLORS["bg_card"])
        res_row.pack(fill=tk.X, pady=5)
        
        self.create_label(res_row, "Rộng:", pack=False).pack(side=tk.LEFT)
        self.width_var = tk.IntVar(value=1920)
        self.width_var.trace_add("write", self.update_preview)
        self.create_entry(res_row, self.width_var, width=8).pack(side=tk.LEFT, padx=(5, 20))

        self.create_label(res_row, "Cao:", pack=False).pack(side=tk.LEFT)
        self.height_var = tk.IntVar(value=1080)
        self.height_var.trace_add("write", self.update_preview)
        self.create_entry(res_row, self.height_var, width=8).pack(side=tk.LEFT, padx=5)
        
        preset_row = tk.Frame(res_card, bg=COLORS["bg_card"])
        preset_row.pack(fill=tk.X, pady=(8, 0))
        
        for name, w, h in [("1080p", 1920, 1080), ("720p", 1280, 720), ("4K", 3840, 2160), ("9:16", 1080, 1920)]:
            self.create_button(preset_row, name, lambda w=w, h=h: self.set_resolution(w, h), width=7).pack(side=tk.LEFT, padx=3)
        
        # Subtitle
        sub_card = self.create_card(scroll_frame, "📝 PHỤ ĐỀ", 200, width=470)
        
        font_row = tk.Frame(sub_card, bg=COLORS["bg_card"])
        font_row.pack(fill=tk.X, pady=3)
        
        self.create_label(font_row, "Font:", width=10, pack=False).pack(side=tk.LEFT)
        self.font_var = tk.StringVar(value="Arial")
        self.font_file_var = tk.StringVar()
        font_combo = ttk.Combobox(font_row, textvariable=self.font_var, values=self.get_font_list(), width=15, state="readonly")
        font_combo.pack(side=tk.LEFT, padx=5)
        font_combo.bind("<<ComboboxSelected>>", self.on_font_change)
        self.create_button(font_row, "📂 Chọn", self.browse_font, width=8).pack(side=tk.LEFT, padx=5)
        
        size_row = tk.Frame(sub_card, bg=COLORS["bg_card"])
        size_row.pack(fill=tk.X, pady=3)
        
        self.create_label(size_row, "Cỡ chữ:", width=10, pack=False).pack(side=tk.LEFT)
        self.fontsize_var = tk.IntVar(value=72)
        self.create_entry(size_row, self.fontsize_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Scale(size_row, from_=24, to=150, variable=self.fontsize_var, orient=tk.HORIZONTAL, length=180,
                 command=lambda v: self.update_preview()).pack(side=tk.LEFT, padx=10)
        
        color_row = tk.Frame(sub_card, bg=COLORS["bg_card"])
        color_row.pack(fill=tk.X, pady=3)
        
        self.create_label(color_row, "Màu chữ:", width=10, pack=False).pack(side=tk.LEFT)
        self.fontcolor_var = tk.StringVar(value="#FFFFFF")
        self.fontcolor_btn = tk.Button(color_row, width=4, bg="#FFFFFF", relief=tk.FLAT, cursor="hand2",
                                       command=lambda: self.pick_color(self.fontcolor_var, self.fontcolor_btn))
        self.fontcolor_btn.pack(side=tk.LEFT, padx=5)
        
        self.create_label(color_row, "Viền:", pack=False).pack(side=tk.LEFT, padx=(30, 0))
        self.outlinecolor_var = tk.StringVar(value="#000000")
        self.outlinecolor_btn = tk.Button(color_row, width=4, bg="#000000", relief=tk.FLAT, cursor="hand2",
                                          command=lambda: self.pick_color(self.outlinecolor_var, self.outlinecolor_btn))
        self.outlinecolor_btn.pack(side=tk.LEFT, padx=5)
        
        outline_row = tk.Frame(sub_card, bg=COLORS["bg_card"])
        outline_row.pack(fill=tk.X, pady=3)
        
        self.create_label(outline_row, "Độ dày viền:", width=10, pack=False).pack(side=tk.LEFT)
        self.outline_var = tk.IntVar(value=3)
        self.outline_var.trace_add("write", self.update_preview)
        self.create_entry(outline_row, self.outline_var, width=6).pack(side=tk.LEFT, padx=5)

        margin_row = tk.Frame(sub_card, bg=COLORS["bg_card"])
        margin_row.pack(fill=tk.X, pady=3)

        self.create_label(margin_row, "Cách đáy:", width=10, pack=False).pack(side=tk.LEFT)
        self.margin_var = tk.IntVar(value=50)
        self.margin_var.trace_add("write", self.update_preview)
        self.create_entry(margin_row, self.margin_var, width=6).pack(side=tk.LEFT, padx=5)
        
        # Logo
        logo_card = self.create_card(scroll_frame, "🏷 LOGO", 140, width=470)
        
        self.logo_enabled_var = tk.BooleanVar(value=False)
        self.logo_enabled_var.trace_add("write", self.update_preview)
        ttk.Checkbutton(logo_card, text="Thêm logo vào video", variable=self.logo_enabled_var).pack(anchor=tk.W, pady=3)
        
        logo_file_row = tk.Frame(logo_card, bg=COLORS["bg_card"])
        logo_file_row.pack(fill=tk.X, pady=3)
        
        self.create_label(logo_file_row, "File:", width=8, pack=False).pack(side=tk.LEFT)
        self.logo_var = tk.StringVar()
        self.logo_var.trace_add("write", self.update_preview)
        self.logo_combo = ttk.Combobox(logo_file_row, textvariable=self.logo_var, values=self.get_logo_list(), width=18, state="readonly")
        self.logo_combo.pack(side=tk.LEFT, padx=5)
        self.create_button(logo_file_row, "📂", self.browse_logo, width=3).pack(side=tk.LEFT, padx=5)
        
        logo_pos_row = tk.Frame(logo_card, bg=COLORS["bg_card"])
        logo_pos_row.pack(fill=tk.X, pady=3)
        
        self.create_label(logo_pos_row, "Vị trí:", width=8, pack=False).pack(side=tk.LEFT)
        self.logopos_var = tk.StringVar(value="top_right")
        self.logopos_var.trace_add("write", self.update_preview)
        for pos, text in [("top_left", "⬆️ Trái"), ("top_right", "⬆️ Phải"), ("bottom_left", "⬇️ Trái"), ("bottom_right", "⬇️ Phải")]:
            ttk.Radiobutton(logo_pos_row, text=text, value=pos, variable=self.logopos_var).pack(side=tk.LEFT, padx=5)
        
        logo_size_row = tk.Frame(logo_card, bg=COLORS["bg_card"])
        logo_size_row.pack(fill=tk.X, pady=3)
        
        self.create_label(logo_size_row, "Kích cỡ:", width=8, pack=False).pack(side=tk.LEFT)
        self.logosize_var = tk.IntVar(value=15)
        self.logosize_var.trace_add("write", self.update_preview)
        self.create_entry(logo_size_row, self.logosize_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(logo_size_row, text="%", bg=COLORS["bg_card"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        
        # Effects
        effect_card = self.create_card(scroll_frame, "✨ HIỆU ỨNG", 180, width=470)
        
        filter_row = tk.Frame(effect_card, bg=COLORS["bg_card"])
        filter_row.pack(fill=tk.X, pady=3)
        
        self.create_label(filter_row, "Bộ lọc:", width=10, pack=False).pack(side=tk.LEFT)
        self.filter_var = tk.StringVar(value="none")
        filter_combo = ttk.Combobox(filter_row, textvariable=self.filter_var, width=25, state="readonly")
        filter_combo['values'] = [f"{k}: {v['name']}" for k, v in FILTER_EFFECTS.items()]
        filter_combo.current(0)
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self.filter_var.set(self.filter_var.get().split(":")[0]))
        
        overlay_row = tk.Frame(effect_card, bg=COLORS["bg_card"])
        overlay_row.pack(fill=tk.X, pady=3)
        
        self.create_label(overlay_row, "Overlay:", width=10, pack=False).pack(side=tk.LEFT)
        self.overlay_var = tk.StringVar(value="")
        overlay_combo = ttk.Combobox(overlay_row, textvariable=self.overlay_var, width=18, state="readonly")
        overlay_combo['values'] = [f"{k}: {v['name']}" if k else "Không có" for k, v in VIDEO_OVERLAY_EFFECTS.items()]
        overlay_combo.current(0)
        overlay_combo.pack(side=tk.LEFT, padx=5)
        overlay_combo.bind("<<ComboboxSelected>>", lambda e: self.overlay_var.set(self.overlay_var.get().split(":")[0] if ":" in self.overlay_var.get() else ""))
        
        self.create_button(overlay_row, "🎬 Tạo", self.create_overlays, width=8).pack(side=tk.LEFT, padx=10)
        
        opacity_row = tk.Frame(effect_card, bg=COLORS["bg_card"])
        opacity_row.pack(fill=tk.X, pady=3)
        
        self.create_label(opacity_row, "Độ mờ:", width=10, pack=False).pack(side=tk.LEFT)
        self.overlay_opacity_var = tk.DoubleVar(value=0.5)
        self.create_entry(opacity_row, self.overlay_opacity_var, width=5).pack(side=tk.LEFT, padx=5)
        
        trans_row = tk.Frame(effect_card, bg=COLORS["bg_card"])
        trans_row.pack(fill=tk.X, pady=3)

        self.trans_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(trans_row, text="Chuyển cảnh", variable=self.trans_var).pack(side=tk.LEFT)

        self.create_label(trans_row, "Kiểu:", pack=False).pack(side=tk.LEFT, padx=(15, 0))
        self.transtype_var = tk.StringVar(value="random")
        ttk.Combobox(trans_row, textvariable=self.transtype_var, width=10, state="readonly",
                    values=["random", "fade", "fadeblack", "fadewhite", "wipeleft", "wiperight",
                            "wipeup", "wipedown", "slideleft", "slideright", "slideup", "slidedown",
                            "circlecrop", "rectcrop", "distance", "pixelize"]).pack(side=tk.LEFT, padx=5)

        self.create_label(trans_row, "Thời gian:", pack=False).pack(side=tk.LEFT, padx=(10, 0))
        self.transdur_var = tk.DoubleVar(value=0.8)
        self.create_entry(trans_row, self.transdur_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(trans_row, text="giây", bg=COLORS["bg_card"], fg=COLORS["text_primary"]).pack(side=tk.LEFT)
        
        # Whisper
        whisper_card = self.create_card(scroll_frame, "🎤 WHISPER AI", 80, width=470)
        whisper_row = tk.Frame(whisper_card, bg=COLORS["bg_card"])
        whisper_row.pack(fill=tk.X)
        
        self.create_label(whisper_row, "Model:", pack=False).pack(side=tk.LEFT)
        self.whisper_var = tk.StringVar(value="tiny")
        ttk.Combobox(whisper_row, textvariable=self.whisper_var, width=10, state="readonly",
                    values=["tiny", "base", "small", "medium", "large"]).pack(side=tk.LEFT, padx=5)
        
        self.create_label(whisper_row, "Ngôn ngữ:", pack=False).pack(side=tk.LEFT, padx=(25, 0))
        self.lang_var = tk.StringVar(value="en")
        ttk.Combobox(whisper_row, textvariable=self.lang_var, width=6, state="readonly",
                    values=["vi", "en", "zh", "ja", "ko", "fr", ""]).pack(side=tk.LEFT, padx=5)
        
        # Preview
        right = tk.Frame(container, bg=COLORS["bg_main"])
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        preview_card = tk.Frame(right, bg=COLORS["bg_card"], padx=15, pady=15)
        preview_card.pack(fill=tk.BOTH, expand=True)
        
        preview_header = tk.Frame(preview_card, bg=COLORS["bg_card"])
        preview_header.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(preview_header, text="👁 XEM TRƯỚC", font=("Segoe UI", 12, "bold"),
                bg=COLORS["bg_card"], fg=COLORS["accent_light"]).pack(side=tk.LEFT)
        
        self.create_button(preview_header, "🔄", self.update_preview, width=3).pack(side=tk.RIGHT)
        self.create_button(preview_header, "📂 Chọn ảnh", self.load_preview_image, width=10).pack(side=tk.RIGHT, padx=10)
        
        canvas_container = tk.Frame(preview_card, bg=COLORS["border"], padx=3, pady=3)
        canvas_container.pack(expand=True)
        
        self.preview_canvas = tk.Canvas(canvas_container, bg="#000000", width=640, height=360, highlightthickness=0)
        self.preview_canvas.pack()
        
        self.preview_info = tk.Label(preview_card, text="Chọn ảnh để xem trước",
                                    bg=COLORS["bg_card"], fg=COLORS["text_dim"], font=("Segoe UI", 10))
        self.preview_info.pack(pady=10)
    
    def setup_progress_panel(self, parent):
        info_row = tk.Frame(parent, bg=COLORS["bg_main"])
        info_row.pack(fill=tk.X, pady=(0, 8))
        
        self.progress_label = tk.Label(info_row, text="Sẵn sàng", bg=COLORS["bg_main"],
                                      fg=COLORS["text_primary"], font=("Segoe UI", 11, "bold"))
        self.progress_label.pack(side=tk.LEFT)
        
        self.eta_label = tk.Label(info_row, text="", bg=COLORS["bg_main"],
                                 fg=COLORS["accent_light"], font=("Segoe UI", 10))
        self.eta_label.pack(side=tk.RIGHT)
        
        self.percent_label = tk.Label(info_row, text="0%", bg=COLORS["bg_main"],
                                     fg=COLORS["success"], font=("Segoe UI", 14, "bold"))
        self.percent_label.pack(side=tk.RIGHT, padx=25)
        
        progress_bg = tk.Frame(parent, bg=COLORS["bg_card"], height=25)
        progress_bg.pack(fill=tk.X, pady=(0, 8))
        progress_bg.pack_propagate(False)
        
        self.progress_fill = tk.Frame(progress_bg, bg=COLORS["accent"], width=0)
        self.progress_fill.pack(side=tk.LEFT, fill=tk.Y)
        
        log_frame = tk.Frame(parent, bg=COLORS["bg_secondary"])
        log_frame.pack(fill=tk.X)
        
        self.log_text = tk.Text(log_frame, height=4, bg=COLORS["bg_secondary"], fg=COLORS["text_primary"],
                               font=("Consolas", 10), relief=tk.FLAT, wrap=tk.WORD,
                               insertbackground=COLORS["text_primary"], padx=10, pady=8)
        self.log_text.pack(fill=tk.X)
    
    # ===== UI HELPERS =====
    def create_card(self, parent, title, height, width=None):
        card = tk.Frame(parent, bg=COLORS["bg_card"], padx=15, pady=12)
        if width:
            card.configure(width=width)
        card.pack(fill=tk.X, pady=(0, 12))
        tk.Label(card, text=title, font=("Segoe UI", 11, "bold"),
                bg=COLORS["bg_card"], fg=COLORS["accent_light"]).pack(anchor=tk.W, pady=(0, 8))
        return card
    
    def create_label(self, parent, text, width=None, pack=True):
        label = tk.Label(parent, text=text, bg=COLORS["bg_card"], fg=COLORS["text_primary"], font=("Segoe UI", 10))
        if width:
            label.configure(width=width, anchor=tk.W)
        if pack:
            label.pack(anchor=tk.W)
        return label
    
    def create_entry(self, parent, textvariable, width=None):
        entry = tk.Entry(parent, textvariable=textvariable, bg=COLORS["input_bg"], fg=COLORS["input_text"],
                        font=("Segoe UI", 10), relief=tk.FLAT, insertbackground=COLORS["input_text"],
                        highlightthickness=1, highlightbackground=COLORS["input_border"], highlightcolor=COLORS["accent"])
        if width:
            entry.configure(width=width)
        return entry
    
    def create_button(self, parent, text, command, width=None):
        btn = tk.Button(parent, text=text, command=command, bg=COLORS["bg_card"], fg=COLORS["text_primary"],
                       font=("Segoe UI", 10), relief=tk.FLAT, cursor="hand2",
                       activebackground=COLORS["accent"], activeforeground="#FFFFFF")
        if width:
            btn.configure(width=width)
        return btn
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_text.see(tk.END)
    
    def update_progress(self, percent, text="", eta_seconds=-1):
        self.percent_label.config(text=f"{percent:.1f}%")
        try:
            max_width = self.root.winfo_width() - 40
            fill_width = int(max_width * percent / 100)
            self.progress_fill.config(width=max(0, fill_width))
        except:
            pass
        if text:
            self.progress_label.config(text=text)
        if eta_seconds >= 0:
            # Calculate estimated finish time
            finish_time = datetime.now() + timedelta(seconds=eta_seconds)
            finish_str = finish_time.strftime("%H:%M")
            self.eta_label.config(text=f"⏱ Còn: {format_time(eta_seconds)} | Xong lúc: {finish_str}")
        else:
            self.eta_label.config(text="")
    
    def pick_color(self, var, btn):
        color = colorchooser.askcolor(color=var.get())[1]
        if color:
            var.set(color)
            btn.configure(bg=color)
            self.update_preview()
    
    def set_resolution(self, w, h):
        self.width_var.set(w)
        self.height_var.set(h)
        self.update_preview()
    
    def browse_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_var.set(folder)
            self.refresh_list()
    
    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)
            self.refresh_list()
    
    def browse_font(self):
        file = filedialog.askopenfilename(initialdir=FONTS_DIR, filetypes=[("Fonts", "*.ttf *.otf")])
        if file:
            dest = os.path.join(FONTS_DIR, os.path.basename(file))
            if file != dest:
                shutil.copy2(file, dest)
            font_name = get_font_name_from_file(dest) or os.path.splitext(os.path.basename(file))[0]
            self.font_var.set(font_name)
            self.font_file_var.set(dest)
            self.log(f"📝 Font đã chọn: {font_name}")
    
    def browse_logo(self):
        file = filedialog.askopenfilename(initialdir=LOGOS_DIR, filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if file:
            dest = os.path.join(LOGOS_DIR, os.path.basename(file))
            if file != dest:
                shutil.copy2(file, dest)
            self.logo_var.set(os.path.basename(file))
            self.logo_combo['values'] = self.get_logo_list()
    
    def get_font_list(self):
        fonts = list(VIETNAMESE_FONTS)
        if os.path.isdir(FONTS_DIR):
            for f in os.listdir(FONTS_DIR):
                if f.lower().endswith(('.ttf', '.otf')):
                    name = get_font_name_from_file(os.path.join(FONTS_DIR, f))
                    if name and name not in fonts:
                        fonts.append(name)
        return fonts
    
    def get_logo_list(self):
        return [f for f in os.listdir(LOGOS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))] if os.path.isdir(LOGOS_DIR) else []
    
    def on_font_change(self, event=None):
        font_name = self.font_var.get()
        if os.path.isdir(FONTS_DIR):
            for f in os.listdir(FONTS_DIR):
                if f.lower().endswith(('.ttf', '.otf')):
                    path = os.path.join(FONTS_DIR, f)
                    if get_font_name_from_file(path) == font_name:
                        self.font_file_var.set(path)
                        self.update_preview()
                        return
        sys_path = find_system_font(font_name)
        if sys_path:
            self.font_file_var.set(sys_path)
        else:
            self.font_file_var.set("")
        self.update_preview()

    def create_overlays(self):
        self.log("Đang tạo video overlay...")
        created = 0
        for key in VIDEO_OVERLAY_EFFECTS:
            if not key:
                continue
            output_path = os.path.join(EFFECTS_DIR, f"{key}.mp4")
            if not os.path.exists(output_path):
                self.log(f"  Tạo {key}...")
                if create_overlay_video(key, output_path):
                    created += 1
        self.log(f"✓ Đã tạo {created} overlay mới")
        messagebox.showinfo("Hoàn thành", f"Đã tạo {created} video overlay")
    
    def on_tree_select(self, event=None):
        selection = self.tree.selection()
        count = len(selection)
        if count == 0:
            self.selection_label.config(text="")
        elif count == 1:
            item = self.tree.item(selection[0])
            folder_name = item['values'][0]
            tpl = item['values'][1]
            self.selection_label.config(text=f"Đã chọn: {folder_name}")
            self.quick_template_var.set(tpl)
        else:
            self.selection_label.config(text=f"Đã chọn: {count} mục")

    def on_tree_click(self, event=None):
        """Single click - check if clicked on 'open' column to open file"""
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if not item:
            return

        # Column #4 is "open" (index 3)
        col_idx = int(column.replace("#", "")) - 1
        if col_idx == 3:  # "open" column
            values = self.tree.item(item, 'values')
            if len(values) > 3 and values[3] == "📂":
                folder_name = values[0]
                self._open_output_file(folder_name)

    def _open_output_file(self, folder_name):
        """Open the output file for the given folder"""
        if folder_name in self.completed_outputs:
            output_path = self.completed_outputs[folder_name]
        else:
            output_folder = self.output_var.get()
            output_path = os.path.join(output_folder, f"{folder_name}.mp4")

        if os.path.exists(output_path):
            import subprocess
            if sys.platform == 'win32':
                os.startfile(output_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', output_path])
            else:
                subprocess.run(['xdg-open', output_path])
            self.log(f"📂 Mở file: {output_path}")
        else:
            messagebox.showinfo("Thông báo", f"File không tồn tại:\n{output_path}")

    def on_tree_double_click(self, event=None):
        """Double-click on a row to change its template via popup menu"""
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if not item:
            return

        # Get column index (e.g., "#2" -> 1)
        col_idx = int(column.replace("#", "")) - 1

        # If double-click on template column (index 1), show template selector
        if col_idx == 1:
            self._show_template_popup(item, event)

    def _show_template_popup(self, item, event):
        """Show a popup menu to select template for a specific item"""
        popup = tk.Menu(self.root, tearoff=0, bg=COLORS["bg_card"], fg=COLORS["text_primary"])

        templates = self.template_manager.get_list()
        input_folder = self.input_var.get()
        values = list(self.tree.item(item, 'values'))
        folder_name = values[0]
        folder_path = os.path.join(input_folder, folder_name)

        def set_template(tpl_name):
            if folder_path in self.folder_data:
                self.folder_data[folder_path]["template"] = tpl_name
                values[1] = tpl_name
                self.tree.item(item, values=values)
                self.log(f"📋 Template '{tpl_name}' cho '{folder_name}'")

        for tpl in templates:
            popup.add_command(label=tpl, command=lambda t=tpl: set_template(t))

        try:
            popup.tk_popup(event.x_root, event.y_root)
        finally:
            popup.grab_release()

    def _auto_refresh_list(self):
        """Auto-refresh file list when input folder changes"""
        if not self._auto_refresh_enabled:
            return

        try:
            input_folder = self.input_var.get()
            if input_folder and os.path.isdir(input_folder):
                # Check if folder was modified
                current_mtime = os.path.getmtime(input_folder)

                # Also check subfolder count
                try:
                    current_items = set(os.listdir(input_folder))
                except:
                    current_items = set()

                # Store previous state
                if not hasattr(self, '_last_folder_items'):
                    self._last_folder_items = set()

                # Refresh if mtime changed or items changed
                if (current_mtime != self._last_folder_mtime or
                    current_items != self._last_folder_items):

                    self._last_folder_mtime = current_mtime
                    self._last_folder_items = current_items

                    # Only refresh if not processing
                    if not self.processing:
                        self.refresh_list()
        except Exception as e:
            pass  # Silently ignore errors

        # Schedule next check (every 3 seconds)
        self.root.after(3000, self._auto_refresh_list)

    def apply_template_to_selected(self, event=None):
        selection = self.tree.selection()
        if not selection:
            return
        template_name = self.quick_template_var.get()
        input_folder = self.input_var.get()
        for item in selection:
            values = list(self.tree.item(item, 'values'))
            folder_name = values[0]
            folder_path = os.path.join(input_folder, folder_name)
            if folder_path in self.folder_data:
                self.folder_data[folder_path]["template"] = template_name
                values[1] = template_name
                self.tree.item(item, values=values)
        self.log(f"📋 Đã áp dụng template '{template_name}' cho {len(selection)} items")
    
    def enable_selected(self):
        selection = self.tree.selection()
        if not selection:
            selection = self.tree.get_children()
        input_folder = self.input_var.get()
        count = 0
        for item in selection:
            values = list(self.tree.item(item, 'values'))
            folder_name = values[0]
            folder_path = os.path.join(input_folder, folder_name)
            if folder_path in self.folder_data:
                self.folder_data[folder_path]["enabled"] = True
                values[2] = "✓ Chờ xử lý"
                self.tree.item(item, values=values, tags=("enabled",))
                count += 1
        self.tree.tag_configure("enabled", foreground=COLORS["success"])
        self.update_stats()
        self.log(f"✓ Đã bật {count} items")
    
    def disable_selected(self):
        selection = self.tree.selection()
        if not selection:
            return
        input_folder = self.input_var.get()
        count = 0
        for item in selection:
            values = list(self.tree.item(item, 'values'))
            folder_name = values[0]
            folder_path = os.path.join(input_folder, folder_name)
            if folder_path in self.folder_data:
                self.folder_data[folder_path]["enabled"] = False
                values[2] = "○ Bỏ qua"
                self.tree.item(item, values=values, tags=("disabled",))
                count += 1
        self.tree.tag_configure("disabled", foreground=COLORS["text_dim"])
        self.update_stats()
        self.log(f"○ Đã tắt {count} items")
    
    def select_all(self):
        items = self.tree.get_children()
        self.tree.selection_set(items)
        self.on_tree_select()
    
    def deselect_all(self):
        self.tree.selection_remove(self.tree.selection())
        self.on_tree_select()
    
    def update_stats(self):
        total = len(self.folder_data)
        selected = sum(1 for d in self.folder_data.values() if d.get("enabled", True))
        done = sum(1 for item in self.tree.get_children() if "Xong" in str(self.tree.item(item, 'values')[2]))
        self.stats_label.config(text=f"Tổng: {total} | Bật: {selected} | Xong: {done}")
    
    def refresh_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.folder_data = {}
        input_folder = self.input_var.get()
        output_folder = self.output_var.get()
        if not input_folder or not os.path.isdir(input_folder):
            return
        if hasattr(self, 'quick_template_combo'):
            self.quick_template_combo['values'] = self.template_manager.get_list()
        for name in sorted(os.listdir(input_folder)):
            path = os.path.join(input_folder, name)
            if not os.path.isdir(path):
                continue
            audio = len(list_media_files(path, (".mp3", ".wav", ".m4a")))
            video = len(list_media_files(path, (".mp4", ".mov", ".mkv")))
            img = len(list_media_files(path, (".jpg", ".jpeg", ".png", ".webp")))
            out_path = os.path.join(output_folder, f"{name}.mp4") if output_folder else ""
            open_btn = ""  # Empty by default
            if out_path and os.path.exists(out_path):
                status = "✓ Xong"
                tag = "done"
                enabled = False
                open_btn = "📂"  # Show open icon
                self.completed_outputs[name] = out_path  # Store for quick access
            elif audio and (video or img):
                status = "✓ Chờ xử lý"
                tag = "enabled"
                enabled = True
            else:
                status = "⚠ Thiếu file"
                tag = "error"
                enabled = False
            default_tpl = self.template_var.get()
            self.folder_data[path] = {"enabled": enabled, "template": default_tpl}
            self.tree.insert("", tk.END, values=(name, default_tpl, status, open_btn, audio, video, img), tags=(tag,))
        self.tree.tag_configure("done", foreground=COLORS["success"])
        self.tree.tag_configure("enabled", foreground=COLORS["warning"])
        self.tree.tag_configure("error", foreground=COLORS["error"])
        self.tree.tag_configure("disabled", foreground=COLORS["text_dim"])
        self.update_stats()
    
    def load_preview_image(self):
        file = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")])
        if file:
            self.load_image_to_preview(file)
    
    def load_image_to_preview(self, image_path):
        if not PIL_AVAILABLE:
            return
        try:
            self.preview_source_image = Image.open(image_path)
            self.update_preview()
        except Exception as e:
            self.log(f"Lỗi tải ảnh: {e}")
    
    def update_preview(self, *args):
        if not PIL_AVAILABLE:
            return
        if self.preview_source_image is None:
            img = Image.new('RGB', (1920, 1080), color=(20, 20, 35))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except:
                font = ImageFont.load_default()
            draw.text((960, 540), "Chọn ảnh để xem trước", fill=(80, 80, 100), anchor="mm", font=font)
        else:
            img = self.preview_source_image.copy()
        
        target_w = self.width_var.get()
        target_h = self.height_var.get()
        img_ratio = img.width / img.height
        target_ratio = target_w / target_h
        if img_ratio > target_ratio:
            new_h = target_h
            new_w = int(img.width * target_h / img.height)
        else:
            new_w = target_w
            new_h = int(img.height * target_w / img.width)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
        
        if self.logo_enabled_var.get() and self.logo_var.get():
            logo_path = os.path.join(LOGOS_DIR, self.logo_var.get())
            if os.path.exists(logo_path):
                try:
                    logo_img = Image.open(logo_path).convert("RGBA")
                    logo_size = int(target_w * self.logosize_var.get() / 100)
                    logo_h = int(logo_img.height * logo_size / logo_img.width)
                    logo_img = logo_img.resize((logo_size, logo_h), Image.Resampling.LANCZOS)
                    pos = self.logopos_var.get()
                    margin = 30
                    positions = {"top_left": (margin, margin), "top_right": (target_w - logo_size - margin, margin),
                                "bottom_left": (margin, target_h - logo_h - margin),
                                "bottom_right": (target_w - logo_size - margin, target_h - logo_h - margin)}
                    x, y = positions.get(pos, positions["top_right"])
                    img = img.convert("RGBA")
                    img.paste(logo_img, (x, y), logo_img)
                    img = img.convert("RGB")
                except:
                    pass
        
        draw = ImageDraw.Draw(img)
        font_size = self.fontsize_var.get()
        margin_bottom = self.margin_var.get()
        outline_width = self.outline_var.get() if hasattr(self, 'outline_var') else 3

        # FFmpeg force_style uses FontSize literally without resolution scaling
        # So preview should also use font_size directly
        try:
            font_file = self.font_file_var.get()
            if font_file and os.path.exists(font_file):
                font = ImageFont.truetype(font_file, font_size)
            else:
                font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        sample_text = "Sample Subtitle - Hello World!"
        bbox = draw.textbbox((0, 0), sample_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (target_w - text_w) // 2
        # Position from bottom edge
        text_y = target_h - margin_bottom - text_h

        outline_color = self.outlinecolor_var.get()
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx or dy:
                    draw.text((text_x + dx, text_y + dy), sample_text, font=font, fill=outline_color)
        draw.text((text_x, text_y), sample_text, font=font, fill=self.fontcolor_var.get())
        
        canvas_w = 640
        canvas_h = int(canvas_w * target_h / target_w)
        if canvas_h > 400:
            canvas_h = 400
            canvas_w = int(canvas_h * target_w / target_h)
        
        img = img.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        self.preview_canvas.configure(width=canvas_w, height=canvas_h)
        self.preview_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=self.preview_photo)
        self.preview_info.config(text=f"{target_w}x{target_h} | Font: {font_size}px | Viền: {outline_width}px")
    
    def get_ui_template(self):
        font_file = ""
        if hasattr(self, 'font_file_var'):
            font_file = self.font_file_var.get()
        font_name = self.font_var.get()
        if not font_file and os.path.isdir(FONTS_DIR):
            for f in os.listdir(FONTS_DIR):
                if f.lower().endswith(('.ttf', '.otf')):
                    fpath = os.path.join(FONTS_DIR, f)
                    fname = get_font_name_from_file(fpath)
                    if fname == font_name:
                        font_file = fpath
                        break
        if font_file and os.path.exists(font_file):
            real_name = get_font_name_from_file(font_file)
            if real_name:
                font_name = real_name
        logo_file = os.path.join(LOGOS_DIR, self.logo_var.get()) if self.logo_var.get() else ""
        filter_val = self.filter_var.get().split(":")[0] if ":" in self.filter_var.get() else self.filter_var.get()
        overlay_val = self.overlay_var.get().split(":")[0] if ":" in self.overlay_var.get() else self.overlay_var.get()
        return {
            "name": self.template_var.get(), "target_width": self.width_var.get(), "target_height": self.height_var.get(),
            "font_file": font_file, "font_name": font_name, "font_size": self.fontsize_var.get(),
            "font_color": self.fontcolor_var.get(), "outline_color": self.outlinecolor_var.get(),
            "outline_width": self.outline_var.get() if hasattr(self, 'outline_var') else 3,
            "margin_bottom": self.margin_var.get(), "logo_enabled": self.logo_enabled_var.get(),
            "logo_file": logo_file, "logo_position": self.logopos_var.get(),
            "logo_size_percent": self.logosize_var.get(), "logo_opacity": 0.85,
            "filter_effect": filter_val, "video_overlay": overlay_val,
            "overlay_opacity": self.overlay_opacity_var.get(), "transition_enabled": self.trans_var.get(),
            "transition_duration": self.transdur_var.get(), "transition_type": self.transtype_var.get(),
            "whisper_model": self.whisper_var.get(),
            "whisper_language": self.lang_var.get(), "use_gpu": self.gpu_var.get(),
        }
    
    def load_template_to_ui(self, tpl):
        self.width_var.set(tpl.get("target_width", 1920))
        self.height_var.set(tpl.get("target_height", 1080))
        font_file = tpl.get("font_file", "")
        if hasattr(self, 'font_file_var'):
            self.font_file_var.set(font_file)
        if font_file and os.path.exists(font_file):
            real_name = get_font_name_from_file(font_file)
            if real_name:
                self.font_var.set(real_name)
            else:
                self.font_var.set(tpl.get("font_name", "Arial"))
        else:
            self.font_var.set(tpl.get("font_name", "Arial"))
        self.fontsize_var.set(tpl.get("font_size", 72))
        self.fontcolor_var.set(tpl.get("font_color", "#FFFFFF"))
        self.outlinecolor_var.set(tpl.get("outline_color", "#000000"))
        if hasattr(self, 'outline_var'):
            self.outline_var.set(tpl.get("outline_width", 3))
        self.margin_var.set(tpl.get("margin_bottom", 50))
        self.logo_enabled_var.set(tpl.get("logo_enabled", False))
        self.logo_var.set(os.path.basename(tpl.get("logo_file", "")) if tpl.get("logo_file") else "")
        self.logopos_var.set(tpl.get("logo_position", "top_right"))
        self.logosize_var.set(tpl.get("logo_size_percent", 15))
        self.filter_var.set(tpl.get("filter_effect", "none"))
        self.overlay_var.set(tpl.get("video_overlay", ""))
        self.overlay_opacity_var.set(tpl.get("overlay_opacity", 0.5))
        self.trans_var.set(tpl.get("transition_enabled", True))
        self.transdur_var.set(tpl.get("transition_duration", 0.8))
        self.transtype_var.set(tpl.get("transition_type", "random"))
        self.whisper_var.set(tpl.get("whisper_model", "tiny"))
        self.lang_var.set(tpl.get("whisper_language", "en"))
        self.fontcolor_btn.configure(bg=self.fontcolor_var.get())
        self.outlinecolor_btn.configure(bg=self.outlinecolor_var.get())
    
    def on_template_change(self, event=None):
        name = self.template_var.get()
        self.current_template = name
        tpl = self.template_manager.get(name)
        self.load_template_to_ui(tpl)
        self.log(f"📋 Đã tải template: {name}")
    
    def save_template(self):
        name = self.template_var.get()
        tpl = self.get_ui_template()
        self.template_manager.save(name, tpl)
        self.template_combo['values'] = self.template_manager.get_list()
        if hasattr(self, 'quick_template_combo'):
            self.quick_template_combo['values'] = self.template_manager.get_list()
        font_info = tpl.get("font_name", "Arial")
        if tpl.get("font_file"):
            font_info += f" ({os.path.basename(tpl['font_file'])})"
        self.log(f"💾 Đã lưu template: {name} | Font: {font_info}")
    
    def new_template(self):
        name = simpledialog.askstring("Template mới", "Tên template:", parent=self.root)
        if name and name not in self.template_manager.templates:
            self.template_manager.save(name, self.get_ui_template())
            self.template_combo['values'] = self.template_manager.get_list()
            if hasattr(self, 'quick_template_combo'):
                self.quick_template_combo['values'] = self.template_manager.get_list()
            self.template_var.set(name)
            self.log(f"➕ Đã tạo template: {name}")
    
    def delete_template(self):
        name = self.template_var.get()
        if name == "Mặc định":
            messagebox.showwarning("Lỗi", "Không thể xóa template mặc định!")
            return
        if messagebox.askyesno("Xác nhận", f"Xóa template '{name}'?"):
            self.template_manager.delete(name)
            self.template_combo['values'] = self.template_manager.get_list()
            if hasattr(self, 'quick_template_combo'):
                self.quick_template_combo['values'] = self.template_manager.get_list()
            self.template_var.set("Mặc định")
            self.on_template_change()
            self.log(f"🗑 Đã xóa template: {name}")
    
    def start_process(self):
        input_folder = self.input_var.get()
        output_folder = self.output_var.get()
        if not input_folder or not output_folder:
            messagebox.showerror("Lỗi", "Vui lòng chọn thư mục nguồn và thư mục xuất!")
            return
        os.makedirs(output_folder, exist_ok=True)
        self.save_config()
        self.save_template()
        folders_to_process = []
        for path, data in self.folder_data.items():
            if not data.get("enabled", False):
                continue
            out_path = os.path.join(output_folder, f"{os.path.basename(path)}.mp4")
            if self.skip_var.get() and os.path.exists(out_path):
                continue
            audio = list_media_files(path, (".mp3", ".wav", ".m4a"))
            video = list_media_files(path, (".mp4", ".mov", ".mkv"))
            img = list_media_files(path, (".jpg", ".jpeg", ".png", ".webp"))
            if audio and (video or img):
                template_name = data.get("template", self.template_var.get())
                folders_to_process.append((path, template_name))
                self.log(f"📁 Thêm: {os.path.basename(path)} → {template_name}")
        if not folders_to_process:
            messagebox.showinfo("Thông báo", "Không có file nào cần xử lý!")
            return
        self.log(f"--- TEMPLATE INFO ---")
        for path, template_name in folders_to_process:
            tpl = self.template_manager.get(template_name)
            font_file = tpl.get("font_file", "")
            font_name = tpl.get("font_name", "Arial")
            font_exists = "✓" if (font_file and os.path.exists(font_file)) else "system"
            self.log(f"  {os.path.basename(path)}: {template_name} → Font: {font_name} ({font_exists})")
        self.log(f"-------------------")
        self.processing = True
        self.stop_flag = False
        self.process_times = []
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        thread = threading.Thread(target=self.process_all, args=(folders_to_process,), daemon=True)
        thread.start()
    
    def process_all(self, folders_with_templates):
        total = len(folders_with_templates)
        completed = 0
        success, failed = 0, 0
        output_folder = self.output_var.get()
        workers = self.workers_var.get()
        use_gpu = self.gpu_var.get()
        start_time = time.time()
        current_video_progress = [0]  # Use list to allow modification in nested function

        self.log(f"{'='*50}")
        self.log(f"▶ BẮT ĐẦU XỬ LÝ: {total} video | Workers: {workers}")
        self.log(f"{'='*50}")
        # Show initial progress immediately
        self.root.after(0, lambda: self.update_progress(0, f"Đang xử lý: 0/{total} video...", -1))

        def log_fn(msg):
            self.root.after(0, lambda m=msg: self.log(m))

        def progress_fn(step_percent):
            """Update progress for current video step"""
            current_video_progress[0] = step_percent
            # Calculate overall: completed videos + current video progress
            overall = (completed + step_percent / 100) / total * 100
            elapsed_total = time.time() - start_time
            status_text = f"Video {completed + 1}/{total} ({step_percent}%) | ⏱ {format_time(elapsed_total)}"
            self.root.after(0, lambda p=overall, t=status_text: self.update_progress(p, t, -1))

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for path, template_name in folders_with_templates:
                if self.stop_flag:
                    break
                template_data = deepcopy(self.template_manager.get(template_name))
                template_data["use_gpu"] = use_gpu
                self.log(f"📋 {os.path.basename(path)} → Template: {template_name}")
                future = executor.submit(process_folder, path, output_folder, template_data, log_fn, progress_fn)
                futures[future] = (path, template_name)
            for future in concurrent.futures.as_completed(futures):
                if self.stop_flag:
                    break
                path, tpl_name = futures[future]
                completed += 1
                ok, status, output_path, elapsed = future.result()
                if ok and status == "success":
                    success += 1
                    self.process_times.append(elapsed)
                    # Store output path for "Open file" feature
                    if output_path:
                        folder_name = os.path.basename(path)
                        self.completed_outputs[folder_name] = output_path
                elif status != "skipped":
                    failed += 1
                percent = completed / total * 100
                eta = -1
                if self.process_times:
                    avg_time = sum(self.process_times) / len(self.process_times)
                    remaining = total - completed
                    eta = avg_time * remaining / max(1, workers)
                elapsed_total = time.time() - start_time
                status_text = f"Xử lý: {completed}/{total} | ✓ {success} | ✗ {failed} | ⏱ {format_time(elapsed_total)}"
                self.root.after(0, lambda p=percent, t=status_text, e=eta: self.update_progress(p, t, e))
                def update_tree_item(folder_path, is_ok):
                    for item in self.tree.get_children():
                        values = list(self.tree.item(item, 'values'))
                        if values[0] == os.path.basename(folder_path):
                            if is_ok:
                                values[2] = "✓ Xong"
                                values[3] = "📂"  # Show open button
                                self.tree.item(item, values=values, tags=("done",))
                            else:
                                values[2] = "✗ Lỗi"
                                values[3] = ""  # No open button for failed
                                self.tree.item(item, values=values, tags=("error",))
                            break
                    self.update_stats()
                self.root.after(0, lambda p=path, o=ok: update_tree_item(p, o and status == "success"))
        elapsed_total = time.time() - start_time
        end_time_str = datetime.now().strftime("%H:%M:%S")
        start_time_str = (datetime.now() - timedelta(seconds=elapsed_total)).strftime("%H:%M:%S")
        self.log(f"{'='*50}")
        self.log(f"✓ HOÀN THÀNH: {success} thành công | {failed} lỗi")
        self.log(f"⏱ Bắt đầu: {start_time_str} | Kết thúc: {end_time_str} | Tổng: {format_time(elapsed_total)}")
        if self.process_times:
            avg = sum(self.process_times) / len(self.process_times)
            self.log(f"📊 Trung bình: {format_time(avg)}/video")
        self.log(f"{'='*50}")
        self.root.after(0, lambda: self.update_progress(100, f"✓ Hoàn thành: {success} thành công, {failed} lỗi", -1))
        self.processing = False
        self.root.after(0, lambda: self.start_btn.configure(state=tk.NORMAL))
        self.root.after(0, lambda: self.stop_btn.configure(state=tk.DISABLED))
    
    def stop_process(self):
        self.stop_flag = True
        self.log("⏹ Đang dừng xử lý...")

# ===== MAIN =====
def main():
    root = tk.Tk()
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
    app = UnixAutoEdit(root)
    root.mainloop()

if __name__ == "__main__":
    main()
