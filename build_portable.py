# -*- coding: utf-8 -*-
"""
Build Portable Package for Uni-x Auto Edit
==========================================
Script to create a portable Windows package with:
- Python 3.11 embedded
- All required dependencies
- FFmpeg binaries
- Auto-update launcher

Usage:
    python build_portable.py

Output:
    UnixAutoEdit_Portable/
"""

import os
import sys
import shutil
import urllib.request
import zipfile
import subprocess
import tempfile
from pathlib import Path

# ===== CONFIGURATION =====
PYTHON_VERSION = "3.11.9"
PYTHON_EMBED_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

# Dependencies to install
DEPENDENCIES = [
    "pillow",
    "faster-whisper",
    "fonttools",
    "torch",
    "torchaudio",
]

# Optional dependencies (will try to install but won't fail if they don't)
OPTIONAL_DEPS = [
    "whisper-timestamped",
    "openai-whisper",
]

# Build directory
BUILD_DIR = "UnixAutoEdit_Portable"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_system_python():
    """Find system Python installation with tkinter."""
    possible_paths = [
        # Common installation paths
        r"C:\Python311",
        r"C:\Python310",
        r"C:\Python39",
        r"C:\Python312",
        # User installation paths
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python310"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python39"),
        # Program Files
        r"C:\Program Files\Python311",
        r"C:\Program Files\Python310",
        # Current Python
        os.path.dirname(sys.executable),
    ]

    for path in possible_paths:
        if not path or not os.path.isdir(path):
            continue
        # Check if this Python has tkinter
        tkinter_path = os.path.join(path, "Lib", "tkinter")
        if os.path.isdir(tkinter_path):
            return path

    return None


def copy_tkinter(build_dir):
    """Copy tkinter from system Python to embedded Python."""
    log("Looking for system Python with tkinter...", "INFO")

    src_python = find_system_python()

    if not src_python:
        log("Could not find system Python with tkinter!", "ERROR")
        log("Please install Python from python.org with tkinter included", "ERROR")
        return False

    log(f"Found system Python: {src_python}", "SUCCESS")

    python_dir = os.path.join(build_dir, "python")

    # Files/folders to copy
    copy_items = [
        # tkinter module
        (os.path.join(src_python, "Lib", "tkinter"), os.path.join(python_dir, "Lib", "tkinter")),
        # _tkinter.pyd (the C extension)
        (os.path.join(src_python, "DLLs", "_tkinter.pyd"), os.path.join(python_dir, "_tkinter.pyd")),
        # Tcl/Tk DLLs
        (os.path.join(src_python, "DLLs", "tcl86t.dll"), os.path.join(python_dir, "tcl86t.dll")),
        (os.path.join(src_python, "DLLs", "tk86t.dll"), os.path.join(python_dir, "tk86t.dll")),
        # Tcl/Tk libraries
        (os.path.join(src_python, "tcl"), os.path.join(python_dir, "tcl")),
    ]

    # Alternative DLL locations (some Python versions)
    alt_dll_paths = [
        os.path.join(src_python, "DLLs"),
        src_python,
        os.path.join(src_python, "Library", "bin"),
    ]

    success_count = 0

    for src, dst in copy_items:
        if os.path.exists(src):
            try:
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                log(f"Copied: {os.path.basename(src)}", "SUCCESS")
                success_count += 1
            except Exception as e:
                log(f"Failed to copy {src}: {e}", "WARNING")
        else:
            # Try alternative locations for DLLs
            basename = os.path.basename(src)
            found = False
            for alt_path in alt_dll_paths:
                alt_src = os.path.join(alt_path, basename)
                if os.path.exists(alt_src):
                    try:
                        shutil.copy2(alt_src, dst)
                        log(f"Copied (alt): {basename}", "SUCCESS")
                        success_count += 1
                        found = True
                        break
                    except:
                        pass
            if not found:
                log(f"Not found: {basename}", "WARNING")

    # Verify tkinter works
    if success_count >= 3:  # At minimum need tkinter folder, _tkinter.pyd, and tcl folder
        log("Tkinter files copied successfully", "SUCCESS")
        return True
    else:
        log("Some tkinter files could not be copied", "WARNING")
        return True  # Continue anyway


def log(msg, level="INFO"):
    """Print log message with formatting."""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
    }
    reset = "\033[0m"
    print(f"{colors.get(level, '')}{level}: {msg}{reset}")


def download_file(url, dest, desc=""):
    """Download file with progress indicator."""
    log(f"Downloading {desc or url}...")

    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 // total_size)
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "=" * filled + "-" * (bar_len - filled)
            print(f"\r[{bar}] {percent}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, progress_hook)
        print()  # New line after progress bar
        return True
    except Exception as e:
        log(f"Download failed: {e}", "ERROR")
        return False


def extract_zip(zip_path, dest_dir, desc=""):
    """Extract zip file."""
    log(f"Extracting {desc or zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        return True
    except Exception as e:
        log(f"Extraction failed: {e}", "ERROR")
        return False


def setup_python_embedded(build_dir):
    """Download and setup Python embedded."""
    python_dir = os.path.join(build_dir, "python")
    os.makedirs(python_dir, exist_ok=True)

    # Download Python embedded
    python_zip = os.path.join(tempfile.gettempdir(), "python_embed.zip")
    if not download_file(PYTHON_EMBED_URL, python_zip, f"Python {PYTHON_VERSION} Embedded"):
        return False

    # Extract Python
    if not extract_zip(python_zip, python_dir, "Python"):
        return False

    # Modify python311._pth to enable site-packages
    pth_file = os.path.join(python_dir, f"python{PYTHON_VERSION.replace('.', '')[:2]}._pth")
    # Find the correct pth file
    for f in os.listdir(python_dir):
        if f.endswith("._pth"):
            pth_file = os.path.join(python_dir, f)
            break

    if os.path.exists(pth_file):
        log("Configuring Python paths...")
        with open(pth_file, "r") as f:
            content = f.read()

        # Uncomment import site and add Lib\site-packages
        content = content.replace("#import site", "import site")
        if "Lib\\site-packages" not in content:
            content += "\nLib\\site-packages\n"

        with open(pth_file, "w") as f:
            f.write(content)

    # Create Lib/site-packages directory
    site_packages = os.path.join(python_dir, "Lib", "site-packages")
    os.makedirs(site_packages, exist_ok=True)

    # Download and run get-pip.py
    get_pip = os.path.join(tempfile.gettempdir(), "get-pip.py")
    if not download_file(GET_PIP_URL, get_pip, "get-pip.py"):
        return False

    log("Installing pip...")
    python_exe = os.path.join(python_dir, "python.exe")
    result = subprocess.run(
        [python_exe, get_pip, "--no-warn-script-location"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        log(f"pip installation failed: {result.stderr}", "ERROR")
        return False

    log("pip installed successfully", "SUCCESS")

    # Clean up
    try:
        os.remove(python_zip)
        os.remove(get_pip)
    except:
        pass

    return True


def install_dependencies(build_dir):
    """Install Python dependencies."""
    python_dir = os.path.join(build_dir, "python")
    python_exe = os.path.join(python_dir, "python.exe")

    # Upgrade pip first
    log("Upgrading pip...")
    subprocess.run(
        [python_exe, "-m", "pip", "install", "--upgrade", "pip", "--no-warn-script-location"],
        capture_output=True
    )

    # Install main dependencies
    for dep in DEPENDENCIES:
        log(f"Installing {dep}...")
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", dep, "--no-warn-script-location"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            log(f"Warning: {dep} installation may have issues: {result.stderr}", "WARNING")
        else:
            log(f"{dep} installed", "SUCCESS")

    # Try to install optional dependencies
    for dep in OPTIONAL_DEPS:
        log(f"Installing optional: {dep}...")
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", dep, "--no-warn-script-location"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            log(f"{dep} installed", "SUCCESS")
        else:
            log(f"{dep} skipped (optional)", "WARNING")

    return True


def setup_ffmpeg(build_dir):
    """Download and setup FFmpeg."""
    ffmpeg_dir = os.path.join(build_dir, "ffmpeg")
    os.makedirs(ffmpeg_dir, exist_ok=True)

    # Download FFmpeg
    ffmpeg_zip = os.path.join(tempfile.gettempdir(), "ffmpeg.zip")
    if not download_file(FFMPEG_URL, ffmpeg_zip, "FFmpeg"):
        log("Trying alternative FFmpeg source...", "WARNING")
        # Alternative: gyan.dev builds
        alt_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        if not download_file(alt_url, ffmpeg_zip, "FFmpeg (alternative)"):
            return False

    # Extract to temp and copy binaries
    temp_extract = os.path.join(tempfile.gettempdir(), "ffmpeg_extract")
    if os.path.exists(temp_extract):
        shutil.rmtree(temp_extract)

    if not extract_zip(ffmpeg_zip, temp_extract, "FFmpeg"):
        return False

    # Find ffmpeg.exe and ffprobe.exe
    log("Locating FFmpeg binaries...")
    for root, dirs, files in os.walk(temp_extract):
        for f in files:
            if f in ["ffmpeg.exe", "ffprobe.exe"]:
                src = os.path.join(root, f)
                dst = os.path.join(ffmpeg_dir, f)
                shutil.copy2(src, dst)
                log(f"Copied {f}", "SUCCESS")

    # Clean up
    try:
        os.remove(ffmpeg_zip)
        shutil.rmtree(temp_extract)
    except:
        pass

    # Verify
    if os.path.exists(os.path.join(ffmpeg_dir, "ffmpeg.exe")):
        return True
    else:
        log("FFmpeg binaries not found", "ERROR")
        return False


def create_folder_structure(build_dir):
    """Create the folder structure."""
    folders = [
        "app",
        "templates",
        "fonts",
        "logos",
        "effects",
    ]

    for folder in folders:
        path = os.path.join(build_dir, folder)
        os.makedirs(path, exist_ok=True)
        log(f"Created {folder}/", "SUCCESS")

    return True


def copy_app_files(build_dir):
    """Copy application files."""
    # Copy main script
    src_script = os.path.join(SCRIPT_DIR, "unixautoedit.py")
    dst_script = os.path.join(build_dir, "app", "unixautoedit.py")

    if os.path.exists(src_script):
        shutil.copy2(src_script, dst_script)
        log("Copied unixautoedit.py", "SUCCESS")
    else:
        log("unixautoedit.py not found - will need to be added manually", "WARNING")

    # Copy version.txt
    src_version = os.path.join(SCRIPT_DIR, "version.txt")
    dst_version = os.path.join(build_dir, "app", "version.txt")

    if os.path.exists(src_version):
        shutil.copy2(src_version, dst_version)
        log("Copied version.txt", "SUCCESS")
    else:
        # Create default version.txt
        with open(dst_version, "w") as f:
            f.write("2.3.0\n")
        log("Created version.txt", "SUCCESS")

    return True


def create_launcher(build_dir):
    """Create launcher.py with auto-update functionality."""
    launcher_content = '''# -*- coding: utf-8 -*-
"""
Uni-x Auto Edit Launcher
========================
- Checks for updates from GitHub
- Downloads new version if available
- Launches main application
"""

import os
import sys
import urllib.request
import json
import tkinter as tk
from tkinter import messagebox
import subprocess
import threading
import time

# ===== CONFIGURATION =====
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/nguyenvantinh2391994-blip/UnixAutoEdit/main"
VERSION_URL = f"{GITHUB_RAW_BASE}/version.txt"
SCRIPT_URL = f"{GITHUB_RAW_BASE}/unixautoedit.py"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
APP_SCRIPT = os.path.join(SCRIPT_DIR, "unixautoedit.py")
VERSION_FILE = os.path.join(SCRIPT_DIR, "version.txt")


def get_local_version():
    """Read local version from version.txt"""
    try:
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return "0.0.0"


def get_remote_version():
    """Fetch remote version from GitHub"""
    try:
        with urllib.request.urlopen(VERSION_URL, timeout=10) as response:
            return response.read().decode("utf-8").strip()
    except Exception as e:
        print(f"Could not check for updates: {e}")
        return None


def compare_versions(local, remote):
    """Compare version strings. Returns True if remote is newer."""
    try:
        local_parts = [int(x) for x in local.split(".")]
        remote_parts = [int(x) for x in remote.split(".")]

        # Pad shorter version with zeros
        while len(local_parts) < len(remote_parts):
            local_parts.append(0)
        while len(remote_parts) < len(local_parts):
            remote_parts.append(0)

        return remote_parts > local_parts
    except:
        return False


def download_update(progress_callback=None):
    """Download the new script version"""
    try:
        # Download new script
        with urllib.request.urlopen(SCRIPT_URL, timeout=60) as response:
            content = response.read()

        # Backup old script
        if os.path.exists(APP_SCRIPT):
            backup_path = APP_SCRIPT + ".backup"
            try:
                os.replace(APP_SCRIPT, backup_path)
            except:
                pass

        # Save new script
        with open(APP_SCRIPT, "wb") as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Update download failed: {e}")
        # Restore backup if exists
        backup_path = APP_SCRIPT + ".backup"
        if os.path.exists(backup_path):
            try:
                os.replace(backup_path, APP_SCRIPT)
            except:
                pass
        return False


def update_local_version(version):
    """Update local version.txt"""
    try:
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            f.write(version)
    except:
        pass


class UpdateDialog:
    """Simple update dialog using Tkinter"""

    def __init__(self, local_ver, remote_ver):
        self.result = None
        self.local_ver = local_ver
        self.remote_ver = remote_ver

        self.root = tk.Tk()
        self.root.title("Update Available")
        self.root.geometry("400x200")
        self.root.resizable(False, False)

        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 400) // 2
        y = (self.root.winfo_screenheight() - 200) // 2
        self.root.geometry(f"400x200+{x}+{y}")

        # Icon
        try:
            self.root.iconbitmap(os.path.join(BASE_DIR, "icon.ico"))
        except:
            pass

        # Content
        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="Uni-x Auto Edit", font=("Segoe UI", 14, "bold")).pack()
        tk.Label(frame, text="", font=("Segoe UI", 10)).pack()
        tk.Label(frame, text=f"Current version: {local_ver}", font=("Segoe UI", 10)).pack()
        tk.Label(frame, text=f"New version: {remote_ver}", font=("Segoe UI", 10), fg="green").pack()
        tk.Label(frame, text="", font=("Segoe UI", 10)).pack()
        tk.Label(frame, text="Do you want to update?", font=("Segoe UI", 10)).pack()

        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=15)

        tk.Button(btn_frame, text="Update Now", command=self.on_update,
                  width=12, bg="#22c55e", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Skip", command=self.on_skip,
                  width=12).pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_skip)

    def on_update(self):
        self.result = True
        self.root.destroy()

    def on_skip(self):
        self.result = False
        self.root.destroy()

    def show(self):
        self.root.mainloop()
        return self.result


class ProgressDialog:
    """Download progress dialog"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Updating...")
        self.root.geometry("300x100")
        self.root.resizable(False, False)

        # Center
        x = (self.root.winfo_screenwidth() - 300) // 2
        y = (self.root.winfo_screenheight() - 100) // 2
        self.root.geometry(f"300x100+{x}+{y}")

        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        self.label = tk.Label(frame, text="Downloading update...", font=("Segoe UI", 10))
        self.label.pack()

        self.progress = tk.Label(frame, text="Please wait...", font=("Segoe UI", 9), fg="gray")
        self.progress.pack(pady=10)

        self.root.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close

    def update_status(self, text):
        self.progress.config(text=text)
        self.root.update()

    def close(self):
        self.root.destroy()

    def run_update(self, remote_ver):
        self.root.after(100, lambda: self._do_update(remote_ver))
        self.root.mainloop()

    def _do_update(self, remote_ver):
        self.update_status("Downloading...")

        if download_update():
            update_local_version(remote_ver)
            self.update_status("Update complete!")
            self.root.after(1000, self.close)
        else:
            self.update_status("Update failed!")
            self.root.after(2000, self.close)


def check_and_update():
    """Check for updates and prompt user if available"""
    local_ver = get_local_version()
    remote_ver = get_remote_version()

    if remote_ver is None:
        print(f"Running version {local_ver} (offline mode)")
        return True

    if compare_versions(local_ver, remote_ver):
        print(f"Update available: {local_ver} -> {remote_ver}")

        # Show update dialog
        dialog = UpdateDialog(local_ver, remote_ver)
        should_update = dialog.show()

        if should_update:
            progress = ProgressDialog()
            progress.run_update(remote_ver)
            return True
    else:
        print(f"Running latest version {local_ver}")

    return True


def setup_environment():
    """Setup environment variables for portable installation"""
    # Add FFmpeg to PATH
    ffmpeg_dir = os.path.join(BASE_DIR, "ffmpeg")
    if os.path.isdir(ffmpeg_dir):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    # Set working directory
    os.chdir(BASE_DIR)

    # Add app directory to Python path
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)


def launch_app():
    """Launch the main application"""
    if not os.path.exists(APP_SCRIPT):
        messagebox.showerror("Error", f"Application script not found:\\n{APP_SCRIPT}")
        return False

    # Import and run
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("unixautoedit", APP_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules["unixautoedit"] = module
        spec.loader.exec_module(module)

        # Run main
        if hasattr(module, "main"):
            module.main()
        else:
            print("No main() function found in unixautoedit.py")

        return True
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch application:\\n{e}")
        return False


def main():
    """Main entry point"""
    print("=" * 50)
    print("Uni-x Auto Edit Launcher")
    print("=" * 50)

    # Setup environment
    setup_environment()

    # Check for updates
    check_and_update()

    # Launch application
    launch_app()


if __name__ == "__main__":
    main()
'''

    launcher_path = os.path.join(build_dir, "app", "launcher.py")
    with open(launcher_path, "w", encoding="utf-8") as f:
        f.write(launcher_content)

    log("Created launcher.py", "SUCCESS")
    return True


def create_run_bat(build_dir):
    """Create Run.bat file."""
    bat_content = '''@echo off
cd /d "%~dp0"
start "" python\\pythonw.exe app\\launcher.py
'''

    bat_path = os.path.join(build_dir, "Run.bat")
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)

    log("Created Run.bat", "SUCCESS")

    # Also create a debug version that shows console
    debug_bat_content = '''@echo off
cd /d "%~dp0"
python\\python.exe app\\launcher.py
pause
'''

    debug_bat_path = os.path.join(build_dir, "Run_Debug.bat")
    with open(debug_bat_path, "w", encoding="utf-8") as f:
        f.write(debug_bat_content)

    log("Created Run_Debug.bat (for troubleshooting)", "SUCCESS")

    return True


def create_readme(build_dir):
    """Create README file."""
    readme_content = '''# Uni-x Auto Edit - Portable Version

## Quick Start
1. Double-click `Run.bat` to start the application
2. Use `Run_Debug.bat` if you need to see error messages

## Folder Structure
- `python/` - Embedded Python interpreter
- `app/` - Application scripts
- `ffmpeg/` - FFmpeg binaries
- `templates/` - Video templates
- `fonts/` - Custom fonts
- `logos/` - Logo images
- `effects/` - Video overlay effects

## Auto-Update
The application will automatically check for updates when launched.
Updates are downloaded from the official GitHub repository.

## Requirements
- Windows 10/11 (64-bit)
- No installation needed

## Troubleshooting
- If the application doesn't start, run `Run_Debug.bat` to see error messages
- Make sure Windows Defender or antivirus is not blocking the application
- If FFmpeg is missing, download it manually and place ffmpeg.exe and ffprobe.exe in the ffmpeg/ folder

## Credits
Uni-x Auto Edit v2.3
GitHub: https://github.com/nguyenvantinh2391994-blip/UnixAutoEdit
'''

    readme_path = os.path.join(build_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    log("Created README.txt", "SUCCESS")
    return True


def main():
    """Main build process."""
    print("=" * 60)
    print("  Uni-x Auto Edit - Portable Package Builder")
    print("=" * 60)
    print()

    # Create build directory
    build_dir = os.path.join(SCRIPT_DIR, BUILD_DIR)

    if os.path.exists(build_dir):
        log(f"Removing existing build directory: {build_dir}", "WARNING")
        shutil.rmtree(build_dir)

    os.makedirs(build_dir)
    log(f"Build directory: {build_dir}", "INFO")
    print()

    # Step 1: Create folder structure
    log("Step 1: Creating folder structure...", "INFO")
    if not create_folder_structure(build_dir):
        return 1
    print()

    # Step 2: Setup Python embedded
    log("Step 2: Setting up Python embedded...", "INFO")
    if not setup_python_embedded(build_dir):
        log("Failed to setup Python", "ERROR")
        return 1
    print()

    # Step 2.5: Copy tkinter from system Python
    log("Step 2.5: Copying tkinter from system Python...", "INFO")
    if not copy_tkinter(build_dir):
        log("Tkinter copy failed - GUI may not work", "WARNING")
    print()

    # Step 3: Install dependencies
    log("Step 3: Installing dependencies...", "INFO")
    if not install_dependencies(build_dir):
        log("Some dependencies may not be installed", "WARNING")
    print()

    # Step 4: Setup FFmpeg
    log("Step 4: Setting up FFmpeg...", "INFO")
    if not setup_ffmpeg(build_dir):
        log("FFmpeg setup failed - manual installation required", "WARNING")
    print()

    # Step 5: Copy application files
    log("Step 5: Copying application files...", "INFO")
    if not copy_app_files(build_dir):
        log("Failed to copy app files", "ERROR")
        return 1
    print()

    # Step 6: Create launcher
    log("Step 6: Creating launcher...", "INFO")
    if not create_launcher(build_dir):
        log("Failed to create launcher", "ERROR")
        return 1
    print()

    # Step 7: Create batch files
    log("Step 7: Creating batch files...", "INFO")
    if not create_run_bat(build_dir):
        log("Failed to create batch files", "ERROR")
        return 1
    print()

    # Step 8: Create README
    log("Step 8: Creating documentation...", "INFO")
    create_readme(build_dir)
    print()

    # Done
    print("=" * 60)
    log("BUILD COMPLETE!", "SUCCESS")
    print("=" * 60)
    print()
    print(f"Output directory: {build_dir}")
    print()
    print("Next steps:")
    print("1. Copy the entire folder to the target machine")
    print("2. Double-click Run.bat to start")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
