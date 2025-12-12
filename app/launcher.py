# -*- coding: utf-8 -*-
"""
Uni-x Auto Edit Launcher
========================
- Checks for updates from GitHub
- Downloads new version if available
- Launches main application

Author: UnixAutoEdit Team
Version: 2.3.1
"""

import os
import sys
import urllib.request
import urllib.error
import threading
import time
import traceback

# Try to import tkinter with helpful error message
TKINTER_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import messagebox
    TKINTER_AVAILABLE = True
except ImportError as e:
    print("=" * 60)
    print("ERROR: tkinter is not available!")
    print("=" * 60)
    print(f"Import error: {e}")
    print()
    print("Possible solutions:")
    print("1. Run build_portable.py again to setup tkinter")
    print("2. Make sure Python was installed with tkinter support")
    print("3. On Windows, tkinter files may be missing from python/")
    print()
    print("Required files for tkinter:")
    print("  - python/DLLs/_tkinter.pyd")
    print("  - python/tcl86t.dll, tk86t.dll")
    print("  - python/tcl/ folder")
    print("  - python/Lib/tkinter/ folder")
    print("=" * 60)

    # Create dummy tk module for fallback
    class DummyTk:
        BOTH = "both"
        X = "x"
        Y = "y"
        LEFT = "left"
        RIGHT = "right"
        W = "w"
        FLAT = "flat"

        class Tk:
            def __init__(self):
                raise RuntimeError("tkinter not available")

        class Frame:
            pass

        class Label:
            pass

        class Button:
            pass

    class DummyMessagebox:
        @staticmethod
        def showerror(title, message):
            print(f"ERROR: {title}")
            print(message)

        @staticmethod
        def showinfo(title, message):
            print(f"INFO: {title}")
            print(message)

    tk = DummyTk()
    messagebox = DummyMessagebox()

# ===== CONFIGURATION =====
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/nguyenvantinh2391994-blip/UnixAutoEdit/main"
VERSION_URL = f"{GITHUB_RAW_BASE}/version.txt"
SCRIPT_URL = f"{GITHUB_RAW_BASE}/unixautoedit.py"
LAUNCHER_URL = f"{GITHUB_RAW_BASE}/app/launcher.py"  # Launcher tự update

# Connection settings
TIMEOUT = 15  # seconds
MAX_RETRIES = 2

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
APP_SCRIPT = os.path.join(SCRIPT_DIR, "unixautoedit.py")
LAUNCHER_SCRIPT = os.path.join(SCRIPT_DIR, "launcher.py")  # Chính nó
VERSION_FILE = os.path.join(SCRIPT_DIR, "version.txt")


def log(msg):
    """Print log message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def get_local_version():
    """Read local version from version.txt"""
    try:
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            version = f.read().strip()
            log(f"Local version: {version}")
            return version
    except FileNotFoundError:
        log("version.txt not found, assuming 0.0.0")
        return "0.0.0"
    except Exception as e:
        log(f"Error reading version: {e}")
        return "0.0.0"


def fetch_url(url, timeout=TIMEOUT, retries=MAX_RETRIES):
    """Fetch URL with retry logic."""
    last_error = None

    for attempt in range(retries + 1):
        try:
            log(f"Fetching {url} (attempt {attempt + 1})")

            # Create request with headers
            request = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'UnixAutoEdit-Updater/2.3',
                    'Accept': 'text/plain, application/octet-stream',
                }
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()

        except urllib.error.URLError as e:
            last_error = e
            log(f"URL error: {e.reason}")
            if attempt < retries:
                time.sleep(1)

        except urllib.error.HTTPError as e:
            last_error = e
            log(f"HTTP error {e.code}: {e.reason}")
            if attempt < retries:
                time.sleep(1)

        except Exception as e:
            last_error = e
            log(f"Error: {e}")
            if attempt < retries:
                time.sleep(1)

    raise last_error if last_error else Exception("Unknown error")


def get_remote_version():
    """Fetch remote version from GitHub"""
    try:
        data = fetch_url(VERSION_URL)
        version = data.decode("utf-8").strip()
        log(f"Remote version: {version}")
        return version
    except Exception as e:
        log(f"Could not check for updates: {e}")
        return None


def compare_versions(local, remote):
    """
    Compare version strings.
    Returns True if remote is newer than local.
    """
    try:
        # Parse versions
        local_parts = [int(x) for x in local.replace("-", ".").split(".")]
        remote_parts = [int(x) for x in remote.replace("-", ".").split(".")]

        # Pad shorter version with zeros
        max_len = max(len(local_parts), len(remote_parts))
        while len(local_parts) < max_len:
            local_parts.append(0)
        while len(remote_parts) < max_len:
            remote_parts.append(0)

        # Compare
        is_newer = remote_parts > local_parts
        log(f"Version comparison: {local} vs {remote} -> newer={is_newer}")
        return is_newer

    except Exception as e:
        log(f"Version comparison error: {e}")
        return False


def download_update():
    """Download the new script version and launcher"""
    try:
        log("Downloading update...")

        # 1. Download main script (unixautoedit.py)
        content = fetch_url(SCRIPT_URL, timeout=60)

        # Validate content
        if len(content) < 1000:
            log("Downloaded content too small, may be invalid")
            return False

        # Check if it looks like Python code
        content_str = content.decode("utf-8", errors="ignore")
        if "import" not in content_str or "def " not in content_str:
            log("Downloaded content doesn't appear to be valid Python")
            return False

        # Backup old script
        if os.path.exists(APP_SCRIPT):
            backup_path = APP_SCRIPT + ".backup"
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(APP_SCRIPT, backup_path)
                log(f"Created backup: {backup_path}")
            except Exception as e:
                log(f"Backup warning: {e}")

        # Save new script
        with open(APP_SCRIPT, "wb") as f:
            f.write(content)

        log(f"Main script updated: {len(content)} bytes")

        # 2. Download launcher.py (tự update chính nó)
        try:
            log("Updating launcher...")
            launcher_content = fetch_url(LAUNCHER_URL, timeout=30)

            if len(launcher_content) > 1000:
                launcher_str = launcher_content.decode("utf-8", errors="ignore")
                if "import" in launcher_str and "def " in launcher_str:
                    # Save new launcher
                    with open(LAUNCHER_SCRIPT, "wb") as f:
                        f.write(launcher_content)
                    log(f"Launcher updated: {len(launcher_content)} bytes")
        except Exception as e:
            log(f"Launcher update skipped: {e}")
            # Không fail nếu không update được launcher

        return True

    except Exception as e:
        log(f"Update download failed: {e}")

        # Restore backup if exists
        backup_path = APP_SCRIPT + ".backup"
        if os.path.exists(backup_path) and not os.path.exists(APP_SCRIPT):
            try:
                os.rename(backup_path, APP_SCRIPT)
                log("Restored from backup")
            except:
                pass

        return False


def update_local_version(version):
    """Update local version.txt"""
    try:
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            f.write(version + "\n")
        log(f"Updated local version to {version}")
    except Exception as e:
        log(f"Failed to update version file: {e}")


# ===== GUI DIALOGS =====

class UpdateDialog:
    """Dialog to ask user about update"""

    def __init__(self, local_ver, remote_ver):
        self.result = None
        self.local_ver = local_ver
        self.remote_ver = remote_ver

        self.root = tk.Tk()
        self.root.title("Uni-x Auto Edit - Update Available")
        self.root.geometry("420x220")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - 420) // 2
        y = (self.root.winfo_screenheight() - 220) // 2
        self.root.geometry(f"420x220+{x}+{y}")

        # Try to set icon
        try:
            icon_path = os.path.join(BASE_DIR, "icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass

        self._create_widgets()

    def _create_widgets(self):
        # Main frame
        main = tk.Frame(self.root, bg="#1a1a2e", padx=30, pady=20)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            main,
            text="UNI-X AUTO EDIT",
            font=("Segoe UI", 18, "bold"),
            bg="#1a1a2e",
            fg="#6366f1"
        ).pack()

        # Update icon
        tk.Label(
            main,
            text="Update Available",
            font=("Segoe UI", 11),
            bg="#1a1a2e",
            fg="#22c55e"
        ).pack(pady=(5, 15))

        # Version info frame
        ver_frame = tk.Frame(main, bg="#252540", padx=15, pady=10)
        ver_frame.pack(fill=tk.X)

        tk.Label(
            ver_frame,
            text=f"Current version:  {self.local_ver}",
            font=("Segoe UI", 10),
            bg="#252540",
            fg="#94a3b8"
        ).pack(anchor=tk.W)

        tk.Label(
            ver_frame,
            text=f"New version:      {self.remote_ver}",
            font=("Segoe UI", 10, "bold"),
            bg="#252540",
            fg="#22c55e"
        ).pack(anchor=tk.W)

        # Question
        tk.Label(
            main,
            text="Do you want to update now?",
            font=("Segoe UI", 10),
            bg="#1a1a2e",
            fg="#f8fafc"
        ).pack(pady=15)

        # Buttons
        btn_frame = tk.Frame(main, bg="#1a1a2e")
        btn_frame.pack()

        tk.Button(
            btn_frame,
            text="Update Now",
            command=self.on_update,
            font=("Segoe UI", 10, "bold"),
            width=14,
            bg="#22c55e",
            fg="white",
            activebackground="#16a34a",
            relief=tk.FLAT,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Skip",
            command=self.on_skip,
            font=("Segoe UI", 10),
            width=14,
            bg="#374151",
            fg="white",
            activebackground="#4b5563",
            relief=tk.FLAT,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=5)

        # Close button handler
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
    """Progress dialog for download"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Updating...")
        self.root.geometry("350x120")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")

        # Center
        x = (self.root.winfo_screenwidth() - 350) // 2
        y = (self.root.winfo_screenheight() - 120) // 2
        self.root.geometry(f"350x120+{x}+{y}")

        # Content
        main = tk.Frame(self.root, bg="#1a1a2e", padx=25, pady=20)
        main.pack(fill=tk.BOTH, expand=True)

        self.title_label = tk.Label(
            main,
            text="Downloading Update...",
            font=("Segoe UI", 12, "bold"),
            bg="#1a1a2e",
            fg="#f8fafc"
        )
        self.title_label.pack()

        self.status_label = tk.Label(
            main,
            text="Please wait...",
            font=("Segoe UI", 10),
            bg="#1a1a2e",
            fg="#94a3b8"
        )
        self.status_label.pack(pady=10)

        # Progress bar (simple)
        self.progress_frame = tk.Frame(main, bg="#374151", height=8)
        self.progress_frame.pack(fill=tk.X, pady=5)
        self.progress_frame.pack_propagate(False)

        self.progress_bar = tk.Frame(self.progress_frame, bg="#6366f1", width=0)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.Y)

        self.root.protocol("WM_DELETE_WINDOW", lambda: None)

    def update_status(self, text, progress=None):
        self.status_label.config(text=text)
        if progress is not None:
            width = int(300 * progress / 100)
            self.progress_bar.config(width=width)
        self.root.update()

    def close(self):
        self.root.destroy()

    def run_update(self, remote_ver):
        """Run update in separate thread"""
        self.remote_ver = remote_ver
        self.success = False

        def do_update():
            self.root.after(100, lambda: self.update_status("Connecting...", 10))
            time.sleep(0.3)

            self.root.after(100, lambda: self.update_status("Downloading...", 30))

            if download_update():
                self.root.after(100, lambda: self.update_status("Updating version...", 80))
                time.sleep(0.2)

                update_local_version(self.remote_ver)

                self.root.after(100, lambda: self.update_status("Update complete!", 100))
                self.success = True
                time.sleep(1)
            else:
                self.root.after(100, lambda: self.update_status("Update failed!", 0))
                time.sleep(2)

            self.root.after(100, self.close)

        thread = threading.Thread(target=do_update, daemon=True)
        thread.start()

        self.root.mainloop()
        return self.success


# ===== MAIN FUNCTIONS =====

def setup_environment():
    """Setup environment for portable installation"""
    log("Setting up environment...")

    # Add FFmpeg to PATH
    ffmpeg_dir = os.path.join(BASE_DIR, "ffmpeg")
    if os.path.isdir(ffmpeg_dir):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        log(f"Added FFmpeg to PATH: {ffmpeg_dir}")

    # Set working directory to BASE_DIR
    os.chdir(BASE_DIR)
    log(f"Working directory: {BASE_DIR}")

    # Add app directory to Python path
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)


def check_and_update():
    """Check for updates and auto-update if available"""
    log("Checking for updates...")

    local_ver = get_local_version()
    remote_ver = get_remote_version()

    if remote_ver is None:
        log("Offline mode - skipping update check")
        return True

    if compare_versions(local_ver, remote_ver):
        log(f"Update available: {local_ver} -> {remote_ver}")
        log("Auto-updating...")

        # Check if tkinter is available for progress dialog
        if not TKINTER_AVAILABLE:
            # Update without GUI
            log("Downloading update (no GUI)...")
            if download_update():
                update_local_version(remote_ver)
                log("Update completed successfully!")
            else:
                log("Update failed - continuing with current version")
            return True

        try:
            # Show progress dialog and auto-update
            progress = ProgressDialog()
            success = progress.run_update(remote_ver)

            if success:
                log("Update completed successfully")
            else:
                log("Update failed - continuing with current version")

        except Exception as e:
            log(f"Update error: {e}")
            log("Continuing with current version...")
    else:
        log(f"Already at latest version: {local_ver}")

    return True


def launch_app():
    """Launch the main application"""
    log(f"Launching application: {APP_SCRIPT}")

    if not os.path.exists(APP_SCRIPT):
        error_msg = f"Application script not found:\n{APP_SCRIPT}"
        log(error_msg)
        messagebox.showerror("Error", error_msg)
        return False

    try:
        # Import using importlib
        import importlib.util

        spec = importlib.util.spec_from_file_location("unixautoedit", APP_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules["unixautoedit"] = module

        log("Loading module...")
        spec.loader.exec_module(module)

        # Run main function
        if hasattr(module, "main"):
            log("Starting main()...")
            module.main()
        else:
            log("No main() function found, module loaded only")

        return True

    except Exception as e:
        error_msg = f"Failed to launch application:\n{e}\n\n{traceback.format_exc()}"
        log(error_msg)
        messagebox.showerror("Error", error_msg)
        return False


def main():
    """Main entry point"""
    print("=" * 55)
    print("  Uni-x Auto Edit Launcher")
    print("=" * 55)

    # Setup paths and environment
    setup_environment()

    # Check for updates (with GUI)
    check_and_update()

    # Launch the main application
    launch_app()


if __name__ == "__main__":
    main()
