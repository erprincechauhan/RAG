"""
Shared utilities for Video-RAG Event Extraction Chatbot.
Provides path management, hashing, time formatting, and logging.
"""

import os
import hashlib
import logging
import shutil

# ── Directory constants ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")

# Create top-level dirs
for d in [UPLOADS_DIR, PROCESSED_DIR, CLIPS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video-rag")

# ── Ensure ffmpeg is in PATH for Whisper ─────────────────────────────
ffmpeg_dir = r"C:\ffmpeg\bin"
if os.path.exists(ffmpeg_dir) and ffmpeg_dir.lower() not in os.environ.get("PATH", "").lower():
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")



def get_video_hash(video_path: str) -> str:
    """Return a fast hash based on file size and filename to prevent I/O freezing."""
    try:
        file_size = os.path.getsize(video_path)
        file_name = os.path.basename(video_path)
        hash_string = f"{file_name}_{file_size}"
        return hashlib.sha256(hash_string.encode()).hexdigest()[:12]
    except Exception as e:
        logger.error("Hash error: %s", e)
        return "default_hash_001"


def ensure_dirs(video_hash: str) -> dict:
    """
    Create the processed/<hash>/frames/ directory structure.
    Returns dict of key paths.
    """
    root = os.path.join(PROCESSED_DIR, video_hash)
    frames = os.path.join(root, "frames")
    os.makedirs(frames, exist_ok=True)
    return {
        "root": root,
        "frames": frames,
        "audio": os.path.join(root, "audio.wav"),
        "scenes": os.path.join(root, "scenes.json"),
        "transcript": os.path.join(root, "transcript.json"),
        "text_index": os.path.join(root, "text_index.faiss"),
        "visual_index": os.path.join(root, "visual_index.faiss"),
        "metadata": os.path.join(root, "metadata.json"),
    }


def format_timestamp(seconds: float) -> str:
    """Convert seconds → HH:MM:SS string."""
    seconds = max(0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_ffmpeg_path() -> str:
    """Locate the ffmpeg binary. Returns 'ffmpeg' if on PATH, else raises."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    # Common Windows locations
    for candidate in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\ffmpeg\bin\ffmpeg.exe"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "ffmpeg not found. Install it and add to PATH, "
        "or place it in C:\\ffmpeg\\bin\\ffmpeg.exe"
    )


def clean_clips_dir():
    """Remove all previously generated clips."""
    if os.path.exists(CLIPS_DIR):
        shutil.rmtree(CLIPS_DIR)
    os.makedirs(CLIPS_DIR, exist_ok=True)
