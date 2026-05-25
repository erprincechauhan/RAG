"""
Video Processor — Scene detection, audio extraction, keyframe sampling, and transcription.
Orchestrates the full ingestion pipeline for uploaded videos.
"""

import os
import json
import subprocess
import cv2
import numpy as np
from scenedetect import detect, ContentDetector, open_video
import whisper

from utils import logger, ensure_dirs, get_ffmpeg_path, format_timestamp


# ── Scene Detection ──────────────────────────────────────────────────

def detect_scenes(video_path: str, threshold: float = 27.0) -> list[dict]:
    """
    Detect scene boundaries using PySceneDetect's ContentDetector.
    Returns list of {index, start_sec, end_sec, start_tc, end_tc}.
    """
    logger.info("Detecting scenes (threshold=%.1f)...", threshold)
    # Use PyAV backend for better MKV support instead of OpenCV
    video = open_video(video_path, backend='pyav')
    
    from scenedetect import SceneManager
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        scenes.append({
            "index": i,
            "start_sec": start.get_seconds(),
            "end_sec": end.get_seconds(),
            "start_tc": format_timestamp(start.get_seconds()),
            "end_tc": format_timestamp(end.get_seconds()),
        })
    logger.info("Detected %d scenes.", len(scenes))

    # If no scenes detected (e.g. very uniform video), treat entire video as 1 scene
    if not scenes:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps
        cap.release()
        scenes.append({
            "index": 0,
            "start_sec": 0.0,
            "end_sec": duration,
            "start_tc": "00:00:00",
            "end_tc": format_timestamp(duration),
        })
        logger.info("No scene cuts found — using full video as single scene (%.1fs).", duration)

    return scenes


# ── Audio Extraction ────────────────────────────────────────────────

def extract_audio(video_path: str, output_wav: str) -> str:
    """Extract audio from video → 16 kHz mono WAV for Whisper."""
    logger.info("Extracting audio → %s", output_wav)
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        output_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    logger.info("Audio extraction complete.")
    return output_wav


# ── Keyframe Extraction ─────────────────────────────────────────────

def extract_keyframes(video_path: str, scenes: list[dict], frames_dir: str) -> list[dict]:
    """
    Extract one representative frame per scene (at midpoint).
    Returns list of {frame_path, scene_index, timestamp_sec}.
    """
    logger.info("Extracting keyframes for %d scenes...", len(scenes))
    frames_info = []
    ffmpeg = get_ffmpeg_path()

    for scene in scenes:
        mid_sec = (scene["start_sec"] + scene["end_sec"]) / 2
        fname = f"frame_{scene['index']:05d}.jpg"
        fpath = os.path.join(frames_dir, fname)
        
        # Use fast seeking (-ss before -i) to exact mid_sec, extract 1 frame
        cmd = [
            ffmpeg, "-y", "-ss", str(mid_sec), "-i", video_path, 
            "-vframes", "1", "-q:v", "2", "-loglevel", "error", fpath
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(fpath):
            frames_info.append({
                "frame_path": fpath,
                "scene_index": scene["index"],
                "timestamp_sec": mid_sec,
            })

    logger.info("Extracted %d keyframes.", len(frames_info))
    return frames_info


# ── Whisper Transcription ───────────────────────────────────────────

def transcribe_audio(audio_path: str, model_name: str = "base") -> list[dict]:
    """
    Transcribe audio using Whisper. Returns segments with timestamps.
    Each segment: {text, start_sec, end_sec}.
    """
    logger.info("Loading Whisper model '%s'...", model_name)
    model = whisper.load_model(model_name)

    logger.info("Transcribing audio (this may take a while for long videos)...")
    result = model.transcribe(audio_path, word_timestamps=True, verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "text": seg["text"].strip(),
            "start_sec": seg["start"],
            "end_sec": seg["end"],
        })

    logger.info("Transcription complete — %d segments.", len(segments))
    return segments


def align_transcript_to_scenes(segments: list[dict], scenes: list[dict]) -> list[dict]:
    """
    Assign each transcript segment to its best-matching scene.
    Adds 'scene_index' to each segment. Also creates merged scene-level chunks.
    """
    scene_chunks = []
    for scene in scenes:
        matching_text = []
        for seg in segments:
            seg_mid = (seg["start_sec"] + seg["end_sec"]) / 2
            if scene["start_sec"] <= seg_mid < scene["end_sec"]:
                matching_text.append(seg["text"])
                seg["scene_index"] = scene["index"]

        combined = " ".join(matching_text).strip()
        if combined:
            scene_chunks.append({
                "text": combined,
                "start_sec": scene["start_sec"],
                "end_sec": scene["end_sec"],
                "scene_index": scene["index"],
                "type": "scene_transcript",
            })

    # Also keep individual segments as fine-grained chunks
    fine_chunks = []
    for seg in segments:
        fine_chunks.append({
            "text": seg["text"],
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "scene_index": seg.get("scene_index", -1),
            "type": "segment",
        })

    return scene_chunks + fine_chunks


# ── Full Pipeline ───────────────────────────────────────────────────

def process_video(video_path: str, video_hash: str, config: dict,
                  progress_callback=None) -> dict:
    """
    Run the full video processing pipeline:
    1. Scene detection
    2. Audio extraction
    3. Keyframe extraction
    4. Whisper transcription
    5. Align transcript to scenes

    Returns metadata dict with all extracted info.
    """
    paths = ensure_dirs(video_hash)

    def update_progress(msg, pct):
        logger.info("[%d%%] %s", int(pct * 100), msg)
        if progress_callback:
            progress_callback(pct, desc=msg)

    # Step 1: Scene detection
    update_progress("Detecting scenes...", 0.05)
    threshold = config.get("scene_threshold", 27.0)
    scenes = detect_scenes(video_path, threshold)
    with open(paths["scenes"], "w") as f:
        json.dump(scenes, f, indent=2)

    # Step 2: Audio extraction
    update_progress("Extracting audio...", 0.15)
    extract_audio(video_path, paths["audio"])

    # Step 3: Keyframe extraction
    update_progress("Extracting keyframes...", 0.25)
    frames_info = extract_keyframes(video_path, scenes, paths["frames"])

    # Step 4: Transcription
    update_progress("Transcribing audio (this is the longest step)...", 0.30)
    whisper_model = config.get("whisper_model", "base")
    raw_segments = transcribe_audio(paths["audio"], whisper_model)

    # Step 5: Align transcript to scenes → create searchable chunks
    update_progress("Aligning transcript to scenes...", 0.75)
    chunks = align_transcript_to_scenes(raw_segments, scenes)

    # Save transcript
    transcript_data = {
        "raw_segments": raw_segments,
        "chunks": chunks,
    }
    with open(paths["transcript"], "w") as f:
        json.dump(transcript_data, f, indent=2)

    update_progress("Video processing complete!", 0.80)

    return {
        "video_hash": video_hash,
        "video_path": video_path,
        "paths": paths,
        "scenes": scenes,
        "frames_info": frames_info,
        "raw_segments": raw_segments,
        "chunks": chunks,
    }
