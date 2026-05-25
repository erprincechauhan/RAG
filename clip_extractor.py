"""
Clip Extractor — Cut video segments using FFmpeg.
Handles padding, overlap merging, and browser-compatible H.264 output.
"""

import os
import subprocess

from utils import logger, get_ffmpeg_path, CLIPS_DIR, format_timestamp


def extract_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> str:
    
    """
    Extract a single clip from video_path between start_sec and end_sec.
    Uses fast seeking (-ss before -i) and H.264/AAC encoding for browser compatibility.
    """
    ffmpeg = get_ffmpeg_path()
    duration = end_sec - start_sec

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start_sec:.2f}",
        "-i", video_path,
        "-t", f"{duration:.2f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",  # Enable streaming
        output_path,
    ]

    logger.info("Cutting clip: %s → %s (%.1fs)", format_timestamp(start_sec), format_timestamp(end_sec), duration)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path


def extract_clips(
    video_path: str,
    timestamps: list[dict],
    padding_sec: float = 3.0,
    max_duration: float = 60.0,
) -> list[dict]:
    """
    Extract multiple clips from a video based on timestamp results.

    Args:
        video_path: Path to the source video.
        timestamps: List of {start_sec, end_sec, score, text, ...} from the query engine.
        padding_sec: Seconds to add before/after each clip for context.
        max_duration: Maximum clip duration in seconds.

    Returns:
        List of {clip_path, start_sec, end_sec, start_tc, end_tc, score, text}
    """
    if not timestamps:
        return []

    # Add padding and clamp
    padded = []
    for ts in timestamps:
        start = max(0, ts["start_sec"] - padding_sec)
        end = ts["end_sec"] + padding_sec
        duration = end - start
        if duration > max_duration:
            end = start + max_duration
        padded.append({
            "start_sec": start,
            "end_sec": end,
            "score": ts.get("score", 0),
            "text": ts.get("text", ""),
        })

    # Merge overlapping padded regions
    padded.sort(key=lambda x: x["start_sec"])
    merged = [padded[0].copy()]
    for item in padded[1:]:
        last = merged[-1]
        if item["start_sec"] <= last["end_sec"]:
            last["end_sec"] = max(last["end_sec"], item["end_sec"])
            last["score"] = max(last["score"], item["score"])
            if item["text"] and item["text"] not in last["text"]:
                last["text"] = (last["text"] + " " + item["text"]).strip()
        else:
            merged.append(item.copy())

    # Cut each merged region
    results = []
    os.makedirs(CLIPS_DIR, exist_ok=True)

    for i, region in enumerate(merged):
        clip_name = f"clip_{i + 1:03d}_{int(region['start_sec'])}s_{int(region['end_sec'])}s.mp4"
        clip_path = os.path.join(CLIPS_DIR, clip_name)

        try:
            extract_clip(video_path, region["start_sec"], region["end_sec"], clip_path)
            results.append({
                "clip_path": clip_path,
                "start_sec": region["start_sec"],
                "end_sec": region["end_sec"],
                "start_tc": format_timestamp(region["start_sec"]),
                "end_tc": format_timestamp(region["end_sec"]),
                "score": region["score"],
                "text": region["text"],
            })
        except subprocess.CalledProcessError as e:
            logger.error("Failed to cut clip %s: %s", clip_name, e)

    logger.info("Extracted %d clip(s).", len(results))
    return results
