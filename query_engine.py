"""
Query Engine — End-to-end semantic search pipeline.
Embeds the user query, searches FAISS indices, extracts clips, and optionally
generates a natural language response via Google Gemini.
"""

import glob
import json
import os

from embedding_engine import embed_query_text, embed_query_clip
from vector_store import (
    load_index, load_metadata, combined_search,
)
from clip_extractor import extract_clips
from utils import logger, format_timestamp, clean_clips_dir


def search_and_extract(
    query: str,
    video_path: str,
    paths: dict,
    config: dict,
) -> dict:
    """
    Full query pipeline:
    1. Embed the query (text + CLIP)
    2. Search both FAISS indices
    3. Fuse results (reciprocal rank fusion)
    4. Cut video clips
    5. Optionally generate LLM summary

    Returns:
        {
            "results": [{start_sec, end_sec, start_tc, end_tc, score, text, clip_path}, ...],
            "summary": str,
            "clips": [clip_path, ...],
        }
    """
    top_k = config.get("top_k_results", 5)

    # ── Step 1: Embed the query ──────────────────────────────────────
    logger.info("Embedding query: '%s'", query)
    text_model = config.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
    clip_model = config.get("clip_model", "sentence-transformers/clip-ViT-B-32")

    text_query_emb = embed_query_text(query, text_model)
    visual_query_emb = embed_query_clip(query, clip_model)

    # ── Step 2: Load indices and metadata ────────────────────────────
    text_index = None
    visual_index = None
    text_metadata = []
    visual_metadata = []

    if os.path.exists(paths["text_index"]):
        text_index = load_index(paths["text_index"])
    if os.path.exists(paths["visual_index"]):
        visual_index = load_index(paths["visual_index"])

    metadata_path = paths["metadata"]
    if os.path.exists(metadata_path):
        all_metadata = load_metadata(metadata_path)
        text_metadata = [m for m in all_metadata if m.get("source") == "text"]
        visual_metadata = [m for m in all_metadata if m.get("source") == "visual"]

    # ── Step 3: Search and fuse ──────────────────────────────────────
    logger.info("Searching indices (top_k=%d)...", top_k)
    results = combined_search(
        text_index=text_index,
        visual_index=visual_index,
        text_metadata=text_metadata,
        visual_metadata=visual_metadata,
        text_query_emb=text_query_emb,
        visual_query_emb=visual_query_emb,
        top_k=top_k,
    )

    if not results:
        logger.warning("No results found for query: '%s'", query)
        return {"results": [], "summary": "No matching segments found.", "clips": []}

    logger.info("Found %d result(s).", len(results))

    # ── Step 4: Extract clips ────────────────────────────────────────
    clean_clips_dir()
    padding = config.get("clip_padding_seconds", 3)
    max_dur = config.get("max_clip_duration", 60)
    clips = extract_clips(video_path, results, padding_sec=padding, max_duration=max_dur)

    # ── Step 5: Generate LLM summary (optional) ─────────────────────
    summary = _build_summary(query, results)
    llm_result = _generate_llm_response(query, results, config, paths)
    if llm_result and llm_result.get("text"):
        summary = llm_result["text"]
    elif llm_result and llm_result.get("error"):
        summary += f"\n\n⚠️ **AI Analysis unavailable:** {llm_result['error']}"

    return {
        "results": clips if clips else results,
        "summary": summary,
        "clips": [c["clip_path"] for c in clips],
    }


def _build_summary(query: str, results: list[dict]) -> str:
    """Build a simple text summary of the search results."""
    lines = [f"**Found {len(results)} matching segment(s) for:** \"{query}\"\n"]
    for i, r in enumerate(results, 1):
        start_tc = format_timestamp(r["start_sec"])
        end_tc = format_timestamp(r["end_sec"])
        score = r.get("score", 0)
        text_preview = r.get("text", "")[:150]
        lines.append(f"**Clip {i}:** {start_tc} → {end_tc}  (relevance: {score:.3f})")
        if text_preview:
            lines.append(f"  > _{text_preview}..._")
        lines.append("")
    return "\n".join(lines)


def _collect_keyframes(results: list[dict], paths: dict, max_frames: int = 5) -> list[str]:
    """
    Collect keyframe image paths relevant to the search results.
    Looks at frame_path in results, then falls back to the frames directory.
    """
    frame_paths = []

    # 1. Collect frames directly referenced in results
    for r in results:
        fp = r.get("frame_path", "")
        if fp and os.path.isfile(fp) and fp not in frame_paths:
            frame_paths.append(fp)
        if len(frame_paths) >= max_frames:
            break

    # 2. If not enough, grab frames from the processed frames directory
    if len(frame_paths) < max_frames and paths:
        frames_dir = paths.get("frames", "")
        if frames_dir and os.path.isdir(frames_dir):
            all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            for fp in all_frames:
                if fp not in frame_paths:
                    frame_paths.append(fp)
                if len(frame_paths) >= max_frames:
                    break

    return frame_paths


def _generate_llm_response(
    query: str, results: list[dict], config: dict, paths: dict = None,
) -> dict | None:
    """
    Use Google Gemini (multimodal) to generate a natural language response.
    Sends keyframe images + transcript context so Gemini can visually analyze
    the video content and answer factual questions.

    Returns dict with 'text' on success or 'error' on failure, or None if no API key.
    """
    api_key = config.get("gemini_api_key", "")
    if not api_key:
        return {"error": "No Gemini API key configured. Add GEMINI_API_KEY to your .env file."}

    try:
        from PIL import Image
        from google import genai

        client = genai.Client(api_key=api_key)

        # ── Build text context from results ──────────────────────────
        context_parts = []
        for i, r in enumerate(results, 1):
            start_tc = format_timestamp(r["start_sec"])
            end_tc = format_timestamp(r["end_sec"])
            text = r.get("text", "N/A")
            context_parts.append(f"Segment {i} ({start_tc} → {end_tc}): {text}")

        context = "\n".join(context_parts)

        # ── Collect keyframe images ──────────────────────────────────
        frame_paths = _collect_keyframes(results, paths, max_frames=5)
        images = []
        for fp in frame_paths:
            try:
                img = Image.open(fp).convert("RGB")
                # Resize large images to save bandwidth / tokens
                max_side = 768
                if max(img.size) > max_side:
                    img.thumbnail((max_side, max_side), Image.LANCZOS)
                images.append(img)
            except Exception as img_err:
                logger.warning("Failed to load frame %s: %s", fp, img_err)

        # ── Build the multimodal prompt ──────────────────────────────
        has_images = len(images) > 0

        if has_images:
            prompt_text = f"""You are an expert video analysis assistant with vision capabilities.

The user uploaded a video and asked: "{query}"

Below are keyframe images extracted from the most relevant segments of the video, followed by the transcript of those segments.

**Transcript context:**
{context}

**Instructions:**
- Carefully analyze the attached images AND the transcript to answer the user's question.
- If the question asks about a specific person, player, object, text on screen, or activity, look at the images to identify them.
- If you can see names, logos, scoreboards, jersey numbers, or any identifying text in the images, mention them.
- Answer the question DIRECTLY and specifically. Do not just say "the video shows bowling" — instead identify the specific people, teams, or context visible.
- Mention relevant timestamps from the transcript.
- Keep your answer concise (3-6 sentences). Use markdown formatting.
- If you cannot determine the answer from the available images and transcript, say so honestly."""
        else:
            prompt_text = f"""You are a helpful video analysis assistant.

The user uploaded a video and asked: "{query}"

Here are the relevant transcript segments found:
{context}

**Instructions:**
- Answer the user's question directly based on the transcript.
- Mention relevant timestamps.
- Keep it concise (3-5 sentences). Use markdown formatting.
- If the transcript doesn't contain enough information to answer, say so honestly."""

        # ── Build contents list (images + text) ─────────────────────
        contents = []
        for img in images:
            contents.append(img)
        contents.append(prompt_text)

        # ── Try multiple models with fallback ────────────────────────
        models_to_try = [
            config.get("gemini_model", "gemini-2.0-flash"),
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ]
        # Deduplicate while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        last_error = None
        for model_name in models_to_try:
            try:
                logger.info("Trying Gemini model: %s (with %d images)", model_name, len(images))
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                logger.info("Gemini response received from %s.", model_name)
                return {"text": response.text}
            except Exception as model_err:
                last_error = model_err
                logger.warning("Model %s failed: %s", model_name, model_err)
                continue

        # All models failed
        error_msg = str(last_error)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {"error": "Gemini API quota exhausted. Please wait or upgrade your API plan at https://ai.google.dev"}
        return {"error": f"Gemini API call failed: {error_msg[:200]}"}

    except Exception as e:
        logger.warning("Gemini LLM call failed: %s", e)
        return {"error": f"Gemini API error: {str(e)[:200]}"}

