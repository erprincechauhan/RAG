"""
Vector Store — FAISS index management for text and visual embeddings.
Supports building, searching, saving, loading, and combined multi-modal retrieval.
"""

import os
import json
import numpy as np
import faiss

from utils import logger


# ── Index Creation ───────────────────────────────────────────────────

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (inner product ≈ cosine similarity for normalized vectors).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index: %d vectors, dim=%d", index.ntotal, dim)
    return index


# ── Search ───────────────────────────────────────────────────────────

def search_index(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple]:
    """
    Search a FAISS index. Returns list of (score, index_position) sorted by score desc.
    """
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:  # FAISS returns -1 for missing entries
            results.append((float(score), int(idx)))
    return results


# ── Persistence ──────────────────────────────────────────────────────

def save_index(index: faiss.Index, path: str):
    """Save a FAISS index to disk."""
    faiss.write_index(index, path)
    logger.info("Saved FAISS index → %s", path)


def load_index(path: str) -> faiss.Index:
    """Load a FAISS index from disk."""
    index = faiss.read_index(path)
    logger.info("Loaded FAISS index ← %s (%d vectors)", path, index.ntotal)
    return index


# ── Metadata Management ─────────────────────────────────────────────

def save_metadata(metadata: list[dict], path: str):
    """Save the metadata list (parallel to FAISS index positions) as JSON."""
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata (%d entries) → %s", len(metadata), path)


def load_metadata(path: str) -> list[dict]:
    """Load metadata from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    logger.info("Loaded metadata (%d entries) ← %s", len(data), path)
    return data


# ── Combined Multi-Modal Search ─────────────────────────────────────

def combined_search(
    text_index: faiss.Index | None,
    visual_index: faiss.Index | None,
    text_metadata: list[dict],
    visual_metadata: list[dict],
    text_query_emb: np.ndarray | None,
    visual_query_emb: np.ndarray | None,
    top_k: int = 5,
    text_weight: float = 0.6,
    visual_weight: float = 0.4,
) -> list[dict]:
    """
    Perform reciprocal rank fusion across text and visual search results.
    Returns unified ranked list of {start_sec, end_sec, score, text, source, scene_index}.
    """
    rrf_k = 60  # RRF constant

    # Accumulate RRF scores keyed by (start_sec, end_sec) tuples
    score_map: dict[tuple, dict] = {}

    def add_results(results, metadata_list, weight, source_type):
        for rank, (raw_score, idx) in enumerate(results):
            if idx < 0 or idx >= len(metadata_list):
                continue
            meta = metadata_list[idx]
            key = (round(meta["start_sec"], 1), round(meta["end_sec"], 1))
            rrf_score = weight / (rrf_k + rank + 1)

            if key not in score_map:
                score_map[key] = {
                    "start_sec": meta["start_sec"],
                    "end_sec": meta["end_sec"],
                    "score": 0.0,
                    "text": meta.get("text", ""),
                    "source": source_type,
                    "scene_index": meta.get("scene_index", -1),
                }
            score_map[key]["score"] += rrf_score
            # If visual source provides frame path
            if "frame_path" in meta:
                score_map[key]["frame_path"] = meta["frame_path"]

    # Text search
    if text_index is not None and text_query_emb is not None and text_index.ntotal > 0:
        text_results = search_index(text_index, text_query_emb, top_k=top_k * 2)
        add_results(text_results, text_metadata, text_weight, "text")

    # Visual search
    if visual_index is not None and visual_query_emb is not None and visual_index.ntotal > 0:
        visual_results = search_index(visual_index, visual_query_emb, top_k=top_k * 2)
        add_results(visual_results, visual_metadata, visual_weight, "visual")

    # Sort by fused score descending
    ranked = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)

    # Merge overlapping / nearby timestamps (within 10s)
    merged = _merge_nearby(ranked, gap_threshold=10.0)

    return merged[:top_k]


def _merge_nearby(results: list[dict], gap_threshold: float = 10.0) -> list[dict]:
    """Merge results whose time windows overlap or are within gap_threshold seconds."""
    if not results:
        return results

    # Sort by start time for merging
    sorted_results = sorted(results, key=lambda x: x["start_sec"])
    merged = [sorted_results[0].copy()]

    for item in sorted_results[1:]:
        last = merged[-1]
        if item["start_sec"] <= last["end_sec"] + gap_threshold:
            # Merge: expand time window, accumulate score, join text
            last["end_sec"] = max(last["end_sec"], item["end_sec"])
            last["score"] = max(last["score"], item["score"])
            if item.get("text") and item["text"] not in last.get("text", ""):
                last["text"] = (last.get("text", "") + " " + item["text"]).strip()
        else:
            merged.append(item.copy())

    # Re-sort by score
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged
