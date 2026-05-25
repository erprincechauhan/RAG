"""
Embedding Engine — Generate text and visual embeddings using sentence-transformers.
Uses all-MiniLM-L6-v2 for text and CLIP ViT-B/32 for images.
"""

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from utils import logger

# Module-level model cache to avoid reloading
_text_model = None
_clip_model = None



def _get_text_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _text_model
    if _text_model is None:
        logger.info("Loading text embedding model: %s", model_name)
        _text_model = SentenceTransformer(model_name)
        logger.info("Text model loaded (dim=%d).", _text_model.get_sentence_embedding_dimension())
    return _text_model


def _get_clip_model(model_name: str = "sentence-transformers/clip-ViT-B-32"):
    global _clip_model
    if _clip_model is None:
        logger.info("Loading CLIP model: %s", model_name)
        _clip_model = SentenceTransformer(model_name)
        logger.info("CLIP model loaded (dim=%d).", _clip_model.get_sentence_embedding_dimension())
    return _clip_model


# ── Text Embeddings ──────────────────────────────────────────────────

def embed_texts(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a list of text strings using all-MiniLM-L6-v2.
    Returns normalized np.ndarray of shape (n, 384).
    """
    model = _get_text_model(model_name)
    logger.info("Embedding %d text chunks...", len(texts))
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    logger.info("Text embeddings shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


# ── Visual Embeddings ────────────────────────────────────────────────

def embed_images(image_paths: list[str], model_name: str = "sentence-transformers/clip-ViT-B-32") -> np.ndarray:
    """
    Embed images using CLIP ViT-B/32.
    Returns normalized np.ndarray of shape (n, 512).
    """
    model = _get_clip_model(model_name)
    logger.info("Embedding %d images...", len(image_paths))
    images = [Image.open(p).convert("RGB") for p in image_paths]
    embeddings = model.encode(images, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    logger.info("Image embeddings shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


# ── Query Embeddings ─────────────────────────────────────────────────

def embed_query_text(query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a text query for transcript search. Returns (1, 384) array."""
    model = _get_text_model(model_name)
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)


def embed_query_clip(query: str, model_name: str = "sentence-transformers/clip-ViT-B-32") -> np.ndarray:
    """Embed a text query using CLIP for visual search. Returns (1, 512) array."""
    model = _get_clip_model(model_name)
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)
