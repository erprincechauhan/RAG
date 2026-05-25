"""
Video-RAG Event Extraction Chatbot — Main Gradio Application
Upload long videos, ask natural language questions, and receive auto-extracted clips.
"""

import os
import json
import shutil
import gradio as gr
import numpy as np
from dotenv import load_dotenv

from utils import (
    logger, get_video_hash, ensure_dirs, format_timestamp,
    UPLOADS_DIR, CLIPS_DIR, clean_clips_dir,
)
from video_processor import process_video
from embedding_engine import embed_texts, embed_images
from vector_store import build_index, save_index, save_metadata
from query_engine import search_and_extract

# ── Load Config ──────────────────────────────────────────────────────
load_dotenv()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# .env → config.json override: API key flows from .env into CONFIG
env_api_key = os.getenv("GEMINI_API_KEY")
if env_api_key:
    CONFIG["gemini_api_key"] = env_api_key

# ── Global State ─────────────────────────────────────────────────────
app_state = {
    "video_path": None,
    "video_hash": None,
    "paths": None,
    "processed": False,
    "processing": False,
    "metadata": None,
}


# ── Video Upload & Processing ────────────────────────────────────────

def handle_video_upload(video_file, progress=gr.Progress()):
    """Handle video upload → full processing pipeline."""
    if video_file is None:
        return (
            "⚠️ No video uploaded.",
            gr.update(interactive=False),
        )

    if app_state["processing"]:
        return (
            "⏳ Already processing a video. Please wait...",
            gr.update(interactive=False),
        )

    app_state["processing"] = True
    app_state["processed"] = False

    try:
        # Copy uploaded file to uploads dir
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        video_name = os.path.basename(video_file)
        dest_path = os.path.join(UPLOADS_DIR, video_name)
        if video_file != dest_path:
            shutil.copy2(video_file, dest_path)

        progress(0.02, desc="Computing video hash...")
        video_hash = get_video_hash(dest_path)
        paths = ensure_dirs(video_hash)

        # Check if already processed
        if os.path.exists(paths["text_index"]) and os.path.exists(paths["metadata"]):
            logger.info("Video already processed (hash=%s). Loading cached data.", video_hash)
            app_state["video_path"] = dest_path
            app_state["video_hash"] = video_hash
            app_state["paths"] = paths
            app_state["processed"] = True
            app_state["processing"] = False

            return (
                f"✅ **Video loaded from cache!**\n\n"
                f"- **Hash:** `{video_hash}`\n"
                f"- **File:** {video_name}\n\n"
                f"💬 You can now ask questions about the video!",
                gr.update(interactive=True),
            )

        # ── Run the processing pipeline ──────────────────────────────
        progress(0.05, desc="Starting video processing pipeline...")

        metadata = process_video(
            video_path=dest_path,
            video_hash=video_hash,
            config=CONFIG,
            progress_callback=progress,
        )

        # ── Generate embeddings ─────────────────────────────────────
        progress(0.82, desc="Generating text embeddings...")
        chunks = metadata["chunks"]
        text_model = CONFIG.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
        clip_model = CONFIG.get("clip_model", "sentence-transformers/clip-ViT-B-32")

        # Text embeddings from transcript chunks
        text_entries = [c for c in chunks if c.get("text")]
        if text_entries:
            texts = [c["text"] for c in text_entries]
            text_embeddings = embed_texts(texts, text_model)
            text_index = build_index(text_embeddings)
            save_index(text_index, paths["text_index"])
        else:
            text_entries = []

        # Visual embeddings from keyframes
        progress(0.90, desc="Generating visual embeddings...")
        frames_info = metadata["frames_info"]
        if frames_info:
            frame_paths = [f["frame_path"] for f in frames_info]
            visual_embeddings = embed_images(frame_paths, clip_model)
            visual_index = build_index(visual_embeddings)
            save_index(visual_index, paths["visual_index"])
        else:
            frames_info = []

        # ── Save unified metadata ───────────────────────────────────
        progress(0.95, desc="Saving metadata...")
        all_metadata = []

        # Text metadata (parallel to text FAISS index)
        for c in text_entries:
            all_metadata.append({
                "start_sec": c["start_sec"],
                "end_sec": c["end_sec"],
                "text": c["text"],
                "scene_index": c.get("scene_index", -1),
                "source": "text",
            })

        # Visual metadata (parallel to visual FAISS index)
        for fi in frames_info:
            scene_idx = fi["scene_index"]
            # Find matching scene for time range
            scene = next((s for s in metadata["scenes"] if s["index"] == scene_idx), None)
            all_metadata.append({
                "start_sec": scene["start_sec"] if scene else fi["timestamp_sec"] - 5,
                "end_sec": scene["end_sec"] if scene else fi["timestamp_sec"] + 5,
                "text": "",
                "scene_index": scene_idx,
                "frame_path": fi["frame_path"],
                "source": "visual",
            })

        save_metadata(all_metadata, paths["metadata"])

        # ── Update state ─────────────────────────────────────────────
        app_state["video_path"] = dest_path
        app_state["video_hash"] = video_hash
        app_state["paths"] = paths
        app_state["metadata"] = metadata
        app_state["processed"] = True
        app_state["processing"] = False

        n_scenes = len(metadata["scenes"])
        n_segments = len(metadata["raw_segments"])
        n_chunks = len(text_entries)
        n_frames = len(frames_info)

        progress(1.0, desc="Processing complete!")

        status_msg = (
            f"✅ **Video processed successfully!**\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Scenes detected | {n_scenes} |\n"
            f"| Transcript segments | {n_segments} |\n"
            f"| Searchable chunks | {n_chunks} |\n"
            f"| Keyframes indexed | {n_frames} |\n"
            f"| Video hash | `{video_hash}` |\n\n"
            f"💬 **You can now ask questions about the video!**"
        )

        return (
            status_msg,
            gr.update(interactive=True),
        )

    except Exception as e:
        app_state["processing"] = False
        logger.exception("Video processing failed")
        return (
            f"❌ **Processing failed:** {str(e)}",
            gr.update(interactive=False),
        )


# ── Chat Handler ─────────────────────────────────────────────────────

def handle_chat(message: str, history: list):
    """Handle user query → semantic search → clip extraction."""
    if not app_state["processed"]:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please upload and process a video first before asking questions."})
        return history, None, ""

    if not message.strip():
        return history, None, ""

    # Add user message
    history.append({"role": "user", "content": message})

    try:
        result = search_and_extract(
            query=message,
            video_path=app_state["video_path"],
            paths=app_state["paths"],
            config=CONFIG,
        )

        summary = result["summary"]
        clips = result.get("clips", [])

        # Build response with clip info
        response = summary
        if clips:
            response += f"\n\n🎬 **{len(clips)} clip(s) extracted and ready to play!**\n"
            response += "_Check the clip player below to watch them._"

        history.append({"role": "assistant", "content": response})

        # Return first clip for auto-play
        first_clip = clips[0] if clips else None
        return history, first_clip, ""

    except Exception as e:
        logger.exception("Query failed")
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, None, ""


# ── Build Gradio UI ──────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global Reset & Dark Base ───────────────────────────────── */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #16161f;
    --bg-elevated: #1c1c28;
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-accent: rgba(139, 92, 246, 0.25);
    --text-primary: #e8e8ed;
    --text-secondary: #8b8b9e;
    --text-muted: #5a5a6e;
    --accent-violet: #8b5cf6;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --glow-violet: rgba(139, 92, 246, 0.15);
    --glow-blue: rgba(59, 130, 246, 0.12);
    --gradient-main: linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
}

/* Force dark background everywhere */
html, body, .gradio-container, .main, .contain,
.gradio-container .main .wrap,
.gradio-container > *,
gradio-app, gradio-app > div {
    background: var(--bg-primary) !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Force side-by-side columns */
.main-layout {
    display: flex !important;
    flex-direction: row !important;
    gap: 20px !important;
    flex-wrap: nowrap !important;
}
.main-layout > div {
    flex-shrink: 1 !important;
}
@media (max-width: 768px) {
    .main-layout {
        flex-direction: column !important;
    }
}

/* ── Animated Header ─────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0f0520 0%, #1a0a3e 30%, #0d1b3e 60%, #091520 100%);
    border-radius: var(--radius-xl);
    padding: 40px 48px;
    margin-bottom: 24px;
    border: 1px solid var(--border-accent);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(139, 92, 246, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 30%, rgba(59, 130, 246, 0.06) 0%, transparent 50%);
    animation: aurora 8s ease-in-out infinite;
    pointer-events: none;
}
@keyframes aurora {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    25% { transform: translate(2%, -1%) rotate(1deg); }
    50% { transform: translate(-1%, 2%) rotate(-1deg); }
    75% { transform: translate(1%, 1%) rotate(0.5deg); }
}
.hero-header h1 {
    background: linear-gradient(135deg, #c4b5fd 0%, #93c5fd 40%, #67e8f9 70%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.hero-header .hero-sub {
    color: #7c7c96;
    font-size: 0.95rem;
    margin: 0;
    position: relative;
    z-index: 1;
    letter-spacing: 0.3px;
}
.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 16px;
    position: relative;
    z-index: 1;
    flex-wrap: wrap;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid rgba(255,255,255,0.08);
}
.badge-violet { background: rgba(139,92,246,0.12); color: #c4b5fd; }
.badge-blue { background: rgba(59,130,246,0.12); color: #93c5fd; }
.badge-cyan { background: rgba(6,182,212,0.12); color: #67e8f9; }
.badge-emerald { background: rgba(16,185,129,0.12); color: #6ee7b7; }

/* ── Section Headers ─────────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-muted);
    margin-bottom: 12px;
    padding-left: 2px;
}
.section-label .label-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
}
.dot-violet { background: var(--accent-violet); box-shadow: 0 0 8px var(--glow-violet); }
.dot-blue { background: var(--accent-blue); box-shadow: 0 0 8px var(--glow-blue); }
.dot-cyan { background: var(--accent-cyan); box-shadow: 0 0 8px rgba(6,182,212,0.3); }

/* ── Glass Card Wrapper ──────────────────────────────────────── */
.glass-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 20px !important;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: var(--border-accent) !important;
    box-shadow: 0 4px 24px rgba(139, 92, 246, 0.06) !important;
}

/* ── Status Card ─────────────────────────────────────────────── */
.status-display {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
    min-height: 80px;
    color: var(--text-secondary) !important;
}
.status-display p, .status-display span, .status-display div {
    color: var(--text-secondary) !important;
}

/* ── Force dark on ALL Gradio internals ──────────────────────── */
.block, .form, .wrap, .panel, .tabs, .tab-nav,
div[class*="block"], div[class*="form"], div[class*="wrap"],
div[class*="panel"], .contain > div {
    background: transparent !important;
    border-color: var(--border-subtle) !important;
}

/* Chatbot */
.chatbot-container .wrap,
.chatbot-container .message-wrap,
div[data-testid="chatbot"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
}
div[data-testid="chatbot"] .message {
    border-radius: var(--radius-md) !important;
}
div[data-testid="chatbot"] .bot {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}
div[data-testid="chatbot"] .user {
    background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(59,130,246,0.12)) !important;
    color: var(--text-primary) !important;
}

/* Text inputs */
textarea, input[type="text"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent-violet) !important;
    box-shadow: 0 0 0 3px var(--glow-violet), 0 0 20px rgba(139,92,246,0.08) !important;
    outline: none !important;
}
textarea::placeholder, input::placeholder {
    color: var(--text-muted) !important;
}

/* Upload zone */
div[data-testid="file"], .upload-btn {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(139, 92, 246, 0.2) !important;
    border-radius: var(--radius-lg) !important;
    transition: all 0.3s ease !important;
    color: var(--text-secondary) !important;
}
div[data-testid="file"]:hover, .upload-btn:hover {
    border-color: rgba(139, 92, 246, 0.5) !important;
    background: rgba(139, 92, 246, 0.04) !important;
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.06) !important;
}
div[data-testid="file"] span,
div[data-testid="file"] p,
div[data-testid="file"] div {
    color: var(--text-secondary) !important;
}

/* Buttons — Primary */
.btn-primary, button.primary {
    background: linear-gradient(135deg, #7c3aed, #6366f1, #3b82f6) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.2) !important;
}
.btn-primary:hover, button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.35) !important;
}
.btn-primary:active, button.primary:active {
    transform: translateY(0) !important;
}

/* Buttons — Secondary */
button.secondary {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
button.secondary:hover {
    border-color: var(--border-accent) !important;
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
}

/* Video player */
video, .video-container, div[data-testid="video"] {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-subtle) !important;
    overflow: hidden;
    background: var(--bg-secondary) !important;
}

/* Labels */
label, .label-wrap, span.svelte-1gfkn6j {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

/* Markdown text */
.prose, .prose p, .prose li, .markdown-text, .md {
    color: var(--text-primary) !important;
}

/* Tables in markdown */
.prose table, .md table {
    border-collapse: collapse !important;
}
.prose th, .md th {
    background: var(--bg-elevated) !important;
    color: var(--accent-violet) !important;
    padding: 8px 12px !important;
    border-bottom: 2px solid var(--border-accent) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}
.prose td, .md td {
    padding: 6px 12px !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a52; }

/* Footer */
footer { opacity: 0.4 !important; }
footer:hover { opacity: 0.8 !important; transition: opacity 0.3s; }

/* Hide default Gradio markdown headings (we use custom HTML labels) */
.hide-default-label > label { display: none !important; }
"""

def create_app():
    with gr.Blocks(
        title="Video-RAG Event Extraction Chatbot",
    ) as app:

        # ── Hero Header ────────────────────────────────────────────
        gr.HTML("""
        <div class="hero-header">
            <h1>🎬 Video-RAG Event Extraction</h1>
            <p class="hero-sub">Upload a long video · Ask natural language questions · Get auto-extracted clips with AI analysis</p>
            <div class="hero-badges">
                <span class="hero-badge badge-violet">⚡ CLIP Visual Search</span>
                <span class="hero-badge badge-blue">🎤 Whisper Transcription</span>
                <span class="hero-badge badge-cyan">🧠 Gemini Multimodal AI</span>
                <span class="hero-badge badge-emerald">✂️ Auto Clip Extraction</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False, elem_classes=["main-layout"]):
            # ── LEFT COLUMN: Upload & Status ────────────────────────
            with gr.Column(scale=1, min_width=280):

                gr.HTML('<div class="section-label"><span class="label-dot dot-violet"></span>VIDEO INPUT</div>')
                video_input = gr.File(
                    label="Drop your video here",
                    file_count="single",
                    file_types=["video"],
                    height=220,
                )
                process_btn = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg",
                )

                gr.HTML('<div class="section-label" style="margin-top:20px;"><span class="label-dot dot-cyan"></span>STATUS</div>')
                status_output = gr.Markdown(
                    value="Upload a video to get started.",
                    elem_classes=["status-display"],
                )

                # ── Clip Player (left column, below status) ─────────
                gr.HTML('<div class="section-label" style="margin-top:20px;"><span class="label-dot dot-blue"></span>CLIP PLAYER</div>')
                clip_player = gr.Video(
                    label="Result Clip",
                    interactive=False,
                    height=240,
                    autoplay=True,
                )

            # ── RIGHT COLUMN: Chat ──────────────────────────────────
            with gr.Column(scale=2, min_width=400):

                gr.HTML('<div class="section-label"><span class="label-dot dot-blue"></span>CONVERSATION</div>')
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=520,
                    placeholder="Upload and process a video, then ask questions here…",
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask about the video… e.g. 'Who is the bowler?' or 'Show me action scenes'",
                        show_label=False,
                        scale=6,
                        interactive=False,
                        lines=1,
                        max_lines=3,
                    )
                    send_btn = gr.Button(
                        "🔍 Search",
                        variant="primary",
                        scale=1,
                        interactive=False,
                    )

                with gr.Row():
                    clear_btn = gr.Button(
                        "🗑️ Clear Chat",
                        variant="secondary",
                        size="sm",
                    )

        # ── Event Handlers ──────────────────────────────────────────
        process_btn.click(
            fn=handle_video_upload,
            inputs=[video_input],
            outputs=[status_output, query_input],
            show_progress="full",
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=[send_btn],
        )

        query_input.submit(
            fn=handle_chat,
            inputs=[query_input, chatbot],
            outputs=[chatbot, clip_player, query_input],
        )

        send_btn.click(
            fn=handle_chat,
            inputs=[query_input, chatbot],
            outputs=[chatbot, clip_player, query_input],
        )

        clear_btn.click(
            fn=lambda: ([], None, ""),
            outputs=[chatbot, clip_player, query_input],
        )

    return app


# ── Entry Point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Video-RAG Event Extraction Chatbot...")
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Glass(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
    )