# app.py
# Hugging Face Spaces entry point for DocuSage.
# HF Spaces requires this exact filename and a top-level launch() call.
# This stub is replaced progressively as pipeline components are built.
from __future__ import annotations
import os, re
import html as html_module

import gradio as gr

from config import get_logger, settings  # Triggers logging + dir setup
from pipeline import RAGPipeline, PipelineResult
from audit.models import SentenceStatus

logger = get_logger(__name__)

# Pipeline singleton (indices eager, weights lazy)
pipeline = RAGPipeline()
logger.info("RAGPipeline initialised; model weights load on first query.")

ALL_RING_LABELS = [
    "Market Investments",
    "Govt Schemes",
    "Banking & RBI",
    "Foreign Investments",
]

STATUS_COLOURS: dict[str, str] = {
    "SUPPORTED":    "rgba(34, 197, 94, 0.20)",
    "UNCERTAIN":    "rgba(234, 179, 8, 0.22)",
    "CONTRADICTED": "rgba(239, 68, 68, 0.30)",
    "UNSUPPORTED":  "transparent",
}

STATUS_LABELS: dict[str, str] = {
    "SUPPORTED":    "Verified",
    "UNCERTAIN":    "Uncertain",
    "CONTRADICTED": "Contradicted",
    "UNSUPPORTED":  "Uncited",
}

# app.py -- Part 2: Answer HTML renderer (append after Part 1)


def _escape(text: str) -> str:
    return html_module.escape(text)


def _truncate(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    t = text[:max_chars]
    last_sp = t.rfind(" ")
    return (t[:last_sp] + "...") if last_sp > 0 else (t + "...")


def render_answer(result: PipelineResult) -> str:
    if result.is_error:
        return (
            '<div class="error-card">'
            '<span class="error-icon">&#9888;</span>'
            f'<span class="error-text">{_escape(result.error or "Unknown error")}</span>'
            '</div>'
        )
    if not result.answer:
        return '<div class="docusage-answer muted">No answer generated.</div>'

    answer_text = result.answer

    # Pass 1: colour-code sentences
    for sa in result.sentence_audits:
        colour = STATUS_COLOURS.get(sa.status, "transparent")
        badge  = STATUS_LABELS.get(sa.status, sa.status)
        if colour == "transparent":
            styled = _escape(sa.sentence)
        else:
            styled = (
                f'<span class="sent-audit" '
                f'style="background:{colour};border-radius:3px;padding:1px 0" '
                f'title="{badge}">{_escape(sa.sentence)}</span>'
            )
        answer_text = answer_text.replace(sa.sentence, styled, 1)

    # Pass 2: replace [N] with superscript cite badge
    def replace_marker(m):
        n = m.group(1)
        return f'<sup><span class="cite-badge" title="Citation {n}">[{n}]</span></sup>'

    answer_html = re.sub(r'\[(\d+)\]', replace_marker, answer_text)

    legend_items = "".join(
        f'<span class="legend-item" style="background:{STATUS_COLOURS[s]}">'
        f'{STATUS_LABELS[s]}</span>'
        for s in ("SUPPORTED", "UNCERTAIN", "CONTRADICTED")
    )
    legend = f'<div class="audit-legend">{legend_items}</div>'

    return f'<div class="docusage-answer">{legend}{answer_html}</div>'

# ── Stub pipeline function (replaced in SNIPPETS_09_FRONTEND.md) ───────────
def run_pipeline_stub(query: str, ring_filter: list) -> tuple:
    """
    Placeholder pipeline function.
    Returns informational messages until real pipeline is wired in.
    """
    logger.info("Stub pipeline called", query=query[:50])

    answer = (
        "🚧 **DocuSage is being built.**\n\n"
        "The full pipeline (BM25 + FAISS + RRF + Reranker + Qwen2.5 + "
        "Hallucination Audit) is being assembled. Check back soon!\n\n"
        f"**Your query:** {query}\n"
        f"**Rings selected:** {ring_filter or 'All'}"
    )
    confidence = "<div style='color: gray;'>⏳ Pipeline not yet loaded</div>"
    subqueries  = "_Query decomposition not yet available._"
    sources     = "_Source retrieval not yet available._"

    return answer, confidence, subqueries, sources  # 4-tuple for Gradio outputs


# ── Gradio UI stub (replaced in SNIPPETS_09_FRONTEND.md) ──────────────────
with gr.Blocks(
    title="DocuSage — Indian Personal Finance Intelligence",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# 🏦 DocuSage\n### Indian Personal Finance Intelligence")
    gr.Markdown(
        "> **Status:** Building... Full pipeline coming soon. "
        "See [MASTER_DOC.md](https://github.com/<YOUR_USERNAME>/docusage/blob/main/MASTER_DOC.md) "
        "for the full architecture."
    )

    with gr.Row():
        with gr.Column(scale=1):
            query_box = gr.Textbox(
                label="Your Question",
                placeholder="e.g. Should I invest in PPF or ELSS to save tax?",
                lines=4,
                max_lines=6,
            )
            ring_filter = gr.CheckboxGroup(
                choices=settings.get_ring_labels(),
                label="Filter by Knowledge Domain (optional)",
                value=[],  # default: all rings
            )
            submit_btn = gr.Button("Ask DocuSage", variant="primary")

        with gr.Column(scale=2):
            answer_out     = gr.Markdown(label="Answer")
            confidence_out = gr.HTML(label="Confidence")
            with gr.Accordion("Sub-queries explored", open=False):
                subqueries_out = gr.Markdown()
            with gr.Accordion("Sources", open=False):
                sources_out = gr.Markdown()

    # Wire submit button to stub pipeline
    submit_btn.click(
        fn=run_pipeline_stub,
        inputs=[query_box, ring_filter],
        outputs=[answer_out, confidence_out, subqueries_out, sources_out],
    )

logger.info(
    "DocuSage app initialised",
    device=settings.DEVICE,
    log_level=settings.LOG_LEVEL,
)

# ── Launch (required at module level for HF Spaces) ────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_port=settings.GRADIO_SERVER_PORT, 
        share=settings.GRADIO_SHARE,   # True in Colab, False locally
        show_api=False,
    )

