# app.py
# Hugging Face Spaces entry point for DocuSage.
# HF Spaces requires this exact filename and a top-level launch() call.
# This stub is replaced progressively as pipeline components are built.

import gradio as gr

from config import get_logger, settings  # Triggers logging + dir setup

logger = get_logger(__name__)

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

