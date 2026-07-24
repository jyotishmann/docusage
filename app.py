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

# app.py -- Part 3: Citation cards (append)


def render_citations(result: PipelineResult) -> str:
    if result.is_error or not result.citations:
        return '<div class="muted">No citations for this response.</div>'

    cards = []
    for cit in result.citations:
        if cit.source_url:
            title_html = (
                f'<a href="{_escape(cit.source_url)}" target="_blank" '
                f'class="cite-title">{_escape(cit.doc_title)}</a>'
            )
        else:
            title_html = f'<span class="cite-title">{_escape(cit.doc_title)}</span>'

        meta_parts = [
            f'<span class="cite-meta">{_escape(cit.governing_body)}</span>',
            f'<span class="cite-ring">{_escape(cit.ring_label)}</span>',
        ]
        if cit.effective_date:
            meta_parts.append(
                f'<span class="cite-date">&#128197; {_escape(cit.effective_date)}</span>')
        if cit.circular_ref:
            meta_parts.append(
                f'<span class="cite-ref">&#128196; {_escape(cit.circular_ref)}</span>')

        meta_html = "  |  ".join(meta_parts)
        excerpt   = _escape(_truncate(cit.chunk.chunk_text))

        cards.append(
            f'<div class="cite-card">'
            f'<div class="cite-header">'
            f'<span class="cite-num">[{cit.marker}]</span>{title_html}</div>'
            f'<div class="cite-meta-row">{meta_html}</div>'
            f'<div class="cite-excerpt">{excerpt}</div>'
            f'</div>'
        )

    return '<div class="citations-panel">' + "".join(cards) + '</div>'

# app.py -- Part 4: Sub-queries and metrics renderers (append)

def render_subqueries(result: PipelineResult) -> str:
    if result.is_error:
        return ""
    decomp = ("Yes -- query was decomposed" if result.was_decomposed
               else "No -- handled directly")
    sub_q_html = "".join(
        f'<li class="subq-item">{_escape(q)}</li>'
        for q in result.sub_queries
    )
    rings = (", ".join(result.ring_filter) if result.ring_filter
             else "All domains")
    return (
        '<div class="subq-panel">'
        f'<div class="subq-row"><span class="subq-label">Decomposed:</span>'
        f'<span class="subq-value">{decomp}</span></div>'
        f'<div class="subq-row"><span class="subq-label">Reason:</span>'
        f'<span class="subq-value muted">{_escape(result.decomposition_reason)}</span></div>'
        f'<div class="subq-row"><span class="subq-label">Sub-queries:</span>'
        f'<ol class="subq-list">{sub_q_html}</ol></div>'
        f'<div class="subq-row"><span class="subq-label">Domains:</span>'
        f'<span class="subq-value">{_escape(rings)}</span></div>'
        '</div>'
    )


def render_metrics(result: PipelineResult) -> str:
    if result.is_error:
        return ""
    if result.flagged:
        audit_badge = (
            f'<span class="badge-red">FLAGGED — '
            f'{_escape(result.flag_reason)}</span>'
        )
    else:
        audit_badge = '<span class="badge-green">CLEAN</span>'
    return (
        '<div class="metrics-panel">'
        f'<span class="metric">&#9201; {result.total_latency_ms:.0f} ms</span>'
        f'<span class="metric">&#128203; {result.tokens_generated} tokens</span>'
        f'<span class="metric">&#128270; {result.retrieval_candidate_count} candidates</span>'
        f'<span class="metric">&#10003; {result.support_rate:.0%} verified</span>'
        f'<span class="metric">Audit: {audit_badge}</span>'
        '</div>'
    )

# app.py -- Part 5: Warning banner and on_submit handler (append)


def render_banner(result: PipelineResult) -> str:
    if result.is_error or not result.flagged:
        return ""
    return (
        '<div class="warning-banner">'
        '<span class="warning-icon">&#9888;</span>'
        '<strong>Hallucination Warning: </strong>'
        f'{_escape(result.flag_reason)}. '
        'Claims marked in red may not be fully supported by cited sources. '
        'Please verify at the source links.'
        '</div>'
    )


def on_submit(
    query: str,
    ring_filter: list[str],
) -> tuple[str, str, str, str, str]:
    """
    Gradio submit handler.
    Returns (answer_html, banner_html, citations_html, subq_html, metrics_html).
    """
    query = (query or "").strip()
    if not query:
        err = pipeline._error_result("Please enter a question.", "", ring_filter)
        return (render_answer(err), "", render_citations(err),
                render_subqueries(err), render_metrics(err))

    # None means search all rings (no filter)
    rings = ring_filter if (ring_filter and len(ring_filter) < len(ALL_RING_LABELS)) else None

    result = pipeline.run_safe(query, rings)
    return (
        render_answer(result),
        render_banner(result),
        render_citations(result),
        render_subqueries(result),
        render_metrics(result),
    )

# app.py -- Part 6: Gradio Blocks layout (append)


EXAMPLE_QUERIES = [
    ["What is the current PPF interest rate and how is it compounded?"],
    ["Compare NPS vs PPF for retirement planning for a 35-year-old in India"],
    ["What are the ELSS tax benefits under Section 80C and the lock-in period?"],
    ["How does the LRS remittance limit work for investing in US stocks from India?"],
    ["What is the DICGC insurance limit for bank deposits in India?"],
    ["PPF vs NPS and also explain ELSS tax saving for a salaried employee"],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="DocuSage -- Indian Personal Finance RAG",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS,
    ) as demo:

        gr.Markdown(
            "# DocuSage &#128218;\n"
            "**Indian Personal Finance Q&A** -- grounded answers with source citations."
        )

        with gr.Tabs():
            with gr.Tab("Ask"):
                with gr.Row():
                    with gr.Column(scale=5):
                        query_box = gr.Textbox(
                            label="Your question",
                            placeholder="e.g. What is the current PPF interest rate?",
                            lines=2, max_lines=5,
                        )
                    with gr.Column(scale=1, min_width=130):
                        submit_btn = gr.Button(
                            "Ask DocuSage", variant="primary", size="lg")

                ring_filter = gr.CheckboxGroup(
                    choices=ALL_RING_LABELS,
                    value=ALL_RING_LABELS,
                    label="Knowledge domains to search",
                )
                banner_out = gr.HTML(value="")

                with gr.Row():
                    with gr.Column(scale=7):
                        answer_out = gr.HTML(
                            value='<div class="muted">Your answer will appear here.</div>',
                            label="Answer",
                        )
                    with gr.Column(scale=3):
                        with gr.Accordion("Citations", open=True):
                            citations_out = gr.HTML(
                                value='<div class="muted">No citations yet.</div>')

                with gr.Accordion("Sub-queries explored", open=False):
                    subq_out = gr.HTML(value="")

                metrics_out = gr.HTML(value="")

                gr.Examples(
                    examples=EXAMPLE_QUERIES,
                    inputs=[query_box],
                    label="Example questions",
                )

            with gr.Tab("System Status"):
                status_out  = gr.JSON(label="Pipeline Status", value={})
                refresh_btn = gr.Button("Refresh Status")

        # Event wiring
        outputs = [answer_out, banner_out, citations_out, subq_out, metrics_out]
        inputs  = [query_box, ring_filter]

        submit_btn.click(fn=on_submit, inputs=inputs, outputs=outputs)
        query_box.submit(fn=on_submit, inputs=inputs, outputs=outputs)
        refresh_btn.click(fn=lambda: pipeline.status(), outputs=[status_out])

    return demo

# app.py -- Part 7: Global CSS and launch (append)

CUSTOM_CSS = """
.docusage-answer { font-size:1rem; line-height:1.75; padding:12px; }
.sent-audit { transition:background 0.2s; border-radius:3px; padding:1px 2px; }
.cite-badge {
    background: var(--color-accent, #3b82f6); color: white;
    border-radius:4px; padding:1px 4px; font-size:0.72em;
    font-weight:600; cursor:help;
}
.error-card {
    background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
    border-radius:8px; padding:14px 18px; display:flex; align-items:center; gap:10px;
}
.error-icon { font-size:1.3rem; }
.error-text { color:#dc2626; }
.warning-banner {
    background:rgba(234,179,8,0.15); border:1px solid rgba(234,179,8,0.5);
    border-radius:8px; padding:12px 18px; margin:8px 0;
}
.warning-icon { font-size:1.2rem; margin-right:6px; }
.cite-card {
    border:1px solid var(--border-color-primary, #e5e7eb);
    border-radius:8px; padding:12px; margin-bottom:10px;
}
.cite-header { display:flex; align-items:baseline; gap:8px; margin-bottom:6px; }
.cite-num { font-weight:700; color:var(--color-accent,#3b82f6); min-width:28px; }
.cite-title { font-weight:600; text-decoration:none; color:inherit; }
.cite-title:hover { text-decoration:underline; }
.cite-meta-row { font-size:0.82rem; color:var(--text-color-subdued,#888); margin-bottom:6px; }
.cite-excerpt {
    font-size:0.88rem; font-style:italic;
    color:var(--text-color-subdued,#666);
    border-left:3px solid var(--border-color-primary,#e5e7eb); padding-left:10px;
}
.audit-legend { display:flex; gap:10px; margin-bottom:12px; flex-wrap:wrap; }
.legend-item { font-size:0.78rem; border-radius:4px; padding:2px 8px; }
.metrics-panel {
    display:flex; gap:18px; flex-wrap:wrap;
    font-size:0.85rem; color:var(--text-color-subdued,#888);
    padding:8px 0; border-top:1px solid var(--border-color-primary,#e5e7eb);
}
.metric { display:flex; align-items:center; gap:4px; }
.badge-green {
    background:rgba(34,197,94,0.2); color:#15803d;
    border-radius:4px; padding:1px 8px; font-size:0.82rem; font-weight:600;
}
.badge-red {
    background:rgba(239,68,68,0.15); color:#dc2626;
    border-radius:4px; padding:1px 8px; font-size:0.82rem; font-weight:600;
}
.subq-panel { padding:8px 4px; }
.subq-row { display:flex; gap:10px; margin-bottom:6px; align-items:baseline; }
.subq-label { font-weight:600; min-width:130px; font-size:0.9rem; }
.subq-value { font-size:0.9rem; }
.subq-list { margin:4px 0 0 0; padding-left:20px; }
.subq-item { margin:3px 0; font-size:0.9rem; }
.muted { color:var(--text-color-subdued,#888); font-style:italic; padding:8px; }
.citations-panel { max-height:600px; overflow-y:auto; padding-right:4px; }
"""

# Module-level build (HF Spaces pattern)
demo = build_ui()

if __name__ == "__main__":
    demo.launch(
        server_port=settings.GRADIO_SERVER_PORT,
        share=False,      # HF Spaces provides the public URL
        show_api=False,   # no raw API endpoint
    )

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

