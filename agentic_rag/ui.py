from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Iterable

import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document

from graph.graph import app


load_dotenv()


def _format_metadata(metadata: dict | None) -> str:
    if not metadata:
        return ""
    lines = []
    for key, value in metadata.items():
        if value is None:
            continue
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def _format_documents(documents: Iterable[Document] | None, preview_chars: int = 600) -> str:
    docs = list(documents or [])
    if not docs:
        return "_No documents retrieved (fallback to web or empty result)._"

    chunks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        metadata_md = _format_metadata(doc.metadata or {})
        content_preview = (doc.page_content or "").strip()
        if preview_chars and len(content_preview) > preview_chars:
            content_preview = content_preview[:preview_chars].rstrip() + "..."
        section = [
            f"### Document {idx}",
            metadata_md or "_No metadata_",
            "",
            "```text",
            content_preview or "(empty)",
            "```",
        ]
        chunks.append("\n".join(section))
    return "\n\n---\n\n".join(chunks)


def answer_question(question: str, user_id: str | None = None) -> tuple[str, str, list, str]:
    if not question or not question.strip():
        return "Please enter a question.", "", [], "_No documents retrieved._"

    buf = io.StringIO()
    payload = {"question": question.strip()}
    if user_id and user_id.strip():
        payload["user_id"] = user_id.strip()

    with redirect_stdout(buf):
        result = app.invoke(input=payload)

    trace = buf.getvalue()
    answer = result.get("generation", str(result))
    sources = result.get("sources", [])
    formatted_docs = _format_documents(result.get("documents"))
    return answer, trace, sources, formatted_docs


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Agentic RAG QA") as demo:
        gr.Markdown("# Agentic RAG • Question → Answer")
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Ask something...", scale=2)
            user_id = gr.Textbox(label="User ID (optional)", placeholder="UUID of learner", scale=1)
            submit = gr.Button("Submit", variant="primary", scale=1)
        with gr.Row():
            with gr.Column(scale=2):
                answer = gr.Textbox(label="Answer", lines=12)
                documents_md = gr.Markdown()
            with gr.Column(scale=1):
                trace = gr.Textbox(label="RAG Trace", lines=20)
                sources = gr.JSON(label="Sources (metadata)")

        submit.click(
            fn=answer_question,
            inputs=[question, user_id],
            outputs=[answer, trace, sources, documents_md],
        )
        question.submit(
            fn=answer_question,
            inputs=[question, user_id],
            outputs=[answer, trace, sources, documents_md],
        )
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()

