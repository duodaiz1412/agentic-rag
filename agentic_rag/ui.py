from __future__ import annotations

from dotenv import load_dotenv
import io
from contextlib import redirect_stdout
import gradio as gr

from graph.graph import app


load_dotenv()


def answer_question(question: str) -> tuple[str, str]:
    if not question or not question.strip():
        return "Please enter a question.", ""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = app.invoke(input={"question": question.strip()})
    trace = buf.getvalue()
    answer = result.get("generation", str(result))
    return answer, trace


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Agentic RAG QA") as demo:
        gr.Markdown("# Agentic RAG • Question → Answer")
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Ask something...", scale=2)
            submit = gr.Button("Submit", variant="primary", scale=1)
        with gr.Row():
            with gr.Column(scale=2):
                answer = gr.Textbox(label="Answer", lines=14)
            with gr.Column(scale=1):
                trace = gr.Textbox(label="RAG Trace", lines=14)

        submit.click(fn=answer_question, inputs=question, outputs=[answer, trace])
        question.submit(fn=answer_question, inputs=question, outputs=[answer, trace])
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()

