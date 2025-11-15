from typing import Any, Dict, List

from langchain_core.documents import Document

from graph.chains.generation import generation_chain
from graph.state import GraphState

SOURCE_KEYS = [
    "document_id",
    "doc_type",
    "course_id",
    "course_title",
    "chapter_id",
    "chapter_title",
    "lesson_id",
    "lesson_title",
    "requires_enrollment",
    "tags",
    "language",
    "course_skill_level",
    "chapter_summary",
    "last_modified",
]


def _build_context(documents: List[Document]) -> str:
    context_chunks = []
    for doc in documents:
        metadata = doc.metadata or {}
        header_parts = []
        if metadata.get("course_title"):
            header_parts.append(f"Course: {metadata['course_title']}")
        if metadata.get("chapter_title"):
            header_parts.append(f"Chapter: {metadata['chapter_title']}")
        if metadata.get("lesson_title"):
            header_parts.append(f"Lesson: {metadata['lesson_title']}")
        header = " • ".join(header_parts)
        chunk = "\n\n".join(filter(None, [header, doc.page_content]))
        context_chunks.append(chunk.strip())
    return "\n\n-----\n\n".join(context_chunks)


def _extract_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents):
        metadata = doc.metadata or {}
        source_entry = {"rank": idx + 1}
        for key in SOURCE_KEYS:
            if key in metadata and metadata[key] is not None:
                source_entry[key] = metadata[key]
        if "distance" in metadata:
            source_entry["distance"] = metadata["distance"]
        sources.append(source_entry)
    return sources


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response to the user question.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): A dictionary containing the generated response and the question
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    context = _build_context(documents)
    generation = generation_chain.invoke({"context": context, "question": question})
    sources = _extract_sources(documents)
    return {
        "generation": generation,
        "documents": documents,
        "question": question,
        "sources": sources,
        "user_id": state.get("user_id"),
    }
