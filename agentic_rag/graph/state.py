from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
    Represents a state of a graph.

    Attributes:
        question: Question
        generation: LLM Generation
        use_web_search: wether to use web search
        documents: List of documents
        sources: List of metadata for surfaced documents
        user_id: Optional user identifier for permission checks
    """

    question: str
    generation: str
    use_web_search: bool
    documents: List[Document]
    sources: List[dict]
    user_id: str
