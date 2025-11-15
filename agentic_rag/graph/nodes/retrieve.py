from typing import Any, Dict, List

from langchain_core.documents import Document

from database import fetch_user_enrollments
from ingestion import retriever
from graph.state import GraphState


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from the retriever.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary containing the retrieved documents and the question
    """
    print("---RETRIEVE---")
    question = state["question"]
    user_id = state.get("user_id")

    documents: List[Document] = retriever.invoke(question)

    if user_id:
        allowed_courses = fetch_user_enrollments(user_id)
        filtered_documents = []
        for doc in documents:
            metadata = doc.metadata or {}
            requires_enrollment = metadata.get("requires_enrollment", False)
            course_id = metadata.get("course_id")
            if not requires_enrollment:
                filtered_documents.append(doc)
            elif course_id and course_id in allowed_courses:
                filtered_documents.append(doc)
            else:
                print("---DOCUMENT FILTERED: USER LACKS ACCESS---")
        documents = filtered_documents

    return {"documents": documents, "question": question, "user_id": user_id}
