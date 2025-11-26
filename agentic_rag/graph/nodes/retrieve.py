from typing import Any, Dict, List

from langchain_core.documents import Document

from database import fetch_user_enrollments
from ingestion import retriever
from graph.state import GraphState


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from the retriever.
    Enhances query with conversation history for better context-aware retrieval.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary containing the retrieved documents and the question
    """
    print("---RETRIEVE---")
    question = state["question"]
    user_id = state.get("user_id")
    chat_history = state.get("chat_history", [])

    # Enhance query with conversation history for follow-up questions
    enhanced_query = question
    if chat_history:
        # Get the last question and answer for context
        last_question, last_answer = chat_history[-1]
        
        # If current question is short/ambiguous, enhance with context
        # Common follow-up patterns: "give me", "show me", "what about", "how about", "tell me more"
        follow_up_keywords = [
            "give me", "show me", "what about", "how about", "tell me more",
            "example", "examples", "code", "demo", "demonstrate",
            "cho tôi", "ví dụ", "code", "mẫu"
        ]
        
        is_follow_up = any(keyword in question.lower() for keyword in follow_up_keywords)
        
        if is_follow_up or len(question.split()) < 5:
            # Enhance query with context from previous conversation
            enhanced_query = f"{last_question} {question} {last_answer[:200]}"
            print(f"---ENHANCED QUERY WITH CONVERSATION CONTEXT---")
            print(f"Original: {question}")
            print(f"Enhanced: {enhanced_query[:200]}...")

    documents: List[Document] = retriever.invoke(enhanced_query)

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

    return {
        "documents": documents,
        "question": question,
        "user_id": user_id,
        "chat_history": state.get("chat_history", []),
    }
