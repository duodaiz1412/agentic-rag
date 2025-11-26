"""
FastAPI server for Agentic RAG API.
Provides REST API endpoints for the frontend to interact with the RAG system.
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure we can import graph module
# When running as a package (agentic_rag.api), use relative import
# When running directly, add path to sys.path
try:
    from .graph.graph import app
except ImportError:
    # Fallback: add agentic_rag to path if running directly
    current_file = Path(__file__).resolve()
    agentic_rag_dir = current_file.parent
    if str(agentic_rag_dir) not in sys.path:
        sys.path.insert(0, str(agentic_rag_dir))
    from graph.graph import app

load_dotenv()

# Initialize FastAPI app
api_app = FastAPI(
    title="Agentic RAG API",
    description="API for querying the Agentic RAG system",
    version="1.0.0",
)

# Configure CORS
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatMessage(BaseModel):
    """Single chat message in conversation history"""

    question: str = Field(..., description="User's question")
    answer: str = Field(..., description="Assistant's answer")


class AskRequest(BaseModel):
    """Request model for asking a question"""

    question: str = Field(..., description="The question to ask", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user ID for permission checks")
    chat_history: Optional[List[ChatMessage]] = Field(
        None, description="Previous conversation history"
    )


class Source(BaseModel):
    """Source document metadata"""

    rank: Optional[int] = Field(None, description="Rank of the document in retrieval results")
    document_id: Optional[str] = None
    doc_type: Optional[str] = None
    course_id: Optional[str] = None
    course_title: Optional[str] = None
    chapter_id: Optional[str] = None
    chapter_title: Optional[str] = None
    lesson_id: Optional[str] = None
    lesson_title: Optional[str] = None
    requires_enrollment: Optional[bool] = None
    tags: Optional[List[str]] = None
    language: Optional[str] = None
    course_skill_level: Optional[str] = None
    chapter_summary: Optional[str] = None
    last_modified: Optional[str] = None
    distance: Optional[float] = Field(None, description="Similarity distance score")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class AskResponse(BaseModel):
    """Response model for ask endpoint"""

    answer: str = Field(..., description="The generated answer")
    trace: str = Field(..., description="Debug trace output from the RAG pipeline")
    sources: List[Source] = Field(default_factory=list, description="Source documents metadata")
    chat_history: List[ChatMessage] = Field(
        default_factory=list, description="Updated conversation history"
    )


@api_app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Agentic RAG API is running", "version": "1.0.0"}


@api_app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@api_app.post("/api/v1/rag/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question to the Agentic RAG system.
    
    This endpoint processes the question through the RAG pipeline and returns:
    - The generated answer
    - Source documents metadata
    - Updated conversation history
    - Debug trace information
    
    Args:
        request: AskRequest containing question, optional user_id and chat_history
        
    Returns:
        AskResponse with answer, sources, chat_history, and trace
        
    Raises:
        HTTPException: If question is empty or processing fails
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Prepare payload for the graph
        payload = {"question": request.question.strip()}
        
        if request.user_id and request.user_id.strip():
            payload["user_id"] = request.user_id.strip()
        
        # Convert chat_history from Pydantic models to tuples
        if request.chat_history:
            payload["chat_history"] = [
                (msg.question, msg.answer) for msg in request.chat_history
            ]

        # Capture stdout for trace output
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = app.invoke(input=payload)

        trace = buf.getvalue()
        answer = result.get("generation", str(result))
        sources_raw = result.get("sources", [])
        updated_history_raw = result.get("chat_history", [])

        # Convert sources to Pydantic models
        sources = []
        for source_raw in sources_raw:
            if isinstance(source_raw, dict):
                sources.append(Source(**source_raw))
            else:
                # Fallback for string sources
                sources.append(Source(metadata={"raw": str(source_raw)}))

        # Convert chat_history from tuples to Pydantic models
        chat_history = []
        for q, a in updated_history_raw:
            chat_history.append(ChatMessage(question=q, answer=a))

        return AskResponse(
            answer=answer,
            trace=trace,
            sources=sources,
            chat_history=chat_history,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agentic_rag.api:api_app",
        host="0.0.0.0",
        port=8001,  # Port 8001 to avoid conflict with Spring Boot backend (port 8000)
        reload=True,
    )

