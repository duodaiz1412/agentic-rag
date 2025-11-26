"""
Script to run the FastAPI server for Agentic RAG API.
Runs on port 8001 to avoid conflict with Spring Boot backend (port 8000).
"""

import sys
from pathlib import Path

# Add the agentic_rag directory to Python path for imports
current_dir = Path(__file__).parent
agentic_rag_dir = current_dir / "agentic_rag"
if str(agentic_rag_dir) not in sys.path:
    sys.path.insert(0, str(agentic_rag_dir))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "agentic_rag.api:api_app",
        host="0.0.0.0",
        port=8001,  # Port 8001 to avoid conflict with Spring Boot backend (port 8000)
        reload=True,
    )

