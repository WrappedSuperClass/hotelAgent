"""
Hotel RAG API - Main entry point.

Run with: python main.py
Or: uvicorn main:app --reload
"""
import uvicorn

from rag.api import app


def main():
    """Start the Hotel RAG API server."""
    print("Starting Hotel RAG API server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()

