"""
FastAPI REST endpoint for hotel data RAG queries.
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.query_engine import get_query_engine, QueryResult
from rag.indexer import rebuild_index


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for querying hotel data."""
    question: str = Field(
        ..., 
        description="Natural language question about the hotel",
        min_length=1,
        max_length=500
    )


class QueryResponse(BaseModel):
    """Response model for hotel data queries."""
    question: str
    relevant_data: list[dict[str, Any]]
    source_texts: list[str]
    categories: list[str]
    has_relevant_data: bool


class RebuildResponse(BaseModel):
    """Response for index rebuild operation."""
    success: bool
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str


# Create FastAPI app
app = FastAPI(
    title="Hotel RAG API",
    description="RAG-powered API for querying hotel information",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="hotel-rag")


@app.post("/query", response_model=QueryResponse)
async def query_hotel(request: QueryRequest):
    """
    Query hotel data with a natural language question.
    
    Returns relevant data from the hotel database based on semantic similarity.
    If no relevant data is found, has_relevant_data will be False.
    """
    try:
        engine = get_query_engine()
        result = engine.query(request.question)
        
        return QueryResponse(
            question=result.question,
            relevant_data=result.relevant_data,
            source_texts=result.source_texts,
            categories=result.categories,
            has_relevant_data=result.has_relevant_data
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Query failed: {str(e)}\n\nDetails:\n{error_details}"
        )


@app.post("/rebuild-index", response_model=RebuildResponse)
async def rebuild_hotel_index():
    """
    Rebuild the vector index from the hotel data file.
    
    Use this endpoint after updating the hotel_data.json file.
    """
    try:
        rebuild_index()
        # Reset the query engine to use the new index
        global _query_engine
        from rag import query_engine as qe_module
        qe_module._query_engine = None
        
        return RebuildResponse(success=True, message="Index rebuilt successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")

