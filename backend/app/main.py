"""
DataSift - Backend Application Entry Point
Universal Data Intelligence & ML Experimentation Platform
"""

from fastapi import FastAPI
from app.routes.upload import router as upload_router

# Initialize the FastAPI application
app = FastAPI(
    title="DataSift API",
    description="Universal Data Intelligence & ML Experimentation Platform - Backend Foundation",
    version="0.1.0"
)

# Register route modules
app.include_router(upload_router)


@app.get("/", summary="Root API Status")
async def root():
    """
    Root endpoint returning a welcoming message and confirmation that the API is active.
    """
    return {
        "service": "DataSift API",
        "status": "online",
        "version": "0.1.0",
        "message": "Welcome to DataSift – Universal Data Intelligence & ML Experimentation Platform API"
    }


@app.get("/health", summary="Health Check")
async def health_check():
    """
    Health-check endpoint for monitoring service availability.
    """
    return {
        "status": "healthy"
    }
