"""
DataSift - Backend Application Entry Point
Universal Data Intelligence & ML Experimentation Platform
"""

from fastapi import FastAPI
from app.routes.upload import router as upload_router
from app.routes.profile import router as profile_router
from app.routes.quality import router as quality_router
from app.routes.intelligence import router as intelligence_router
from app.routes.clean import router as clean_router
from app.routes.eda import router as eda_router
from app.routes.anomaly import router as anomaly_router
from app.routes.classification import router as classification_router
from app.routes.regression import router as regression_router
from app.routes.clustering import router as clustering_router

# Initialize the FastAPI application
app = FastAPI(
    title="DataSift API",
    description="Universal Data Intelligence & ML Experimentation Platform - Backend Foundation",
    version="0.1.0"
)

# Register route modules
app.include_router(upload_router)
app.include_router(profile_router)
app.include_router(quality_router)
app.include_router(intelligence_router)
app.include_router(clean_router)
app.include_router(eda_router)
app.include_router(anomaly_router)
app.include_router(classification_router)
app.include_router(regression_router)
app.include_router(clustering_router)


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
