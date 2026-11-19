"""
Routes - Intelligence
Defines endpoints for dataset structural intelligence, column role inference,
target candidate identification, and analysis workflow recommendations.
"""

from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, status
from app.data_engine.intelligence import analyze_dataset_intelligence
from app.data_engine.storage import resolve_dataset_from_request

# Create an isolated router for intelligence operations
router = APIRouter(tags=["Intelligence"])


@router.post("/intelligence", summary="Analyze dataset intelligence and ML suitability")
async def dataset_intelligence_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Query(None, description="UUID of a previously uploaded dataset"),
):
    """
    Extracts rule-based dataset intelligence:
    - Column role detection (numeric, categorical, datetime, boolean, potential_id, empty).
    - Identification of candidate target variables without auto-selecting one.
    - Recommendations of suitable analysis workflows (EDA, Anomaly Detection,
      Classification, Regression, Clustering) with prerequisite explanations.

    Accepts either:
    1. A newly uploaded CSV file (multipart/form-data).
    2. A dataset_id referencing a dataset previously stored on the server.

    Returns:
        JSON response with the complete intelligence report.
    """
    df, filename, stored_id = await resolve_dataset_from_request(
        request, file=file, dataset_id=dataset_id
    )

    try:
        report = analyze_dataset_intelligence(df)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while analyzing dataset intelligence: {str(err)}",
        )

    return {
        "status": "success",
        "message": "Dataset intelligence analysis completed successfully.",
        "filename": filename,
        "dataset_id": stored_id,
        "data": report,
    }
