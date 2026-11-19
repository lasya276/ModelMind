"""
Routes - Quality
Defines endpoints for dataset quality inspection and health checks.
"""

from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, status
from app.data_engine.quality import check_data_quality
from app.data_engine.storage import resolve_dataset_from_request

# Create an isolated router for data quality operations
router = APIRouter(tags=["Quality"])


@router.post("/quality", summary="Evaluate dataset quality and detect anomalies")
async def quality_check_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Query(None, description="UUID of a previously uploaded dataset"),
):
    """
    Performs comprehensive data quality checks on a dataset:
    - Missing value counts, percentages, and severity classification.
    - Full row duplicate detection and impact percentage.
    - Completely empty columns and zero-variance constant columns.
    - Heuristic potential identifier column detection.
    - Inconsistent categorical values (casing and whitespace discrepancies).
    - Invalid numerical values (+inf / -inf).

    Accepts either:
    1. A newly uploaded CSV file (multipart/form-data).
    2. A dataset_id referencing a dataset previously stored on the server.

    Returns:
        JSON response with the complete quality report.
    """
    df, filename, stored_id = await resolve_dataset_from_request(
        request, file=file, dataset_id=dataset_id
    )

    try:
        report = check_data_quality(df)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while evaluating dataset quality: {str(err)}",
        )

    return {
        "status": "success",
        "message": "Data quality analysis completed successfully.",
        "filename": filename,
        "dataset_id": stored_id,
        "data": report,
    }
