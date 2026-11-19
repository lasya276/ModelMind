"""
Routes - Profile
Defines endpoints for dataset profiling and structural summaries.
"""

from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, status
from app.data_engine.profiler import profile_dataset
from app.data_engine.storage import resolve_dataset_from_request

# Create an isolated router for profiling operations
router = APIRouter(tags=["Profiling"])


@router.post("/profile", summary="Generate a comprehensive dataset profile")
async def profile_dataset_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Query(None, description="UUID of a previously uploaded dataset"),
):
    """
    Generates a complete, JSON-serializable profiling report for a dataset.

    Accepts either:
    1. A newly uploaded CSV file (multipart/form-data).
    2. A dataset_id referencing a dataset previously stored on the server.

    Returns:
        JSON response containing the full profiling report including row counts,
        column counts, data types, missing value percentages, unique counts,
        numeric statistics, and categorical summaries.
    """
    df, filename, stored_id = await resolve_dataset_from_request(
        request, file=file, dataset_id=dataset_id
    )

    try:
        report = profile_dataset(df)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while profiling the dataset: {str(err)}"
        )

    return {
        "status": "success",
        "message": "Dataset profiling completed successfully.",
        "filename": filename,
        "dataset_id": stored_id,
        "data": report,
    }
