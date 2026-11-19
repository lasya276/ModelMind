"""
Routes - Exploratory Data Analysis (EDA)
Defines endpoints for dataset structural summaries, numerical distributions,
categorical frequencies, Pearson correlation matrices, and missingness analysis.
"""

from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Request, status
from app.data_engine.eda import analyze_eda
from app.data_engine.storage import resolve_dataset_from_request

# Create an isolated router for EDA operations
router = APIRouter(tags=["EDA"])


@router.post("/eda", summary="Generate comprehensive Exploratory Data Analysis report")
async def eda_analysis_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Query(None, description="UUID of a previously uploaded dataset"),
):
    """
    Generates a structured Exploratory Data Analysis (EDA) report:
    - Overall dataset summary (counts, types, shapes).
    - Numerical distributions with histogram bins and descriptive statistics.
    - Categorical frequency distributions with bar-chart ready percentages.
    - Pearson correlation matrix with pairwise relationship rankings.
    - Missing value analysis with column-level visual metrics.

    Accepts either:
    1. A newly uploaded CSV file (multipart/form-data).
    2. A dataset_id referencing a dataset previously stored on the server.

    Returns:
        JSON response with the complete EDA report.
    """
    df, filename, stored_id = await resolve_dataset_from_request(
        request, file=file, dataset_id=dataset_id
    )

    try:
        report = analyze_eda(df)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while generating EDA analysis: {str(err)}",
        )

    return {
        "status": "success",
        "message": "Exploratory Data Analysis completed successfully.",
        "filename": filename,
        "dataset_id": stored_id,
        "data": report,
    }
