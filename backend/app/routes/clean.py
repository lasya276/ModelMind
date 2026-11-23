"""
Routes - Clean
Defines endpoints for user-approved dataset cleaning transformations
and downloading cleaned CSV datasets.
"""

from typing import Optional, Dict, Any
import json
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Query,
    Form,
    Request,
    Response,
    status,
)
from app.data_engine.cleaner import clean_dataset
from app.data_engine.storage import (
    resolve_dataset_from_request,
    save_dataset,
    get_dataset,
    get_dataset_preview,
)

# Supported top-level cleaning operation keys
VALID_OPERATION_KEYS = {
    "missing_values",
    "remove_duplicates",
    "duplicate_subset",
    "trim_whitespace",
    "casing",
    "drop_empty_columns",
    "drop_constant_columns",
    "invalid_numerical_values",
    "value_mapping",
}

# Create an isolated router for cleaning operations
router = APIRouter(tags=["Clean"])


@router.post("/clean", summary="Clean a dataset based on explicit user directives")
async def clean_dataset_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Query(None, description="UUID of a previously uploaded dataset"),
    operations: Optional[str] = Form(None, description="JSON-serialized cleaning operations dictionary"),
):
    """
    Applies user-approved data cleaning operations:
    - Missing value imputation (mean, median, mode, constant).
    - Duplicate row removal (optional column subset).
    - Categorical formatting (whitespace trimming, lowercase/uppercase conversions).
    - Constant and empty column detection and optional removal.

    Never modifies or overwrites the original dataset.
    Saves the cleaned dataset and returns a download link.

    Accepts either:
    1. A newly uploaded CSV file (multipart/form-data).
    2. A dataset_id referencing a dataset previously stored on the server.
    Cleaning operations can be passed via form field, query parameter, or JSON body.

    Returns:
        JSON response with the cleaning report, original dataset_id,
        cleaned_dataset_id, and download_url.
    """
    # 1. Resolve dataset, original filename, and original dataset_id
    df, filename, orig_dataset_id = await resolve_dataset_from_request(
        request, file=file, dataset_id=dataset_id
    )

    # 2. Extract and parse operations dictionary
    ops_dict: Optional[Dict[str, Any]] = None

    # Check if operations was passed as form or query string
    if operations is not None:
        if isinstance(operations, str) and operations.strip():
            try:
                parsed = json.loads(operations)
                if isinstance(parsed, dict):
                    ops_dict = parsed
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Cleaning operations must be a JSON dictionary of directives.",
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON string in operations parameter: {str(e)}",
                )
        elif isinstance(operations, dict):
            ops_dict = operations

    # If operations was not passed via form, check JSON body
    if ops_dict is None:
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                body = await request.json()
                if isinstance(body, dict) and "operations" in body:
                    body_ops = body["operations"]
                    if isinstance(body_ops, dict):
                        ops_dict = body_ops
                    elif body_ops is not None:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Cleaning operations must be a dictionary/object.",
                        )
            except HTTPException:
                raise
            except Exception:
                pass

    # Default to empty dict if no operations were requested
    if ops_dict is None:
        ops_dict = {}

    # 3. Validate supported operations
    unsupported_keys = set(ops_dict.keys()) - VALID_OPERATION_KEYS
    if unsupported_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported cleaning operation(s): {sorted(list(unsupported_keys))}. "
                f"Supported operations are: {sorted(list(VALID_OPERATION_KEYS))}."
            ),
        )

    # 4. Perform cleaning without modifying original DataFrame
    try:
        cleaned_df, report = clean_dataset(df, operations=ops_dict)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while cleaning the dataset: {str(err)}",
        )

    # 5. Save cleaned dataset for download (preserving original dataset untouched)
    cleaned_csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    cleaned_filename = f"cleaned_{filename}" if not filename.startswith("cleaned_") else filename
    cleaned_dataset_id = save_dataset(file_bytes=cleaned_csv_bytes, filename=cleaned_filename)

    return {
        "status": "success",
        "message": "Dataset successfully cleaned.",
        "filename": cleaned_filename,
        "original_dataset_id": orig_dataset_id,
        "cleaned_dataset_id": cleaned_dataset_id,
        "download_url": f"/clean/download/{cleaned_dataset_id}",
        "data": report,
    }


@router.get("/clean/preview/{dataset_id}", summary="Preview a cleaned dataset")
async def preview_cleaned_dataset(dataset_id: str):
    """Returns the first 20 rows and total dimensions of a stored dataset."""
    try:
        return get_dataset_preview(dataset_id, preview_limit=20)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err),
        )


@router.get("/clean/download/{dataset_id}", summary="Download a cleaned dataset as CSV")
async def download_cleaned_dataset(dataset_id: str):
    """
    Downloads a stored CSV dataset by its unique dataset_id.
    """
    try:
        file_bytes, filename = get_dataset(dataset_id)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(err),
        )

    return Response(
        content=file_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/clean/download", summary="Download a cleaned dataset as CSV (query parameter)")
async def download_cleaned_dataset_query(dataset_id: str = Query(...)):
    """
    Query parameter fallback for downloading a stored CSV dataset.
    """
    return await download_cleaned_dataset(dataset_id)
