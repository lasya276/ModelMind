"""
Routes - Upload
Defines endpoints for file uploads and initial structural validation.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.data_engine.loader import load_csv
from app.data_engine.storage import save_dataset, read_upload_file_safely

# Create an isolated router for upload-related operations
router = APIRouter(tags=["Upload"])


@router.post("/upload", summary="Upload and inspect a CSV dataset")
async def upload_csv_file(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV file, validates its extension, reads it with Pandas,
    and returns basic structural metadata (filename, row count, column count, columns).
    """
    # 1. Validate that a file was provided with a filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in upload."
        )

    # 2. Validate the file extension (must be .csv)
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only CSV files (.csv) are supported."
        )

    # 3. Read file contents safely with upload size enforcement
    contents = await read_upload_file_safely(file)

    # 4. Process and parse CSV using data_engine.loader
    try:
        dataset_info = load_csv(file_bytes=contents, filename=file.filename)
        # Preserve dataset in temporary server-side storage for subsequent analysis
        dataset_id = save_dataset(file_bytes=contents, filename=file.filename)
        dataset_info["dataset_id"] = dataset_id

        return {
            "status": "success",
            "message": "CSV file successfully processed.",
            "data": dataset_info
        }
    except ValueError as err:
        # Gracefully handle empty or malformed CSV errors with HTTP 400
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err)
        )
    finally:
        # Always close the file handle after reading
        await file.close()
