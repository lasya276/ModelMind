"""
Data Engine - Storage Module
Provides simple, secure, filesystem-based temporary dataset storage for ModelMind:
- Atomic file writes using temporary files and safe replacement.
- Maximum upload size enforcement (50 MB) via chunked streaming.
- Automatic 48-hour dataset expiration and cleanup policy.
- Strict UUID format validation guarding against path traversal.
- Zero external database dependencies and zero unsafe global mutable state.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import re
import uuid
import time
import json
import pandas as pd
from fastapi import HTTPException, status, UploadFile, Request
from app.data_engine.loader import read_csv_dataframe

# Storage directory inside project workspace
STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / ".dataset_storage"

# Maximum upload size: 50 MB
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024

# Default dataset retention period: 48 hours
EXPIRATION_HOURS = 48

# UUID validation pattern to guard against path traversal attacks
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def get_storage_dir() -> Path:
    """Ensures storage directory exists and returns its Path."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR


def _validate_dataset_id(dataset_id: str) -> None:
    """Validates dataset_id format to prevent directory traversal."""
    if not dataset_id or not isinstance(dataset_id, str):
        raise ValueError("dataset_id must be a non-empty string.")
    if not UUID_PATTERN.match(dataset_id.strip()):
        raise ValueError(f"Invalid dataset_id format: '{dataset_id}'. Must be a valid UUID.")


async def read_upload_file_safely(
    file: UploadFile,
    max_size_bytes: Optional[int] = None,
) -> bytes:
    """
    Reads an uploaded file in chunks to prevent unbounded memory allocation
    and enforces maximum upload size limits.

    Args:
        file: The FastAPI UploadFile instance.
        max_size_bytes: Max allowed bytes (defaults to MAX_UPLOAD_SIZE_BYTES).

    Returns:
        bytes: The full file content.

    Raises:
        HTTPException(413): If file size exceeds max_size_bytes.
        HTTPException(500): If an unexpected reading error occurs.
    """
    limit = max_size_bytes if max_size_bytes is not None else MAX_UPLOAD_SIZE_BYTES

    chunks = []
    total_size = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > limit:
                limit_mb = limit / (1024 * 1024)
                limit_label = f"{int(limit_mb)} MB" if limit_mb == int(limit_mb) else f"{limit_mb:.2f} MB"
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File size exceeds the maximum allowed limit of {limit_label}.",
                )
            chunks.append(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    return b"".join(chunks)


def cleanup_expired_datasets(max_age_hours: int = EXPIRATION_HOURS) -> int:
    """
    Removes datasets older than max_age_hours from temporary storage.
    Safely deletes both .csv and .json metadata files, as well as orphaned .tmp files.

    Args:
        max_age_hours: Maximum dataset age in hours (default: 48).

    Returns:
        int: Number of expired datasets cleaned up.
    """
    s_dir = get_storage_dir()
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0

    # 1. Clean up orphaned temporary files older than 1 hour
    for tmp_file in s_dir.glob("*.tmp"):
        try:
            if (now - tmp_file.stat().st_mtime) > 3600:
                tmp_file.unlink(missing_ok=True)
        except OSError:
            pass

    # 2. Inspect metadata files for expiration
    for meta_file in s_dir.glob("*.json"):
        dataset_id = meta_file.stem
        if not UUID_PATTERN.match(dataset_id):
            continue

        data_file = s_dir / f"{dataset_id}.csv"
        is_expired = False

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            created_at = meta.get("created_at")
            if created_at is not None:
                if (now - float(created_at)) > max_age_seconds:
                    is_expired = True
            else:
                # Fallback to file modification time if created_at is missing
                if (now - meta_file.stat().st_mtime) > max_age_seconds:
                    is_expired = True
        except Exception:
            if (now - meta_file.stat().st_mtime) > max_age_seconds:
                is_expired = True

        if is_expired:
            try:
                data_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
                cleaned_count += 1
            except OSError:
                pass

    return cleaned_count


def save_dataset(file_bytes: bytes, filename: str) -> str:
    """
    Saves raw CSV bytes to temporary server storage using atomic file replacement
    and records metadata with a creation timestamp.

    Args:
        file_bytes: Raw bytes of the uploaded CSV.
        filename: Original filename.

    Returns:
        str: Unique dataset_id (UUID).
    """
    dataset_id = str(uuid.uuid4())
    s_dir = get_storage_dir()

    data_file = s_dir / f"{dataset_id}.csv"
    meta_file = s_dir / f"{dataset_id}.json"
    temp_data_file = s_dir / f"{dataset_id}.csv.tmp"
    temp_meta_file = s_dir / f"{dataset_id}.json.tmp"

    metadata: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "filename": filename,
        "size_bytes": len(file_bytes),
        "created_at": time.time(),
    }

    try:
        # Write to temporary files first to prevent partial/truncated writes
        temp_data_file.write_bytes(file_bytes)
        temp_meta_file.write_text(json.dumps(metadata), encoding="utf-8")

        # Atomically replace destination files
        temp_data_file.replace(data_file)
        temp_meta_file.replace(meta_file)
    except Exception:
        temp_data_file.unlink(missing_ok=True)
        temp_meta_file.unlink(missing_ok=True)
        raise

    # Opportunistically prune expired datasets
    try:
        cleanup_expired_datasets(max_age_hours=EXPIRATION_HOURS)
    except Exception:
        pass

    return dataset_id


def get_dataset(dataset_id: str) -> Tuple[bytes, str]:
    """
    Retrieves stored dataset bytes and original filename.

    Args:
        dataset_id: UUID of the dataset.

    Returns:
        Tuple of (file_bytes, filename).

    Raises:
        ValueError: If dataset_id format is invalid.
        FileNotFoundError: If dataset does not exist in storage.
    """
    _validate_dataset_id(dataset_id)
    clean_id = dataset_id.strip()
    s_dir = get_storage_dir()

    data_file = s_dir / f"{clean_id}.csv"
    meta_file = s_dir / f"{clean_id}.json"

    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset '{clean_id}' was not found in temporary storage.")

    file_bytes = data_file.read_bytes()

    filename = "dataset.csv"
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            filename = meta.get("filename", filename)
        except Exception:
            pass

    return file_bytes, filename


def get_dataset_preview(dataset_id: str, preview_limit: int = 20) -> Dict[str, Any]:
    """Reads a bounded CSV preview and counts rows without loading the full dataset."""
    _validate_dataset_id(dataset_id)
    clean_id = dataset_id.strip()
    s_dir = get_storage_dir()
    data_file = s_dir / f"{clean_id}.csv"
    meta_file = s_dir / f"{clean_id}.json"

    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset '{clean_id}' was not found in temporary storage.")

    now = time.time()
    created_at = None
    if meta_file.is_file():
        try:
            created_at = json.loads(meta_file.read_text(encoding="utf-8")).get("created_at")
        except (OSError, ValueError):
            pass
    try:
        dataset_created_at = float(created_at) if created_at is not None else data_file.stat().st_mtime
    except (TypeError, ValueError):
        dataset_created_at = data_file.stat().st_mtime
    if now - dataset_created_at > EXPIRATION_HOURS * 3600:
        raise FileNotFoundError(f"Dataset '{clean_id}' has expired and is no longer available.")

    if data_file.stat().st_size == 0:
        return {
            "dataset_id": clean_id,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "total_columns": 0,
        }

    try:
        preview_df = pd.read_csv(data_file, nrows=preview_limit)
        columns = [str(column) for column in preview_df.columns]
        preview_rows = [
            [
                _preview_value(value)
                for value in row
            ]
            for row in preview_df.itertuples(index=False, name=None)
        ]
        total_rows = sum(
            len(chunk)
            for chunk in pd.read_csv(data_file, chunksize=10_000)
        )
    except pd.errors.EmptyDataError:
        return {
            "dataset_id": clean_id,
            "columns": [],
            "rows": [],
            "total_rows": 0,
            "total_columns": 0,
        }
    except (pd.errors.ParserError, UnicodeDecodeError) as err:
        raise ValueError(f"Unable to preview the stored CSV dataset: {err}") from err

    return {
        "dataset_id": clean_id,
        "columns": columns,
        "rows": preview_rows,
        "total_rows": total_rows,
        "total_columns": len(columns),
    }


def _preview_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not pd.notna(value):
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return str(value)
    return value


def get_dataset_metadata(dataset_id: str) -> Dict[str, Any]:
    """
    Retrieves stored dataset metadata.

    Raises:
        ValueError: If dataset_id format is invalid.
        FileNotFoundError: If dataset does not exist.
    """
    _validate_dataset_id(dataset_id)
    clean_id = dataset_id.strip()
    s_dir = get_storage_dir()

    meta_file = s_dir / f"{clean_id}.json"
    data_file = s_dir / f"{clean_id}.csv"

    if not data_file.is_file():
        raise FileNotFoundError(f"Dataset '{clean_id}' was not found in temporary storage.")

    if meta_file.is_file():
        return json.loads(meta_file.read_text(encoding="utf-8"))

    return {
        "dataset_id": clean_id,
        "filename": "dataset.csv",
        "size_bytes": data_file.stat().st_size,
    }


def load_stored_dataframe(dataset_id: str) -> pd.DataFrame:
    """
    Loads a stored dataset into a validated pandas DataFrame.

    Raises:
        ValueError: If dataset_id format is invalid or CSV parsing fails.
        FileNotFoundError: If dataset does not exist.
    """
    file_bytes, _ = get_dataset(dataset_id)
    return read_csv_dataframe(file_bytes)


async def resolve_dataset_from_request(
    request: Request,
    file: Optional[UploadFile] = None,
    dataset_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Extracts and parses a DataFrame, filename, and dataset_id from either:
    1. An uploaded CSV file.
    2. A dataset_id provided via query, form, or JSON body.

    Returns:
        Tuple[pd.DataFrame, str, str]: (df, filename, dataset_id)

    Raises:
        HTTPException(400): If validation fails or neither input is provided.
        HTTPException(404): If dataset_id is not found.
        HTTPException(413): If file exceeds size limit.
        HTTPException(500): On unexpected server error.
    """
    # 1. If file was provided
    if file is not None:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided in upload.",
            )
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only CSV files (.csv) are supported.",
            )
        try:
            contents = await read_upload_file_safely(file)
        finally:
            await file.close()

        try:
            df = read_csv_dataframe(contents)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            )

        stored_id = save_dataset(file_bytes=contents, filename=file.filename)
        return df, file.filename, stored_id

    # 2. If dataset_id was not provided in query, inspect request body
    if not dataset_id:
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    dataset_id = body.get("dataset_id")
            except Exception:
                pass
        elif (
            "multipart/form-data" in content_type
            or "application/x-www-form-urlencoded" in content_type
        ):
            try:
                form = await request.form()
                dataset_id = form.get("dataset_id")
                if not file and "file" in form:
                    potential_file = form["file"]
                    if isinstance(potential_file, UploadFile):
                        file = potential_file
                        return await resolve_dataset_from_request(request, file=file, dataset_id=None)
            except Exception:
                pass

    # 3. If neither file nor dataset_id is found
    if not dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file or dataset_id provided. Please upload a CSV file or provide an existing dataset_id.",
        )

    # 4. Load from stored dataset_id
    try:
        df = load_stored_dataframe(dataset_id)
        metadata = get_dataset_metadata(dataset_id)
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

    return df, metadata.get("filename", "dataset.csv"), dataset_id

