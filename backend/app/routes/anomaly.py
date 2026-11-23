"""
Routes - Anomaly Detection
Defines endpoints for unsupervised anomaly and outlier detection
using Scikit-learn's Isolation Forest algorithm.
"""

from typing import Optional, List, Union, Any
import json
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Query,
    Form,
    Request,
    status,
)
from app.ml_engine.anomaly import detect_anomalies
from app.data_engine.storage import resolve_dataset_from_request

# Create an isolated router for anomaly detection operations
router = APIRouter(tags=["Anomaly Detection"])


def _parse_feature_columns(val: Any) -> Optional[List[str]]:
    """Helper to parse feature_columns from list, JSON string, or comma-separated string."""
    if val is None:
        return None
    if isinstance(val, list):
        return [str(c) for c in val]
    if isinstance(val, str):
        trimmed = val.strip()
        if not trimmed:
            return None
        # Try parsing JSON array
        if trimmed.startswith("[") and trimmed.endswith("]"):
            try:
                parsed = json.loads(trimmed)
                if isinstance(parsed, list):
                    return [str(c) for c in parsed]
            except Exception:
                pass
        # Fall back to comma-separated values
        return [c.strip() for c in trimmed.split(",") if c.strip()]
    return None


def _parse_contamination(val: Any) -> Union[float, str]:
    """Helper to parse contamination parameter into float or 'auto'."""
    if val is None or val == "":
        return "auto"
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        trimmed = val.strip().lower()
        if trimmed == "auto":
            return "auto"
        try:
            return float(trimmed)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Contamination must be 'auto' or a float between 0.0 and 0.5, got '{val}'.",
            )
    return "auto"


@router.post("/anomaly", summary="Detect unusual observations using Isolation Forest")
async def anomaly_detection_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Query(None, description="UUID of a previously uploaded dataset"),
    feature_columns: Optional[str] = Form(None, description="Feature columns as JSON array or comma-separated names"),
    contamination: Optional[str] = Form(None, description="Contamination factor (float in (0, 0.5] or 'auto')"),
):
    """
    Detects unusual rows in a dataset using Isolation Forest:
    - Purely numeric feature extraction with automatic identifier exclusion.
    - Safe median imputation for missing and infinite values.
    - Outputs row-level anomaly labels (1 for normal, -1 for anomaly) and anomaly scores.
    - Includes at most 500 row-aligned observation previews and 100 flagged observations,
      with up to 10 selected feature values per row.
    - Transparently documents detected anomalies as statistical outliers requiring
      investigation, rather than assuming fraud, defects, or errors.

    Accepts either:
    1. A newly uploaded CSV file (multipart/form-data).
    2. A dataset_id referencing a dataset previously stored on the server.
    Parameters (feature_columns, contamination) can be passed via form field,
    query parameter, or JSON body.

    Returns:
        JSON response with anomaly detection results, bounded observation previews, and explanations.
    """
    # 1. Resolve dataset, original filename, and stored dataset_id
    df, filename, stored_id = await resolve_dataset_from_request(
        request, file=file, dataset_id=dataset_id
    )

    # 2. Extract and parse parameters from form, query, or JSON body
    parsed_features: Optional[List[str]] = _parse_feature_columns(feature_columns)
    parsed_contamination: Union[float, str] = _parse_contamination(contamination)

    # Check query parameters fallback if not provided via form
    if parsed_features is None and "feature_columns" in request.query_params:
        parsed_features = _parse_feature_columns(request.query_params.get("feature_columns"))
    if parsed_contamination == "auto" and "contamination" in request.query_params:
        parsed_contamination = _parse_contamination(request.query_params.get("contamination"))

    # Check JSON body fallback if Content-Type is application/json
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                if parsed_features is None and "feature_columns" in body:
                    parsed_features = _parse_feature_columns(body["feature_columns"])
                if parsed_contamination == "auto" and "contamination" in body:
                    parsed_contamination = _parse_contamination(body["contamination"])
        except HTTPException:
            raise
        except Exception:
            pass

    # 3. Execute anomaly detection engine
    try:
        report = detect_anomalies(
            df=df,
            feature_columns=parsed_features,
            contamination=parsed_contamination,
            random_state=42,
        )
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing anomaly detection: {str(err)}",
        )

    return {
        "status": "success",
        "message": "Anomaly detection completed successfully.",
        "filename": filename,
        "dataset_id": stored_id,
        "data": report,
    }
