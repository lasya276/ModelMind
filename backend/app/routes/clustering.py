"""
Routes - Clustering
Defines endpoints for unsupervised K-Means clustering
using the existing clustering engine.
"""

from typing import Optional, List, Any
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
from app.ml_engine.clustering import perform_clustering
from app.data_engine.storage import resolve_dataset_from_request

# Create an isolated router for clustering operations
router = APIRouter(tags=["Clustering"])


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


def _parse_n_clusters(val: Any) -> int:
    """Helper to parse n_clusters parameter into an integer."""
    if val is None:
        return 3  # Default
    if isinstance(val, bool):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="n_clusters must be an integer, not a boolean.",
        )
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        trimmed = val.strip()
        if not trimmed:
            return 3
        try:
            parsed = int(trimmed)
            return parsed
        except ValueError:
            # Try float conversion for values like "3.0"
            try:
                f = float(trimmed)
                if f == int(f):
                    return int(f)
            except ValueError:
                pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"n_clusters must be an integer >= 2, got '{val}'.",
            )
    if isinstance(val, float):
        if val == int(val):
            return int(val)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"n_clusters must be an integer, got {val}.",
        )
    return 3


@router.post("/cluster", summary="Group observations using K-Means clustering")
async def clustering_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Query(None, description="UUID of a previously uploaded dataset"),
    feature_columns: Optional[str] = Form(None, description="Feature columns as JSON array or comma-separated names"),
    n_clusters: Optional[str] = Form(None, description="Number of clusters (integer >= 2, default 3)"),
):
    """
    Groups observations into natural clusters using K-Means clustering:
    - User-specified or automatic numerical feature selection with identifier exclusion.
    - Safe median imputation for missing and infinite values.
    - Feature standardization via StandardScaler.
    - Returns cluster assignments, sizes, centroids (in original feature scale),
      inertia, and silhouette score.
    - Cluster labels are categorical group identifiers and are not automatically
      claimed to represent meaningful real-world categories.

    Accepts either:
    1. A newly uploaded CSV file (multipart/form-data).
    2. A dataset_id referencing a dataset previously stored on the server.
    Parameters (feature_columns, n_clusters) can be passed via form field,
    query parameter, or JSON body.

    Returns:
        JSON response with clustering results and explanations.
    """
    # 1. Resolve dataset, original filename, and stored dataset_id
    df, filename, stored_id = await resolve_dataset_from_request(
        request, file=file, dataset_id=dataset_id
    )

    # 2. Extract and parse parameters from form, query, or JSON body
    parsed_features: Optional[List[str]] = _parse_feature_columns(feature_columns)
    parsed_n_clusters: int = _parse_n_clusters(n_clusters)

    # Check query parameters fallback if not provided via form
    if parsed_features is None and "feature_columns" in request.query_params:
        parsed_features = _parse_feature_columns(request.query_params.get("feature_columns"))
    if n_clusters is None and "n_clusters" in request.query_params:
        parsed_n_clusters = _parse_n_clusters(request.query_params.get("n_clusters"))

    # Check JSON body fallback
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                if parsed_features is None and "feature_columns" in body:
                    parsed_features = _parse_feature_columns(body["feature_columns"])
                if n_clusters is None and "n_clusters" in body:
                    parsed_n_clusters = _parse_n_clusters(body["n_clusters"])
        except HTTPException:
            raise
        except Exception:
            pass

    # 3. Execute clustering engine
    try:
        report = perform_clustering(
            df=df,
            n_clusters=parsed_n_clusters,
            feature_columns=parsed_features,
            random_state=42,
        )
    except (ValueError, TypeError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while performing clustering: {str(err)}",
        )

    return {
        "status": "success",
        "message": "Clustering completed successfully.",
        "filename": filename,
        "dataset_id": stored_id,
        "data": report,
    }
