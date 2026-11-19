"""
Routes - Classification
Defines endpoints for supervised classification model training and evaluation
using Logistic Regression, Decision Tree, and Random Forest classifiers.
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
from app.ml_engine.classification import train_classification_model
from app.data_engine.storage import resolve_dataset_from_request

# Create an isolated router for classification operations
router = APIRouter(tags=["Classification"])


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


@router.post("/classify", summary="Train and compare classification models")
async def classification_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Query(None, description="UUID of a previously uploaded dataset"),
    target_column: Optional[str] = Form(None, description="Target column to predict"),
    feature_columns: Optional[str] = Form(None, description="Feature columns as JSON array or comma-separated names"),
    test_size: Optional[float] = Form(None, description="Test split proportion (0.05 to 0.5)"),
):
    """
    Trains and compares Logistic Regression, Decision Tree, and Random Forest
    classifiers on a user-specified target column:
    - Supports numeric and categorical features with automatic preprocessing.
    - Prevents data leakage with strict train/test pipeline isolation.
    - Returns model metrics, confusion matrices, class distribution, and model comparison.
    - Never automatically selects or assumes a target column.

    Accepts either:
    1. A newly uploaded CSV file (multipart/form-data).
    2. A dataset_id referencing a dataset previously stored on the server.
    Parameters (target_column, feature_columns, test_size) can be passed via
    form field, query parameter, or JSON body.

    Returns:
        JSON response with classification results for all three models.
    """
    # 1. Resolve dataset, original filename, and stored dataset_id
    df, filename, stored_id = await resolve_dataset_from_request(
        request, file=file, dataset_id=dataset_id
    )

    # 2. Extract parameters from form, query, or JSON body
    parsed_target: Optional[str] = target_column
    parsed_features: Optional[List[str]] = _parse_feature_columns(feature_columns)
    parsed_test_size: Optional[float] = test_size

    # Check query parameters fallback
    if parsed_target is None and "target_column" in request.query_params:
        parsed_target = request.query_params.get("target_column")
    if parsed_features is None and "feature_columns" in request.query_params:
        parsed_features = _parse_feature_columns(request.query_params.get("feature_columns"))
    if parsed_test_size is None and "test_size" in request.query_params:
        try:
            parsed_test_size = float(request.query_params.get("test_size"))
        except (ValueError, TypeError):
            pass

    # Check JSON body fallback
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                if parsed_target is None and "target_column" in body:
                    parsed_target = body["target_column"]
                if parsed_features is None and "feature_columns" in body:
                    parsed_features = _parse_feature_columns(body["feature_columns"])
                if parsed_test_size is None and "test_size" in body:
                    try:
                        parsed_test_size = float(body["test_size"])
                    except (ValueError, TypeError):
                        pass
        except HTTPException:
            raise
        except Exception:
            pass

    # 3. Validate required target_column
    if not parsed_target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target_column is required. Please specify the column to predict.",
        )

    # 4. Set default test_size
    if parsed_test_size is None:
        parsed_test_size = 0.2

    # 5. Execute classification engine
    try:
        report = train_classification_model(
            df=df,
            target_column=parsed_target,
            feature_columns=parsed_features,
            test_size=parsed_test_size,
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
            detail=f"An error occurred while training classification models: {str(err)}",
        )

    return {
        "status": "success",
        "message": "Classification training and evaluation completed successfully.",
        "filename": filename,
        "dataset_id": stored_id,
        "data": report,
    }
