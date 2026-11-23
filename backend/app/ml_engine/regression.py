"""
ML Engine - Regression Module
Trains, evaluates, and compares regression algorithms:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Features:
- Pure Scikit-learn pipelines with ColumnTransformer.
- Separate numeric and categorical preprocessing (median imputation + scaling, mode imputation + one-hot encoding).
- Strict prevention of data leakage (preprocessors fit strictly on X_train).
- Fair comparison of all models on the identical evaluation train/test split.
- Guaranteed immutability of input DataFrame.
- Strictly JSON-serializable outputs.
- Bounded, row-aligned actual/predicted test previews for visualization.
"""

from typing import Dict, Any, List, Optional
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREDICTION_PREVIEW_LIMIT = 500


def _safe_float(val: Any, decimals: int = 4) -> Optional[float]:
    """
    Safely converts a numeric value to a standard Python float.
    Returns None if value is NaN or infinite to guarantee strict JSON serializability.
    """
    if val is None or pd.isna(val) or (isinstance(val, (float, np.floating)) and np.isinf(val)):
        return None
    return round(float(val), decimals)


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles numeric and categorical features independently.
    - Numerical: median imputation + standard scaling.
    - Categorical: most-frequent imputation + one-hot encoding (ignores unseen test categories).
    """
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_regression_model(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and compares Linear Regression, Decision Tree, and Random Forest on a numerical target.
    Never modifies the original DataFrame.
    Never automatically selects or assumes a target column.

    Args:
        df: Input pandas DataFrame.
        target_column: The numerical column to predict (must exist in df).
        feature_columns: Optional list of feature columns to use. If None, uses all remaining columns.
        test_size: Proportion of the dataset to allocate to the test split (default: 0.2).
        random_state: Random state for reproducibility.

    Returns:
        Structured, JSON-serializable regression report. Each model includes
        at most 500 aligned actual/predicted test rows in predictions_preview.

    Raises:
        ValueError: If inputs are invalid, target does not exist, is non-numeric,
                    or data is insufficient.
        TypeError: If df is not a pandas DataFrame.
    """
    # -------------------------------------------------------------
    # 1. Input Validation
    # -------------------------------------------------------------
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty or len(df) == 0:
        raise ValueError("Input DataFrame is empty. Cannot train a model.")

    if not target_column or not isinstance(target_column, str):
        raise ValueError("target_column must be a non-empty string.")
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if not isinstance(test_size, (int, float)) or not (0.05 <= test_size <= 0.5):
        raise ValueError(f"test_size must be a float between 0.05 and 0.5, got {test_size}.")

    # Work strictly on a deep copy to prevent mutation
    work_df = df.copy(deep=True)

    # -------------------------------------------------------------
    # 2. Target Validation (Must be numerical, finite, and non-empty)
    # -------------------------------------------------------------
    target_series = work_df[target_column]

    # Validate target is numeric (and not boolean)
    if not (pd.api.types.is_numeric_dtype(target_series) and not pd.api.types.is_bool_dtype(target_series)):
        raise ValueError(
            f"Target column '{target_column}' must contain valid numerical values, "
            f"but found data type '{target_series.dtype}'."
        )

    # Validate no missing values in target
    if target_series.isna().any():
        missing_count = int(target_series.isna().sum())
        raise ValueError(
            f"Target column '{target_column}' contains {missing_count} missing value(s). "
            "Regression target values must be complete."
        )

    # Validate no infinite values in target
    if np.isinf(target_series).any():
        raise ValueError(
            f"Target column '{target_column}' contains infinite values (+inf / -inf). "
            "Regression target values must be finite numbers."
        )

    # Minimum row requirement for train/test split
    total_rows = int(len(work_df))
    if total_rows < 5:
        raise ValueError(
            f"Dataset has {total_rows} rows, which is insufficient for train/test splitting "
            "(minimum 5 rows required)."
        )

    # -------------------------------------------------------------
    # 3. Feature Selection & Target Exclusion
    # -------------------------------------------------------------
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or len(feature_columns) == 0:
            raise ValueError("feature_columns must be a non-empty list of column names.")

        # Check for non-existent columns
        missing_cols = [c for c in feature_columns if c not in work_df.columns]
        if missing_cols:
            raise ValueError(
                f"Specified feature column(s) {missing_cols} do not exist in DataFrame. "
                f"Available columns: {list(work_df.columns)}"
            )

        # Exclude target_column if user accidentally included it
        candidate_features = [c for c in feature_columns if c != target_column]
        if not candidate_features:
            raise ValueError(f"No feature columns remaining after excluding target column '{target_column}'.")

        selected_features = [str(c) for c in candidate_features]
    else:
        # Use all available columns except the target
        selected_features = [str(c) for c in work_df.columns if c != target_column]
        if not selected_features:
            raise ValueError(f"No feature columns available after excluding target column '{target_column}'.")

    # -------------------------------------------------------------
    # 4. Segregate Numerical and Categorical Features
    # -------------------------------------------------------------
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in selected_features:
        series = work_df[col]
        # Treat numeric dtypes (excluding booleans) as numerical
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col)
            # Replace infinities in numerical features with NaN so imputer handles them safely
            work_df[col] = work_df[col].replace([np.inf, -np.inf], np.nan)
        else:
            categorical_cols.append(col)
            # Ensure categorical features are object-typed strings for robust encoding
            work_df[col] = work_df[col].astype(object)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No valid numerical or categorical features found to train models.")

    # -------------------------------------------------------------
    # 5. Train / Test Splitting
    # -------------------------------------------------------------
    X = work_df[selected_features]
    y = work_df[target_column]
    row_positions = np.arange(len(work_df))

    X_train, X_test, y_train, y_test, _, test_row_positions = train_test_split(
        X, y, row_positions, test_size=test_size, random_state=random_state
    )

    train_rows = int(len(X_train))
    test_rows = int(len(X_test))

    if train_rows < 2 or test_rows < 1:
        raise ValueError(
            f"Split resulted in {train_rows} training samples and {test_rows} test samples, "
            "which is insufficient for model fitting and evaluation."
        )

    # -------------------------------------------------------------
    # 6. Target Summary Statistics
    # -------------------------------------------------------------
    target_summary = {
        "mean": _safe_float(y.mean()),
        "std": _safe_float(y.std()) if len(y) > 1 else 0.0,
        "median": _safe_float(y.median()),
        "min": _safe_float(y.min()),
        "max": _safe_float(y.max()),
    }

    # -------------------------------------------------------------
    # 7. Model Definitions
    # -------------------------------------------------------------
    regressors = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(
            random_state=random_state,
        ),
        "random_forest": RandomForestRegressor(
            random_state=random_state,
            n_estimators=100,
        ),
    }

    model_display_names = {
        "linear_regression": "Linear Regression",
        "decision_tree": "Decision Tree Regressor",
        "random_forest": "Random Forest Regressor",
    }

    # -------------------------------------------------------------
    # 8. Train and Evaluate Models on Identical Split
    # -------------------------------------------------------------
    model_evaluations: Dict[str, Any] = {}
    model_comparison_list: List[Dict[str, Any]] = []
    predictions_preview: Dict[str, Any] = {}

    best_model_key = ""
    best_rmse_score = float("inf")

    for model_key, reg in regressors.items():
        # Fresh preprocessor per model to guarantee pipeline independence
        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", reg),
        ])

        # Preprocessing fit strictly on training data (zero data leakage)
        pipeline.fit(X_train, y_train)

        # Predictions evaluated on the shared test set
        y_pred = pipeline.predict(X_test)
        preview_count = min(len(y_test), PREDICTION_PREVIEW_LIMIT)
        predictions_preview[model_key] = {
            "row_indices": [int(position) for position in test_row_positions[:preview_count]],
            "actual_values": [float(value) for value in y_test.iloc[:preview_count]],
            "predicted_values": [
                float(value) if np.isfinite(value) else None
                for value in y_pred[:preview_count]
            ],
            "total_rows": int(len(y_test)),
            "returned_rows": int(preview_count),
            "truncated": len(y_test) > preview_count,
        }

        mae = _safe_float(mean_absolute_error(y_test, y_pred))
        mse = _safe_float(mean_squared_error(y_test, y_pred))
        raw_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        rmse = _safe_float(raw_rmse)

        # R² score calculation (can be negative; undefined if test set has 0 variance)
        try:
            if y_test.nunique() > 1:
                r2_val = r2_score(y_test, y_pred)
                r2 = _safe_float(r2_val)
            else:
                # If target in test split is constant, R² is undefined
                r2 = None
        except Exception:
            r2 = None

        model_evaluations[model_key] = {
            "name": model_display_names[model_key],
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "metrics_explanation": {
                "mae": "Mean Absolute Error: average magnitude of errors (lower is better).",
                "mse": "Mean Squared Error: average squared errors, penalizing larger deviations (lower is better).",
                "rmse": "Root Mean Squared Error: standard deviation of residuals in original target units (lower is better).",
                "r2_score": "R-squared: proportion of variance explained by features (higher is better; can be negative).",
            },
        }

        model_comparison_list.append({
            "model_key": model_key,
            "model_name": model_display_names[model_key],
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "r2_score": r2,
        })

        # Track best model based on lowest RMSE
        comp_rmse = raw_rmse if not np.isnan(raw_rmse) else float("inf")
        if comp_rmse < best_rmse_score:
            best_rmse_score = comp_rmse
            best_model_key = model_key

    # Sort model comparisons ascending by RMSE (lower is better)
    model_comparison_list.sort(
        key=lambda m: m["rmse"] if m["rmse"] is not None else float("inf")
    )

    best_name = model_display_names.get(best_model_key, best_model_key)
    safe_best_rmse = _safe_float(best_rmse_score)

    explanation = (
        f"Trained 3 regression models on {train_rows} training rows and evaluated on {test_rows} test rows. "
        f"Target '{target_column}' has a mean of {target_summary['mean']} and standard deviation of {target_summary['std']}. "
        f"The top-performing model on this evaluation split is {best_name} with an RMSE of {safe_best_rmse}. "
        "Note: Performance reflects this specific train/test split and does not claim universal superiority."
    )

    return {
        "status": "completed",
        "target_column": target_column,
        "selected_feature_columns": selected_features,
        "numerical_features": numeric_cols,
        "categorical_features": categorical_cols,
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "target_summary": target_summary,
        "models": model_evaluations,
        "best_model": {
            "model_key": best_model_key,
            "model_name": best_name,
            "best_rmse": safe_best_rmse,
            "selection_metric": "rmse (lowest)",
            "explanation": (
                f"{best_name} achieved the lowest Root Mean Squared Error (RMSE: {safe_best_rmse}) "
                "among the evaluated models on this test split."
            ),
            "disclaimer": (
                "Selected based on lowest RMSE on the evaluation test split. "
                "This does not guarantee optimal performance on unobserved future distributions."
            ),
        },
        "model_comparison": model_comparison_list,
        "predictions_preview": predictions_preview,
        "explanation": explanation,
    }
