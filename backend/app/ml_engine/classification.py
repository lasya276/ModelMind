"""
ML Engine - Classification Module
Trains, evaluates, and compares classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Features:
- Pure Scikit-learn pipelines with ColumnTransformer.
- Separate numeric and categorical preprocessing (median imputation + scaling, mode imputation + one-hot encoding).
- Strict prevention of data leakage (preprocessors fit strictly on X_train).
- Fair comparison of all models on the identical evaluation train/test split.
- Guaranteed immutability of input DataFrame.
- Strictly JSON-serializable outputs.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


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


def train_classification_model(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains and compares Logistic Regression, Decision Tree, and Random Forest on a user-selected target.
    Never modifies the original DataFrame.
    Never automatically selects or assumes a target column.

    Args:
        df: Input pandas DataFrame.
        target_column: The column to predict (must exist in df).
        feature_columns: Optional list of feature columns to use. If None, uses all remaining columns.
        test_size: Proportion of the dataset to allocate to the test split (default: 0.2).
        random_state: Random state for reproducibility.

    Returns:
        Structured, JSON-serializable classification report.

    Raises:
        ValueError: If inputs are invalid, target does not exist, or data is insufficient.
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

    if not (0.05 <= test_size <= 0.5):
        raise ValueError(f"test_size must be between 0.05 and 0.5, got {test_size}.")

    # Work strictly on a deep copy to prevent mutation
    work_df = df.copy(deep=True)

    # Filter out rows with null target values (supervised classification requires labels)
    valid_target_mask = work_df[target_column].notna()
    work_df = work_df[valid_target_mask]
    if len(work_df) == 0:
        raise ValueError(f"Target column '{target_column}' contains only missing values.")

    # Validate class count
    y_raw = work_df[target_column]
    raw_classes = y_raw.unique()
    if len(raw_classes) < 2:
        raise ValueError(
            f"Classification requires at least 2 distinct classes in target column '{target_column}', "
            f"found {len(raw_classes)}: {[str(c) for c in raw_classes]}."
        )

    classification_type = "binary" if len(raw_classes) == 2 else "multiclass"

    # Require minimum samples for a viable train/test split
    if len(work_df) < 5:
        raise ValueError(
            f"Dataset has {len(work_df)} rows, which is insufficient for train/test splitting "
            "(minimum 5 rows required)."
        )

    # -------------------------------------------------------------
    # 2. Feature Selection & Target Exclusion
    # -------------------------------------------------------------
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or len(feature_columns) == 0:
            raise ValueError("feature_columns must be a non-empty list of column names.")

        # Exclude target_column if user accidentally included it
        candidate_features = [c for c in feature_columns if c != target_column]
        if not candidate_features:
            raise ValueError(f"No feature columns remaining after excluding target column '{target_column}'.")

        missing_cols = [c for c in candidate_features if c not in work_df.columns]
        if missing_cols:
            raise ValueError(
                f"Specified feature column(s) {missing_cols} do not exist in DataFrame. "
                f"Available columns: {list(work_df.columns)}"
            )

        selected_features = [str(c) for c in candidate_features]
    else:
        # Use all available columns except the target
        selected_features = [str(c) for c in work_df.columns if c != target_column]
        if not selected_features:
            raise ValueError(f"No feature columns available after excluding target column '{target_column}'.")

    # -------------------------------------------------------------
    # 3. Segregate Numerical and Categorical Features
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
    # 4. Train / Test Splitting
    # -------------------------------------------------------------
    X = work_df[selected_features]
    y = work_df[target_column]

    # Convert y values to strings for consistent class label representation
    y_str = y.astype(str)
    classes_list = sorted(list(y_str.unique()))

    # Stratify when feasible (each class has at least 2 samples and test split allows it)
    class_counts = y_str.value_counts()
    min_class_count = int(class_counts.min())
    stratify_split = None
    is_stratified = False

    if min_class_count >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_str, test_size=test_size, random_state=random_state, stratify=y_str
            )
            is_stratified = True
        except ValueError:
            # Fall back safely if sample size per class is too sparse for strict stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_str, test_size=test_size, random_state=random_state, stratify=None
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_str, test_size=test_size, random_state=random_state, stratify=None
        )

    # -------------------------------------------------------------
    # 5. Model Definitions
    # -------------------------------------------------------------
    classifiers = {
        "logistic_regression": LogisticRegression(
            random_state=random_state,
            max_iter=1000,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            random_state=random_state,
            n_estimators=100,
        ),
    }

    model_display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree Classifier",
        "random_forest": "Random Forest Classifier",
    }

    # -------------------------------------------------------------
    # 6. Train and Evaluate Models on Identical Split
    # -------------------------------------------------------------
    model_evaluations: Dict[str, Any] = {}
    confusion_matrices: Dict[str, Any] = {}
    model_comparison_list: List[Dict[str, Any]] = []

    best_model_key = ""
    best_f1_score = -1.0

    for model_key, clf in classifiers.items():
        # Fresh preprocessor per model to guarantee pipeline independence
        preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # Preprocessing is fit strictly on training data (zero data leakage)
        pipeline.fit(X_train, y_train)

        # Predictions evaluated on the shared test set
        y_pred = pipeline.predict(X_test)

        acc = _safe_float(accuracy_score(y_test, y_pred))
        prec = _safe_float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = _safe_float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = _safe_float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # Confusion matrix with explicit class labels
        cm = confusion_matrix(y_test, y_pred, labels=classes_list)
        cm_serializable = [[int(cell) for cell in row] for row in cm]

        model_evaluations[model_key] = {
            "name": model_display_names[model_key],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "averaging_method": "weighted",
        }

        confusion_matrices[model_key] = {
            "labels": classes_list,
            "matrix": cm_serializable,
        }

        model_comparison_list.append({
            "model_key": model_key,
            "model_name": model_display_names[model_key],
            "f1_score": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
        })

        # Track best model based on weighted F1 score
        current_f1 = f1 if f1 is not None else 0.0
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_model_key = model_key

    # Sort model comparisons descending by F1-score
    model_comparison_list.sort(key=lambda m: m["f1_score"] if m["f1_score"] is not None else 0.0, reverse=True)

    # Class distribution
    class_distribution = {str(k): int(v) for k, v in class_counts.items()}

    # Explanation of results
    best_name = model_display_names.get(best_model_key, best_model_key)
    explanation = (
        f"Trained 3 classification models on {len(X_train)} training rows and evaluated on {len(X_test)} test rows. "
        f"Target '{target_column}' has {len(classes_list)} classes ({', '.join(classes_list)}). "
        f"The top-performing model on this evaluation split is {best_name} with a weighted F1-score of {best_f1_score:.4f}. "
        "Note: Performance reflects this specific train/test split and does not claim universal superiority."
    )

    return {
        "status": "completed",
        "target_column": target_column,
        "classification_type": classification_type,
        "selected_feature_columns": selected_features,
        "numerical_features": numeric_cols,
        "categorical_features": categorical_cols,
        "total_rows": int(len(work_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "is_stratified": is_stratified,
        "class_distribution": class_distribution,
        "models": model_evaluations,
        "confusion_matrices": confusion_matrices,
        "best_model": {
            "model_key": best_model_key,
            "model_name": best_name,
            "f1_score": best_f1_score,
            "selection_metric": "f1_score (weighted)",
            "disclaimer": (
                "Selected based on highest weighted F1-score on the evaluation test split. "
                "This does not guarantee optimal performance on unobserved future distributions."
            ),
        },
        "model_comparison": model_comparison_list,
        "explanation": explanation,
    }
