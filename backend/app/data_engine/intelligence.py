"""
Data Engine - Intelligence Module
Provides rule-based dataset intelligence:
1. Column role detection (numeric, categorical, datetime-like, boolean, potential_id)
2. Target column candidate identification (never automatically selecting a final target)
3. Recommendation of suitable analysis options (EDA, Anomaly Detection, Classification, Regression, Clustering)
"""

from typing import Dict, Any, List
import re
import pandas as pd


# Regex heuristic matching common identifier column patterns
ID_NAME_PATTERN = re.compile(
    r"(^|_|-)(id|uuid|guid|key|code|pk|identifier)($|_|-)",
    re.IGNORECASE,
)

# Common keywords suggesting potential target or outcome variables
TARGET_NAME_KEYWORDS = {
    "target", "label", "outcome", "churn", "status", "class",
    "result", "price", "salary", "revenue", "converted", "default",
    "survived", "purchased", "fraud", "risk", "score",
}


def _is_datetime_like(series: pd.Series) -> bool:
    """
    Checks if a series is datetime-typed or string values that represent dates.
    Avoids false positives on plain numeric columns.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Numeric series should not be guessed as datetime here
    if pd.api.types.is_numeric_dtype(series):
        return False

    clean = series.dropna()
    if clean.empty or len(clean) == 0:
        return False

    sample = clean.head(10).astype(str)

    # Do not treat very short numbers or single digits as dates
    if all(s.strip().isdigit() and len(s.strip()) < 4 for s in sample):
        return False

    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().all():
            return True
    except Exception:
        pass

    return False


def _is_boolean_like(series: pd.Series) -> bool:
    """
    Checks if a series is boolean or contains values strictly representing binary truth.
    """
    if pd.api.types.is_bool_dtype(series):
        return True

    clean = series.dropna()
    if clean.empty:
        return False

    unique_vals = set(clean.unique())

    # Check for boolean objects or strings
    if unique_vals.issubset({True, False}):
        return True

    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        str_vals = {str(v).strip().lower() for v in unique_vals}
        if str_vals.issubset({"true", "false", "yes", "no", "t", "f"}):
            return len(str_vals) <= 2

    return False


def _check_potential_id(col_name: str, series: pd.Series, total_rows: int) -> bool:
    """
    Heuristic to determine if a column behaves like an entity identifier.
    Excludes datetime columns, float measurements, and non-ID-named features.
    """
    if total_rows < 2:
        return False

    clean = series.dropna()
    if clean.empty:
        return False

    # Datetime columns and float measurements are not entity identifiers
    if _is_datetime_like(series):
        return False

    name_matches = bool(ID_NAME_PATTERN.search(col_name) or col_name.lower().endswith("id"))

    # Continuous float measurements without ID names are not IDs
    if pd.api.types.is_float_dtype(series) and not name_matches:
        return False

    unique_count = int(clean.nunique())
    uniqueness_pct = (unique_count / total_rows) * 100

    # 1. High uniqueness AND name matches identifier pattern
    if name_matches and uniqueness_pct >= 90.0:
        return True

    # 2. Integer index/id column with explicit ID naming
    if pd.api.types.is_integer_dtype(series) and name_matches and uniqueness_pct >= 90.0:
        return True

    # 3. String columns with 100% uniqueness and ID-matching names on >= 5 rows
    if (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)) and name_matches:
        if uniqueness_pct >= 90.0:
            return True

    return False


def _detect_column_role(col_name: str, series: pd.Series, total_rows: int) -> Dict[str, Any]:
    """
    Detects the primary role and structural characteristics of a single column.
    """
    clean = series.dropna()
    unique_count = int(clean.nunique())
    uniqueness_pct = round((unique_count / total_rows * 100), 2) if total_rows > 0 else 0.0
    is_id = _check_potential_id(col_name, series, total_rows)

    if clean.empty:
        primary_role = "empty"
        reason = "Column contains no non-null values."
    elif _is_datetime_like(series):
        primary_role = "datetime"
        reason = "Column contains temporal or timestamp-like values."
    elif _is_boolean_like(series):
        primary_role = "boolean"
        reason = "Column contains binary or truth-like values."
    elif is_id:
        primary_role = "potential_id"
        reason = (
            f"High uniqueness ({uniqueness_pct}%) "
            + ("and identifier naming pattern. " if ID_NAME_PATTERN.search(col_name) else ". ")
            + "Marked as a potential identifier (not confirmed)."
        )
    elif pd.api.types.is_numeric_dtype(series):
        primary_role = "numeric"
        reason = "Column contains numerical values."
    else:
        primary_role = "categorical"
        reason = "Column contains textual or categorical values."

    return {
        "column": col_name,
        "role": primary_role,
        "data_type": str(series.dtype),
        "is_potential_id": is_id,
        "unique_count": unique_count,
        "uniqueness_percentage": uniqueness_pct,
        "reason": reason,
    }


def _identify_possible_targets(
    df: pd.DataFrame,
    column_roles: Dict[str, Dict[str, Any]],
    total_rows: int,
) -> List[Dict[str, Any]]:
    """
    Identifies reasonable candidate columns that could serve as prediction targets.
    Excludes potential ID columns, completely empty/constant columns, and high-cardinality text.
    Never selects a final target.
    """
    candidates: List[Dict[str, Any]] = []

    if total_rows < 5:
        return candidates

    for col in df.columns:
        col_str = str(col)
        role_info = column_roles.get(col_str, {})
        role = role_info.get("role")
        is_id = role_info.get("is_potential_id", False)
        unique_count = role_info.get("unique_count", 0)

        # 1. Exclude obvious identifiers, empty columns, and datetime columns
        if is_id or role in ("empty", "datetime", "potential_id"):
            continue

        # 2. Exclude constant columns (unique count <= 1)
        if unique_count <= 1:
            continue

        col_lower = col_str.lower()
        has_target_keyword = any(kw in col_lower for kw in TARGET_NAME_KEYWORDS)

        # 3. Categorical / Classification Candidate:
        # Suitable if discrete (2 to 20 unique values) and not high-cardinality
        if role in ("categorical", "boolean") or (role == "numeric" and unique_count <= 10):
            if 2 <= unique_count <= 20:
                target_type = "binary" if unique_count == 2 else "multiclass"
                confidence = "high" if has_target_keyword else "medium"
                candidates.append({
                    "column": col_str,
                    "suggested_task": "classification",
                    "target_type": target_type,
                    "unique_values_count": unique_count,
                    "confidence": confidence,
                    "reason": (
                        f"Contains {unique_count} distinct categories. "
                        + ("Column name suggests an outcome variable. " if has_target_keyword else "")
                        + "Candidate for classification tasks."
                    ),
                    "status": "candidate_only",
                })
                continue

        # 4. Numeric / Regression Candidate:
        # Suitable if numeric with more than 10 unique values and not an ID
        if role == "numeric" and unique_count > 10:
            confidence = "high" if has_target_keyword else "medium"
            candidates.append({
                "column": col_str,
                "suggested_task": "regression",
                "target_type": "continuous_numeric",
                "unique_values_count": unique_count,
                "confidence": confidence,
                "reason": (
                    f"Continuous numerical column with {unique_count} distinct values. "
                    + ("Column name suggests a target quantity. " if has_target_keyword else "")
                    + "Candidate for regression tasks."
                ),
                "status": "candidate_only",
            })

    return candidates


def _recommend_analyses(
    total_rows: int,
    total_columns: int,
    column_roles: Dict[str, Dict[str, Any]],
    possible_targets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Evaluates dataset characteristics and recommends suitable analysis workflows.
    Includes prerequisites, reasons, limitations, and target requirements.
    """
    recommendations: List[Dict[str, Any]] = []

    # Count roles
    numeric_count = sum(1 for r in column_roles.values() if r.get("role") == "numeric")
    classification_targets = [c["column"] for c in possible_targets if c.get("suggested_task") == "classification"]
    regression_targets = [c["column"] for c in possible_targets if c.get("suggested_task") == "regression"]

    # 1. Exploratory Data Analysis (EDA)
    eda_recommended = total_rows > 0 and total_columns > 0
    recommendations.append({
        "analysis_type": "exploratory_data_analysis",
        "name": "Exploratory Data Analysis (EDA)",
        "is_recommended": eda_recommended,
        "description": "Inspects distributions, correlations, missingness, and statistical patterns across features.",
        "suitability_reason": (
            f"Dataset has {total_rows} row(s) and {total_columns} column(s). EDA provides foundational understanding."
            if eda_recommended else "Dataset is empty; cannot perform EDA."
        ),
        "prerequisites_and_limitations": "None. Suitable for any non-empty dataset.",
        "requires_target": False,
        "candidate_targets": [],
    })

    # 2. Anomaly Detection
    anomaly_recommended = numeric_count >= 1 and total_rows >= 10
    recommendations.append({
        "analysis_type": "anomaly_detection",
        "name": "Anomaly Detection",
        "is_recommended": anomaly_recommended,
        "description": "Detects rare outliers, unusual data points, or abnormal behaviors in multidimensional space.",
        "suitability_reason": (
            f"Dataset contains {numeric_count} numeric feature(s) and {total_rows} rows, sufficient for outlier detection."
            if anomaly_recommended
            else "Requires at least 1 numeric feature and a minimum of 10 rows to detect statistical anomalies."
        ),
        "prerequisites_and_limitations": "Unsupervised. Requires numeric features; sensitive to unscaled variables.",
        "requires_target": False,
        "candidate_targets": [],
    })

    # 3. Classification
    class_recommended = len(classification_targets) > 0 and total_columns >= 2 and total_rows >= 10
    recommendations.append({
        "analysis_type": "classification",
        "name": "Classification",
        "is_recommended": class_recommended,
        "description": "Predicts categorical classes, states, or labels based on input feature patterns.",
        "suitability_reason": (
            f"Found {len(classification_targets)} potential discrete target candidate(s) ({', '.join(classification_targets[:3])})."
            if class_recommended
            else "No suitable discrete target candidate found, or dataset is too small (<10 rows or <2 columns)."
        ),
        "prerequisites_and_limitations": "Supervised. Requires user selection of a discrete target column.",
        "requires_target": True,
        "candidate_targets": classification_targets,
    })

    # 4. Regression
    reg_recommended = len(regression_targets) > 0 and total_columns >= 2 and total_rows >= 10
    recommendations.append({
        "analysis_type": "regression",
        "name": "Regression",
        "is_recommended": reg_recommended,
        "description": "Predicts a continuous numerical value or quantity based on feature relationships.",
        "suitability_reason": (
            f"Found {len(regression_targets)} potential continuous target candidate(s) ({', '.join(regression_targets[:3])})."
            if reg_recommended
            else "No suitable continuous numerical target found, or dataset is too small (<10 rows or <2 columns)."
        ),
        "prerequisites_and_limitations": "Supervised. Requires user selection of a continuous numeric target column.",
        "requires_target": True,
        "candidate_targets": regression_targets,
    })

    # 5. Clustering
    clustering_recommended = numeric_count >= 2 and total_rows >= 15
    recommendations.append({
        "analysis_type": "clustering",
        "name": "Clustering",
        "is_recommended": clustering_recommended,
        "description": "Groups similar records into natural segments or clusters without requiring target labels.",
        "suitability_reason": (
            f"Dataset has {numeric_count} numeric features and {total_rows} rows, suitable for discovering groupings."
            if clustering_recommended
            else "Requires at least 2 numeric features and a minimum of 15 rows to form meaningful multidimensional clusters."
        ),
        "prerequisites_and_limitations": "Unsupervised. Requires multiple numeric features; results depend on feature scaling.",
        "requires_target": False,
        "candidate_targets": [],
    })

    return recommendations


def analyze_dataset_intelligence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects a Pandas DataFrame structure to extract rule-based dataset intelligence:
    1. Identifies column roles (numeric, categorical, datetime, boolean, potential_id).
    2. Identifies possible target columns for supervised tasks without confirming or selecting one.
    3. Recommends suitable analysis options (EDA, Anomaly Detection, Classification, Regression, Clustering).

    Args:
        df: Input pandas DataFrame to analyze. Original DataFrame is not modified.

    Returns:
        Structured, JSON-serializable intelligence report dictionary.

    Raises:
        ValueError: If input DataFrame is None.
        TypeError: If input is not a pandas DataFrame.
    """
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    total_rows = int(len(df))
    total_columns = int(len(df.columns))

    # 1. Column Role Detection
    column_roles: Dict[str, Dict[str, Any]] = {}
    role_counts: Dict[str, int] = {
        "numeric": 0,
        "categorical": 0,
        "datetime": 0,
        "boolean": 0,
        "potential_id": 0,
        "empty": 0,
    }

    for col in df.columns:
        col_str = str(col)
        role_info = _detect_column_role(col_str, df[col], total_rows)
        column_roles[col_str] = role_info
        role = role_info["role"]
        if role in role_counts:
            role_counts[role] += 1

    # 2. Identify Possible Target Columns
    possible_targets = _identify_possible_targets(df, column_roles, total_rows)

    # 3. Recommend Possible Analysis Options
    recommended_analyses = _recommend_analyses(
        total_rows=total_rows,
        total_columns=total_columns,
        column_roles=column_roles,
        possible_targets=possible_targets,
    )

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "role_counts": role_counts,
        "column_roles": column_roles,
        "possible_targets": possible_targets,
        "recommended_analyses": recommended_analyses,
    }
