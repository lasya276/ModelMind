"""
ML Engine - Clustering Module
Groups observations into natural clusters using Scikit-learn's K-Means algorithm:
- User-specified or automatic numerical feature selection with identifier exclusion.
- Missing value median imputation and infinite value handling.
- Feature standardization via StandardScaler.
- Inverse transformation of cluster centroids to original feature scale.
- Comprehensive cluster summaries: sizes, assignments, centroids, inertia, and silhouette score.
- Guaranteed immutability of input DataFrame.
- Strict JSON serializability.
"""

from typing import Dict, Any, List, Optional
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as sk_silhouette_score


# Regex heuristic to recognize identifier columns that should not be used as features
ID_NAME_PATTERN = re.compile(
    r"(^|_|-)(id|uuid|guid|key|code|pk|identifier)($|_|-)",
    re.IGNORECASE,
)


class ClusterDict(dict):
    """
    Dictionary supporting lookup by both string ('0') and integer (0) keys,
    while remaining directly serializable by standard json.dumps.
    """
    def __getitem__(self, key: Any) -> Any:
        if key in self:
            return super().__getitem__(key)
        if isinstance(key, int) and str(key) in self:
            return super().__getitem__(str(key))
        if isinstance(key, str) and key.isdigit() and int(key) in self:
            return super().__getitem__(int(key))
        return super().__getitem__(key)


def _safe_float(val: Any, decimals: int = 4) -> Optional[float]:
    """
    Safely converts a numeric value to a standard Python float rounded to `decimals`.
    Returns None if value is NaN, infinite, or None to guarantee JSON serializability.
    """
    if val is None or pd.isna(val) or (isinstance(val, (float, np.floating)) and np.isinf(val)):
        return None
    return round(float(val), decimals)


def _is_potential_id_column(col_name: str, series: pd.Series, total_rows: int) -> bool:
    """
    Checks if a numeric column is likely an entity identifier or sequential index.
    """
    if total_rows < 2:
        return False

    name_matches = bool(ID_NAME_PATTERN.search(col_name) or col_name.lower().endswith("id"))
    clean = series.dropna()
    if clean.empty:
        return False

    uniqueness_pct = (clean.nunique() / total_rows) * 100

    if name_matches and uniqueness_pct >= 90.0:
        return True

    return False


def perform_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    feature_columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Performs K-Means clustering on numerical features in a Pandas DataFrame.
    Never modifies the original DataFrame.
    Never silently drops rows.

    Args:
        df: Input pandas DataFrame to cluster.
        n_clusters: Number of clusters to form (must be an integer >= 2 and < len(df)).
        feature_columns: Optional list of numerical column names to use.
                         If None, automatically selects suitable numerical features
                         while excluding potential identifier columns.
        random_state: Random state for reproducibility.

    Returns:
        Structured, JSON-serializable clustering report dictionary.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If input DataFrame is None, empty, has insufficient rows,
                    cluster count is invalid, feature columns are invalid/missing,
                    selected features are non-numeric or all-null, or distinct
                    sample count is insufficient to form n_clusters.
    """
    # -------------------------------------------------------------
    # 1. Input DataFrame Validation
    # -------------------------------------------------------------
    if df is None:
        raise ValueError("Input DataFrame cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty or len(df) == 0:
        raise ValueError("Input DataFrame is empty. Cannot perform clustering.")

    total_rows = int(len(df))

    # Minimum row count required for clustering: at least 3 rows
    if total_rows < 3:
        raise ValueError(
            f"Dataset has {total_rows} row(s), which is insufficient for clustering "
            "(minimum 3 rows required)."
        )

    # -------------------------------------------------------------
    # 2. Cluster Count (n_clusters) Validation
    # -------------------------------------------------------------
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters must be an integer.")
    n_clusters = int(n_clusters)

    if n_clusters < 2:
        raise ValueError(f"n_clusters must be at least 2, got {n_clusters}.")

    if n_clusters >= total_rows:
        raise ValueError(
            f"n_clusters ({n_clusters}) must be strictly less than the number of samples ({total_rows})."
        )

    # -------------------------------------------------------------
    # 3. Feature Selection & Validation
    # -------------------------------------------------------------
    if feature_columns is not None:
        if not isinstance(feature_columns, list) or len(feature_columns) == 0:
            raise ValueError("feature_columns must be a non-empty list of column names.")

        missing_cols = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Specified feature column(s) {missing_cols} do not exist in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        non_numeric = [
            c for c in feature_columns
            if not (pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]))
        ]
        if non_numeric:
            raise ValueError(
                f"Specified feature column(s) {non_numeric} are not numerical. "
                "Clustering requires numerical features."
            )

        all_null = [c for c in feature_columns if df[c].isna().all()]
        if all_null:
            raise ValueError(
                f"Specified feature column(s) {all_null} contain only missing values."
            )

        selected_features = [str(c) for c in feature_columns]
    else:
        # Automatically select numeric features, excluding potential ID columns,
        # all-null columns, and zero-variance constant columns
        selected_features = []
        for col in df.columns:
            col_str = str(col)
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            if is_numeric:
                clean_s = series.dropna()
                if not clean_s.empty and clean_s.nunique() > 1:
                    if not _is_potential_id_column(col_str, series, total_rows):
                        selected_features.append(col_str)

        if not selected_features:
            raise ValueError("No suitable numerical features found for clustering.")

    # -------------------------------------------------------------
    # 4. Data Preprocessing (Immutability, Imputation & Scaling)
    # -------------------------------------------------------------
    # Guarantee Immutability: work strictly on a deep copy of selected features
    X = df[selected_features].copy(deep=True)

    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Impute missing values with column median
    for col in selected_features:
        clean_col = X[col].dropna()
        if clean_col.empty:
            raise ValueError(f"Feature column '{col}' contains only missing values.")
        median_val = float(clean_col.median())
        X[col] = X[col].fillna(median_val)

    # Verify distinct sample count
    distinct_sample_count = int(len(X.drop_duplicates()))
    if distinct_sample_count < n_clusters:
        raise ValueError(
            f"Number of distinct samples ({distinct_sample_count}) is less than the number of clusters ({n_clusters}). "
            f"Cannot form {n_clusters} distinct clusters."
        )

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------------------
    # 5. K-Means Fitting & Cluster Assignments
    # -------------------------------------------------------------
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    raw_labels = kmeans.fit_predict(X_scaled)

    # Row-level cluster assignments (0 to n_clusters - 1)
    cluster_assignments = [int(lbl) for lbl in raw_labels]

    # Cluster sizes
    cluster_sizes = ClusterDict({
        str(c): int((raw_labels == c).sum())
        for c in range(n_clusters)
    })

    # -------------------------------------------------------------
    # 6. Cluster Centroids (Inverted to Original Feature Scale)
    # -------------------------------------------------------------
    scaled_centroids = kmeans.cluster_centers_
    orig_scale_centroids = scaler.inverse_transform(scaled_centroids)

    centroids = ClusterDict({
        str(c): {
            selected_features[i]: _safe_float(orig_scale_centroids[c, i])
            for i in range(len(selected_features))
        }
        for c in range(n_clusters)
    })

    # -------------------------------------------------------------
    # 7. Evaluation: Inertia & Silhouette Score
    # -------------------------------------------------------------
    inertia = _safe_float(kmeans.inertia_)

    # Silhouette score calculation
    unique_labels = set(raw_labels)
    if len(unique_labels) < 2:
        silhouette_score = None
        silhouette_explanation = "Silhouette score cannot be computed because fewer than 2 distinct clusters were formed."
    elif len(unique_labels) >= total_rows:
        silhouette_score = None
        silhouette_explanation = "Silhouette score cannot be computed because the number of clusters equals or exceeds sample count."
    else:
        try:
            raw_sil = sk_silhouette_score(X_scaled, raw_labels)
            silhouette_score = _safe_float(raw_sil)
            silhouette_explanation = (
                "Silhouette score measures how well-separated clusters are "
                "(ranges from -1.0 to +1.0; higher indicates better cohesion and separation)."
            )
        except Exception as e:
            silhouette_score = None
            silhouette_explanation = f"Silhouette score could not be calculated: {str(e)}"

    # -------------------------------------------------------------
    # 8. Report Compilation
    # -------------------------------------------------------------
    sizes_str = ", ".join([f"Cluster {k}: {v}" for k, v in cluster_sizes.items()])
    sil_str = f"Silhouette score: {silhouette_score}. " if silhouette_score is not None else f"Silhouette score unavailable ({silhouette_explanation}). "

    explanation = (
        f"Grouped {total_rows} observations into {n_clusters} clusters using K-Means clustering across "
        f"{len(selected_features)} numerical feature(s) ({', '.join(selected_features)}) with StandardScaler normalization. "
        f"Cluster sizes: {sizes_str}. "
        f"Total inertia (within-cluster sum of squared distances) is {inertia}. "
        f"{sil_str}"
        "Note: Cluster labels (0 to n-1) are categorical group identifiers representing natural partitions in feature space. "
        "K-Means assumes convex, spherical clusters of comparable size. "
        "The chosen cluster count was specified by the user and is not claimed to be automatically optimal."
    )

    return {
        "status": "completed",
        "selected_features": selected_features,
        "features_used": selected_features,
        "total_rows": total_rows,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "centroids": centroids,
        "cluster_assignments": cluster_assignments,
        "inertia": inertia,
        "silhouette_score": silhouette_score,
        "silhouette_explanation": silhouette_explanation,
        "explanation": explanation,
    }
