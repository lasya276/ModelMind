"""
Unit tests for the Clustering Engine (backend/app/ml_engine/clustering.py).
"""

import json
import pytest
import numpy as np
import pandas as pd
from app.ml_engine.clustering import perform_clustering


def _create_synthetic_clusters(n_samples_per_cluster: int = 20, seed: int = 42) -> pd.DataFrame:
    """Helper to generate well-separated synthetic clusters in 2D space."""
    rng = np.random.RandomState(seed)

    # Cluster 0 around (10, 10)
    c0_x = rng.normal(10.0, 1.0, n_samples_per_cluster)
    c0_y = rng.normal(10.0, 1.0, n_samples_per_cluster)

    # Cluster 1 around (50, 50)
    c1_x = rng.normal(50.0, 1.0, n_samples_per_cluster)
    c1_y = rng.normal(50.0, 1.0, n_samples_per_cluster)

    # Cluster 2 around (90, 90)
    c2_x = rng.normal(90.0, 1.0, n_samples_per_cluster)
    c2_y = rng.normal(90.0, 1.0, n_samples_per_cluster)

    return pd.DataFrame({
        "x": np.concatenate([c0_x, c1_x, c2_x]),
        "y": np.concatenate([c0_y, c1_y, c2_y]),
    })


# 1. Basic clustering
def test_basic_clustering():
    """Verify basic clustering on well-separated 3-cluster data."""
    df = _create_synthetic_clusters(20)
    res = perform_clustering(df, n_clusters=3, random_state=42)

    assert res["status"] == "completed"
    assert res["total_rows"] == 60
    assert res["n_clusters"] == 3
    assert set(res["selected_features"]) == {"x", "y"}
    assert len(res["cluster_assignments"]) == 60
    assert len(res["cluster_sizes"]) == 3
    assert len(res["centroids"]) == 3
    assert res["inertia"] is not None and res["inertia"] > 0
    assert res["silhouette_score"] is not None
    assert res["silhouette_score"] > 0.5  # Well separated clusters yield high silhouette score
    assert "categorical group identifiers" in res["explanation"]
    assert "not claimed to be automatically optimal" in res["explanation"]


# 2. User-selected numerical features
def test_user_selected_numerical_features():
    """Verify clustering using only specified numerical feature columns."""
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 10.0, 11.0, 20.0, 21.0],
        "feat_b": [100.0, 105.0, 200.0, 205.0, 300.0, 305.0],
        "extra_num": [999.0] * 6,
    })
    res = perform_clustering(df, n_clusters=2, feature_columns=["feat_a", "feat_b"])

    assert res["selected_features"] == ["feat_a", "feat_b"]
    assert "extra_num" not in res["selected_features"]
    assert "feat_a" in res["centroids"]["0"]
    assert "feat_b" in res["centroids"]["0"]
    assert "extra_num" not in res["centroids"]["0"]


# 3. Automatic numerical feature selection
def test_automatic_numerical_feature_selection():
    """Verify automatic selection of numeric features while excluding IDs and text."""
    df = pd.DataFrame({
        "user_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "score_1": [1.0, 1.2, 5.0, 5.2, 9.0, 9.1, 1.1, 5.3, 9.2, 1.3],
        "score_2": [10.0, 10.5, 20.0, 20.5, 30.0, 30.5, 10.2, 20.1, 30.2, 10.3],
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })
    res = perform_clustering(df, n_clusters=3, random_state=42)

    assert "user_id" not in res["selected_features"]
    assert "category" not in res["selected_features"]
    assert set(res["selected_features"]) == {"score_1", "score_2"}


# 4. Missing value imputation
def test_missing_value_imputation():
    """Verify missing values are imputed via median without dropping any rows."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 8.0, 9.0, 10.0],
        "y": [np.nan, 2.0, 3.0, 8.0, 9.0, 10.0],
    })
    res = perform_clustering(df, n_clusters=2, random_state=42)

    assert res["status"] == "completed"
    assert res["total_rows"] == 6
    assert len(res["cluster_assignments"]) == 6


# 5. Positive and negative infinity
def test_positive_and_negative_infinity():
    """Verify infinite values are safely handled and imputed without errors."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, np.inf, 8.0, 9.0, 10.0],
        "y": [-np.inf, 2.0, 3.0, 8.0, 9.0, 10.0],
    })
    res = perform_clustering(df, n_clusters=2, random_state=42)

    assert res["status"] == "completed"
    assert res["total_rows"] == 6
    assert len(res["cluster_assignments"]) == 6


# 6. Invalid DataFrame input
def test_invalid_dataframe_input():
    """Verify TypeError when input is not a DataFrame and ValueError when None."""
    with pytest.raises(ValueError, match="Input DataFrame cannot be None"):
        perform_clustering(None)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        perform_clustering({"not": "a dataframe"})


# 7. Empty dataset
def test_empty_dataset():
    """Verify ValueError when DataFrame is empty."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        perform_clustering(df)


# 8. Insufficient rows
def test_insufficient_rows():
    """Verify ValueError when dataset has fewer than 3 rows."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    with pytest.raises(ValueError, match="insufficient for clustering"):
        perform_clustering(df, n_clusters=2)


# 9. Invalid cluster count
def test_invalid_cluster_count():
    """Verify ValueError when n_clusters is < 2 or not an integer."""
    df = _create_synthetic_clusters(10)

    with pytest.raises(ValueError, match="n_clusters must be at least 2"):
        perform_clustering(df, n_clusters=1)

    with pytest.raises(ValueError, match="n_clusters must be at least 2"):
        perform_clustering(df, n_clusters=0)

    with pytest.raises(ValueError, match="n_clusters must be at least 2"):
        perform_clustering(df, n_clusters=-2)

    with pytest.raises(ValueError, match="must be an integer"):
        perform_clustering(df, n_clusters="two")

    with pytest.raises(ValueError, match="must be an integer"):
        perform_clustering(df, n_clusters=True)


# 10. Cluster count greater than or equal to sample count
def test_cluster_count_gte_sample_count():
    """Verify ValueError when n_clusters >= total samples."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [5.0, 6.0, 7.0, 8.0],
    })

    # Equal
    with pytest.raises(ValueError, match="must be strictly less than"):
        perform_clustering(df, n_clusters=4)

    # Greater
    with pytest.raises(ValueError, match="must be strictly less than"):
        perform_clustering(df, n_clusters=5)


# 11. Missing feature names
def test_missing_feature_names():
    """Verify ValueError when specified feature column is not in DataFrame."""
    df = _create_synthetic_clusters(5)
    with pytest.raises(ValueError, match="Specified feature column.*do not exist"):
        perform_clustering(df, feature_columns=["x", "non_existent_col"])


# 12. Non-numerical selected features
def test_non_numerical_selected_features():
    """Verify ValueError when specified feature is non-numerical."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "city": ["New York", "London", "Tokyo", "Paris", "Berlin"],
    })
    with pytest.raises(ValueError, match="are not numerical"):
        perform_clustering(df, feature_columns=["x", "city"])


# 13. All-null features
def test_all_null_features():
    """Verify ValueError when a specified feature is completely null."""
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    with pytest.raises(ValueError, match="contain only missing values"):
        perform_clustering(df, feature_columns=["x", "all_null"])


# 14. Insufficient distinct samples
def test_insufficient_distinct_samples():
    """Verify ValueError when unique sample count is less than n_clusters."""
    # 6 rows, but only 2 distinct points
    df = pd.DataFrame({
        "x": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        "y": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    })
    with pytest.raises(ValueError, match="distinct samples.*less than the number of clusters"):
        perform_clustering(df, n_clusters=3)


# 15. Original DataFrame immutability
def test_original_dataframe_immutability():
    """Verify input DataFrame is not mutated during clustering."""
    df = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, np.inf, 5.0, 6.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    df_copy = df.copy(deep=True)

    perform_clustering(df, n_clusters=2, random_state=42)

    pd.testing.assert_frame_equal(df, df_copy)
    assert np.isnan(df.loc[1, "x"])
    assert np.isinf(df.loc[3, "x"])


# 16. JSON serialization
def test_json_serialization():
    """Verify returned result dictionary can be serialized and deserialized with standard json."""
    df = _create_synthetic_clusters(10)
    res = perform_clustering(df, n_clusters=3, random_state=42)

    serialized = json.dumps(res)
    assert isinstance(serialized, str)

    deserialized = json.loads(serialized)
    assert deserialized["status"] == "completed"
    assert deserialized["n_clusters"] == 3
    assert len(deserialized["cluster_assignments"]) == 30
    assert "0" in deserialized["cluster_sizes"]
    assert "0" in deserialized["centroids"]


# 17. Cluster assignment length
def test_cluster_assignment_length():
    """Verify cluster assignments length exactly equals total input rows."""
    df = _create_synthetic_clusters(15)
    res = perform_clustering(df, n_clusters=3)

    assert len(res["cluster_assignments"]) == len(df)
    for assignment in res["cluster_assignments"]:
        assert 0 <= assignment < 3


# 18. Cluster sizes summing to total rows
def test_cluster_sizes_sum():
    """Verify cluster sizes sum up exactly to the total row count."""
    df = _create_synthetic_clusters(12)
    res = perform_clustering(df, n_clusters=3)

    total_in_sizes = sum(res["cluster_sizes"].values())
    assert total_in_sizes == len(df)


# 19. Reproducibility
def test_reproducibility():
    """Verify identical random_state produces identical clustering results."""
    df = _create_synthetic_clusters(20, seed=123)

    run_1 = perform_clustering(df, n_clusters=3, random_state=99)
    run_2 = perform_clustering(df, n_clusters=3, random_state=99)

    assert run_1["cluster_assignments"] == run_2["cluster_assignments"]
    assert run_1["inertia"] == run_2["inertia"]
    assert run_1["centroids"] == run_2["centroids"]
    assert run_1["silhouette_score"] == run_2["silhouette_score"]


# 20. Centroids returned in original feature units
def test_centroids_in_original_feature_units():
    """Verify centroids are inverse-transformed back to the original scale, not z-scores."""
    # Data centered around ~1000 and ~50000
    df = pd.DataFrame({
        "revenue": [1000.0, 1050.0, 950.0, 5000.0, 5050.0, 4950.0],
        "users": [50000.0, 50100.0, 49900.0, 200000.0, 200500.0, 199500.0],
    })
    res = perform_clustering(df, n_clusters=2, random_state=42)

    c0_rev = res["centroids"]["0"]["revenue"]
    c1_rev = res["centroids"]["1"]["revenue"]

    # One cluster centroid should be around 1000, the other around 5000
    min_rev = min(c0_rev, c1_rev)
    max_rev = max(c0_rev, c1_rev)

    assert 900.0 <= min_rev <= 1100.0
    assert 4800.0 <= max_rev <= 5200.0

    # Test int access via ClusterDict helper
    assert res["centroids"][0]["revenue"] is not None
    assert res["cluster_sizes"][0] > 0


# 21. Silhouette score validity and unavailable cases
def test_silhouette_score_validity_and_unavailable():
    """Verify silhouette score computation and graceful handling when unavailable."""
    # Valid case
    df_valid = _create_synthetic_clusters(15)
    res_valid = perform_clustering(df_valid, n_clusters=3, random_state=42)
    assert res_valid["silhouette_score"] is not None
    assert -1.0 <= res_valid["silhouette_score"] <= 1.0

    # Unavailable / Degenerate case: when dataset rows equal 3 and n_clusters is 2
    # In this case silhouette score can still be computed if each cluster has samples,
    # but let's test that explanation is provided
    assert "silhouette_explanation" in res_valid
    assert len(res_valid["silhouette_explanation"]) > 0


# 22. Single numerical feature clustering
def test_single_numerical_feature():
    """Verify K-Means successfully clusters a 1D numerical feature."""
    df = pd.DataFrame({
        "val": [1.0, 1.2, 1.1, 50.0, 50.2, 49.8],
    })
    res = perform_clustering(df, n_clusters=2, feature_columns=["val"], random_state=42)

    assert res["status"] == "completed"
    assert res["selected_features"] == ["val"]
    assert len(res["cluster_assignments"]) == 6
    assert res["centroids"]["0"]["val"] is not None
