"""
API tests for the Clustering endpoint (backend/app/routes/clustering.py).
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data_engine.storage import get_dataset

client = TestClient(app)


def _make_clustering_csv_bytes(n_per_group: int = 15) -> bytes:
    """Helper to generate CSV bytes with 3 natural clusters in 2D."""
    lines = ["x,y,label"]
    for i in range(n_per_group):
        lines.append(f"{1.0 + i * 0.1},{1.0 + i * 0.1},group_a")
    for i in range(n_per_group):
        lines.append(f"{50.0 + i * 0.1},{50.0 + i * 0.1},group_b")
    for i in range(n_per_group):
        lines.append(f"{100.0 + i * 0.1},{100.0 + i * 0.1},group_c")
    return "\n".join(lines).encode("utf-8")


# 1. Successful direct CSV upload with clustering
def test_cluster_successful_direct_upload():
    """Verify clustering on directly uploaded CSV with well-separated groups."""
    csv_content = _make_clustering_csv_bytes(15)
    response = client.post(
        "/cluster",
        files={"file": ("points.csv", csv_content, "text/csv")},
        data={"n_clusters": "3"},
    )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["filename"] == "points.csv"
    assert "dataset_id" in res

    data = res["data"]
    assert data["status"] == "completed"
    assert data["n_clusters"] == 3
    assert data["total_rows"] == 45
    assert len(data["cluster_assignments"]) == 45
    preview = data["visualization_data"]
    assert preview["total_rows"] == 45
    assert preview["returned_rows"] == 45
    assert len(preview["rows"]) == 45
    for row in preview["rows"]:
        assert row["cluster"] == data["cluster_assignments"][row["row_index"]]
    assert len(data["cluster_sizes"]) == 3
    assert len(data["centroids"]) == 3
    assert data["inertia"] is not None
    assert data["silhouette_score"] is not None

    # Verify features used (label is non-numeric, should be excluded)
    assert "x" in data["features_used"]
    assert "y" in data["features_used"]
    assert "label" not in data["features_used"]

    # Verify cluster labels are not claimed to represent real-world categories
    assert "not claimed" in data["explanation"].lower() or "does not claim" in data["explanation"].lower() or "not claim" in data["explanation"].lower()


# 2. Default n_clusters (3) when not specified
def test_cluster_default_n_clusters():
    """Verify default n_clusters=3 when not provided."""
    csv_content = _make_clustering_csv_bytes(15)
    response = client.post(
        "/cluster",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["data"]["n_clusters"] == 3


# 3. Feature columns specification
def test_cluster_with_feature_columns():
    """Verify clustering with explicitly specified feature columns."""
    csv_content = _make_clustering_csv_bytes(15)
    response = client.post(
        "/cluster",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"feature_columns": "x,y", "n_clusters": "2"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["features_used"] == ["x", "y"]
    assert data["n_clusters"] == 2


# 4. JSON feature columns format
def test_cluster_json_feature_columns():
    """Verify feature_columns specified as JSON array."""
    csv_content = _make_clustering_csv_bytes(15)
    response = client.post(
        "/cluster",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"feature_columns": json.dumps(["x"]), "n_clusters": "2"},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["features_used"] == ["x"]


# 5. Via dataset_id query parameter
def test_cluster_via_dataset_id_query():
    """Verify clustering using existing dataset_id query parameter."""
    csv_content = _make_clustering_csv_bytes(15)

    upload_res = client.post(
        "/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(f"/cluster?dataset_id={dataset_id}&n_clusters=3")
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert res.json()["dataset_id"] == dataset_id


# 6. Via JSON body
def test_cluster_via_json_body():
    """Verify clustering using dataset_id and parameters in JSON body."""
    csv_content = _make_clustering_csv_bytes(15)

    upload_res = client.post(
        "/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(
        "/cluster",
        json={
            "dataset_id": dataset_id,
            "feature_columns": ["x", "y"],
            "n_clusters": 2,
        },
    )
    assert res.status_code == 200
    assert res.json()["data"]["n_clusters"] == 2
    assert res.json()["data"]["features_used"] == ["x", "y"]


# 7. Invalid n_clusters (less than 2)
def test_cluster_invalid_n_clusters_too_low():
    """Verify 400 Bad Request when n_clusters is less than 2."""
    csv_content = _make_clustering_csv_bytes(15)
    res = client.post(
        "/cluster",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"n_clusters": "1"},
    )
    assert res.status_code == 400
    assert "at least 2" in res.json()["detail"].lower()


# 8. Invalid n_clusters (non-integer string)
def test_cluster_invalid_n_clusters_non_integer():
    """Verify 400 Bad Request when n_clusters is not a valid integer."""
    csv_content = _make_clustering_csv_bytes(15)
    res = client.post(
        "/cluster",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"n_clusters": "abc"},
    )
    assert res.status_code == 400
    assert "n_clusters" in res.json()["detail"].lower()


# 9. n_clusters >= total rows
def test_cluster_n_clusters_too_high():
    """Verify 400 Bad Request when n_clusters >= sample count."""
    lines = ["x,y"]
    for i in range(5):
        lines.append(f"{float(i)},{float(i * 2)}")
    csv_content = "\n".join(lines).encode("utf-8")
    res = client.post(
        "/cluster",
        files={"file": ("tiny.csv", csv_content, "text/csv")},
        data={"n_clusters": "5"},
    )
    assert res.status_code == 400
    assert "less than" in res.json()["detail"].lower()


# 10. Nonexistent feature columns
def test_cluster_nonexistent_feature_columns():
    """Verify 400 Bad Request when specified feature column does not exist."""
    csv_content = _make_clustering_csv_bytes(15)
    res = client.post(
        "/cluster",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"feature_columns": "x,nonexistent_col"},
    )
    assert res.status_code == 400
    assert "do not exist" in res.json()["detail"].lower()


# 11. Non-numeric feature columns
def test_cluster_non_numeric_feature_columns():
    """Verify 400 Bad Request when specified feature column is non-numeric."""
    csv_content = _make_clustering_csv_bytes(15)
    res = client.post(
        "/cluster",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"feature_columns": "label"},
    )
    assert res.status_code == 400
    assert "not numerical" in res.json()["detail"].lower()


# 12. Empty CSV
def test_cluster_empty_csv():
    """Verify 400 Bad Request when an empty CSV is uploaded."""
    res = client.post(
        "/cluster",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert res.status_code == 400


# 13. Invalid file format
def test_cluster_invalid_file_format():
    """Verify 400 Bad Request when non-CSV file is uploaded."""
    res = client.post(
        "/cluster",
        files={"file": ("image.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert res.status_code == 400
    assert "Invalid file type" in res.json()["detail"]


# 14. Missing file and dataset_id
def test_cluster_missing_inputs():
    """Verify 400 Bad Request when neither file nor dataset_id is provided."""
    res = client.post("/cluster")
    assert res.status_code == 400
    assert "No file or dataset_id provided" in res.json()["detail"]


# 15. Nonexistent dataset_id
def test_cluster_nonexistent_dataset_id():
    """Verify 404 Not Found for a valid UUID that does not exist in storage."""
    fake_uuid = "12345678-abcd-1234-abcd-1234567890ab"
    res = client.post(f"/cluster?dataset_id={fake_uuid}")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# 16. Oversized file rejected
def test_cluster_oversized_file_rejected(monkeypatch):
    """Verify 413 Content Too Large when uploaded file exceeds size limit."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)
    large_content = b"x,y\n" + (b"12345.6,67890.1\n" * 150)
    res = client.post(
        "/cluster",
        files={"file": ("large.csv", large_content, "text/csv")},
    )
    assert res.status_code == 413


# 17. JSON serialization
def test_cluster_json_serialization():
    """Verify clustering response is strictly JSON-serializable."""
    csv_content = _make_clustering_csv_bytes(15)
    res = client.post(
        "/cluster",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"n_clusters": "3"},
    )
    assert res.status_code == 200
    dumped = json.dumps(res.json())
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"


# 18. Original dataset remains unchanged
def test_cluster_original_dataset_remains_unchanged():
    """Verify original dataset stored bytes are strictly preserved after clustering."""
    original_csv = _make_clustering_csv_bytes(15)

    upload_res = client.post(
        "/upload",
        files={"file": ("raw_points.csv", original_csv, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(f"/cluster?dataset_id={dataset_id}&n_clusters=3")
    assert res.status_code == 200

    stored_bytes, stored_filename = get_dataset(dataset_id)
    assert stored_bytes == original_csv
    assert stored_filename == "raw_points.csv"
