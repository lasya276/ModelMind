"""
API tests for the Anomaly Detection endpoint (backend/app/routes/anomaly.py).
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data_engine.storage import get_dataset

client = TestClient(app)


def _make_anomaly_csv_bytes(n_points: int = 30) -> bytes:
    """Helper to generate CSV bytes with 30 normal points and 2 extreme outliers."""
    lines = ["x,y,label"]
    for i in range(n_points):
        lines.append(f"{10.0 + (i % 3)},{20.0 + (i % 4)},normal")
    lines.append("1000.0,1000.0,outlier_1")
    lines.append("-500.0,-500.0,outlier_2")
    return "\n".join(lines).encode("utf-8")


# 1. Successful direct CSV upload with anomaly detection
def test_anomaly_successful_direct_upload():
    """Verify anomaly detection on directly uploaded CSV with extreme outliers."""
    csv_content = _make_anomaly_csv_bytes(30)
    response = client.post(
        "/anomaly",
        files={"file": ("points.csv", csv_content, "text/csv")},
        data={"contamination": "0.1"},
    )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["filename"] == "points.csv"
    assert "dataset_id" in res

    data = res["data"]
    assert data["status"] == "completed"
    assert data["total_rows_analyzed"] == 32
    assert data["anomaly_count"] >= 1
    assert data["normal_count"] + data["anomaly_count"] == 32
    assert len(data["anomaly_labels"]) == 32
    assert len(data["anomaly_scores"]) == 32
    assert len(data["is_anomaly"]) == 32
    assert data["visualization_data"]["returned_rows"] == 32
    assert data["flagged_observations"]["total_count"] == data["anomaly_count"]
    assert all(row["is_anomaly"] for row in data["flagged_observations"]["rows"])
    assert "x" in data["features_used"]
    assert "y" in data["features_used"]
    assert "label" not in data["features_used"]  # Non-numeric text excluded

    # Verify ethical wording: anomalies are unusual observations, not assumed fraud/errors
    assert "not necessarily indicate an error, defect, or fraud" in data["explanation"]


# 2. Parameter support: feature_columns (comma-separated and JSON) and contamination
def test_anomaly_parameters_support():
    """Verify specifying feature_columns and contamination via form parameters."""
    csv_content = _make_anomaly_csv_bytes(30)

    # Test comma-separated feature_columns
    response = client.post(
        "/anomaly",
        files={"file": ("test.csv", csv_content, "text/csv")},
        data={"feature_columns": "x,y", "contamination": "0.05"},
    )
    assert response.status_code == 200
    assert response.json()["data"]["features_used"] == ["x", "y"]

    # Test JSON-serialized list for feature_columns
    response2 = client.post(
        "/anomaly",
        files={"file": ("test.csv", csv_content, "text/csv")},
        data={"feature_columns": json.dumps(["x"]), "contamination": "auto"},
    )
    assert response2.status_code == 200
    assert response2.json()["data"]["features_used"] == ["x"]


# 3. Successful anomaly detection via dataset_id (query param)
def test_anomaly_via_dataset_id_query():
    """Verify anomaly detection using existing dataset_id query parameter."""
    csv_content = _make_anomaly_csv_bytes(25)

    upload_res = client.post(
        "/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(f"/anomaly?dataset_id={dataset_id}&contamination=0.1")
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert res.json()["dataset_id"] == dataset_id
    assert res.json()["data"]["total_rows_analyzed"] == 27


# 4. Successful anomaly detection via JSON body
def test_anomaly_via_dataset_id_json():
    """Verify anomaly detection using dataset_id and parameters in JSON body."""
    csv_content = _make_anomaly_csv_bytes(20)

    upload_res = client.post(
        "/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(
        "/anomaly",
        json={
            "dataset_id": dataset_id,
            "feature_columns": ["x", "y"],
            "contamination": 0.1,
        },
    )
    assert res.status_code == 200
    assert res.json()["data"]["features_used"] == ["x", "y"]


# 5. Reject invalid contamination parameter
def test_anomaly_reject_invalid_contamination():
    """Verify 400 Bad Request when contamination is out of valid range (0, 0.5] or invalid string."""
    csv_content = _make_anomaly_csv_bytes(20)

    # Value too high (> 0.5)
    res_high = client.post(
        "/anomaly",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"contamination": "0.9"},
    )
    assert res_high.status_code == 400
    assert "contamination" in res_high.json()["detail"].lower()

    # Negative value
    res_neg = client.post(
        "/anomaly",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"contamination": "-0.1"},
    )
    assert res_neg.status_code == 400

    # Invalid string
    res_str = client.post(
        "/anomaly",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"contamination": "invalid_string"},
    )
    assert res_str.status_code == 400


# 6. Reject nonexistent feature columns
def test_anomaly_reject_nonexistent_features():
    """Verify 400 Bad Request when specified feature column does not exist."""
    csv_content = _make_anomaly_csv_bytes(20)
    res = client.post(
        "/anomaly",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"feature_columns": "x,nonexistent_col"},
    )
    assert res.status_code == 400
    assert "do not exist" in res.json()["detail"].lower()


# 7. Reject non-numeric feature columns
def test_anomaly_reject_non_numeric_features():
    """Verify 400 Bad Request when specified feature column is non-numeric."""
    csv_content = _make_anomaly_csv_bytes(20)
    res = client.post(
        "/anomaly",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"feature_columns": "label"},
    )
    assert res.status_code == 400
    assert "not numeric" in res.json()["detail"].lower()


# 8. Empty CSV
def test_anomaly_empty_csv():
    """Verify 400 Bad Request when an empty CSV is uploaded."""
    res = client.post(
        "/anomaly",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


# 9. Invalid file format
def test_anomaly_invalid_file_format():
    """Verify 400 Bad Request when non-CSV file is uploaded."""
    res = client.post(
        "/anomaly",
        files={"file": ("script.sh", b"echo hello", "text/x-sh")},
    )
    assert res.status_code == 400
    assert "Invalid file type" in res.json()["detail"]


# 10. Corrupted CSV
def test_anomaly_corrupted_csv():
    """Verify 400 Bad Request when corrupted non-UTF-8 bytes are sent."""
    corrupted_bytes = b"\xff\xfe\x00\x00bad\x80\x81\x82bytes"
    res = client.post(
        "/anomaly",
        files={"file": ("corrupt.csv", corrupted_bytes, "text/csv")},
    )
    assert res.status_code == 400


# 11. Missing file and dataset_id
def test_anomaly_missing_inputs():
    """Verify 400 Bad Request when neither file nor dataset_id is provided."""
    res = client.post("/anomaly")
    assert res.status_code == 400
    assert "No file or dataset_id provided" in res.json()["detail"]


# 12. Nonexistent dataset_id
def test_anomaly_nonexistent_dataset_id():
    """Verify 404 Not Found for a valid UUID that does not exist in storage."""
    fake_uuid = "55555555-6666-7777-8888-999999999999"
    res = client.post(f"/anomaly?dataset_id={fake_uuid}")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# 13. Invalid dataset_id format
def test_anomaly_invalid_dataset_id_format():
    """Verify 400 Bad Request when dataset_id format is invalid."""
    res = client.post("/anomaly?dataset_id=../../traversal")
    assert res.status_code == 400
    assert "invalid dataset_id format" in res.json()["detail"].lower()


# 14. Oversized file rejected
def test_anomaly_oversized_file_rejected(monkeypatch):
    """Verify 413 Content Too Large when uploaded file exceeds size limit."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)
    large_content = b"x,y\n" + (b"12345,67890\n" * 150)
    res = client.post(
        "/anomaly",
        files={"file": ("large.csv", large_content, "text/csv")},
    )
    assert res.status_code == 413


# 15. JSON serialization
def test_anomaly_json_serialization():
    """Verify anomaly detection response is strictly JSON-serializable."""
    csv_content = _make_anomaly_csv_bytes(15)
    res = client.post(
        "/anomaly",
        files={"file": ("data.csv", csv_content, "text/csv")},
    )
    assert res.status_code == 200
    dumped = json.dumps(res.json())
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"


# 16. Original dataset remains unchanged
def test_anomaly_original_dataset_remains_unchanged():
    """Verify original dataset stored bytes are strictly preserved after anomaly detection."""
    original_csv = _make_anomaly_csv_bytes(20)

    upload_res = client.post(
        "/upload",
        files={"file": ("raw_points.csv", original_csv, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    res = client.post(f"/anomaly?dataset_id={dataset_id}")
    assert res.status_code == 200

    stored_bytes, stored_filename = get_dataset(dataset_id)
    assert stored_bytes == original_csv
    assert stored_filename == "raw_points.csv"
