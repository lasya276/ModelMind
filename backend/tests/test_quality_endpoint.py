"""
API tests for the Data Quality endpoint (backend/app/routes/quality.py).
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data_engine.storage import get_dataset

client = TestClient(app)


# 1. Successful direct CSV upload
def test_quality_successful_direct_upload():
    """Verify data quality check on direct CSV upload with intentional quality issues."""
    csv_content = (
        b"id,name,category,constant_col\n"
        b"1,Alice,electronics,fixed\n"
        b"2,Bob,Electronics ,fixed\n"
        b"3,,clothing,fixed\n"
        b"1,Alice,electronics,fixed\n"  # Duplicate row
    )
    response = client.post(
        "/quality",
        files={"file": ("dataset.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert res["filename"] == "dataset.csv"
    assert "dataset_id" in res
    assert len(res["dataset_id"]) > 0

    data = res["data"]
    assert data["total_rows"] == 4
    assert data["total_columns"] == 4
    assert data["has_issues"] is True
    assert data["issue_counts"]["duplicate_rows"] == 1
    assert data["issue_counts"]["constant_columns"] >= 1
    assert data["issue_counts"]["missing_value_columns"] >= 1
    assert "missing_values" in data
    assert "duplicate_rows" in data
    assert "constant_columns" in data
    assert "potential_id_columns" in data
    assert "inconsistent_categorical_values" in data
    assert "invalid_numerical_values" in data


# 2. Successful quality check via dataset_id (query param)
def test_quality_via_dataset_id_query():
    """Verify data quality check using an existing dataset_id query parameter."""
    csv_content = b"a,b\n1,10\n2,20\n3,30\n"

    # Step 1: Upload via /upload
    upload_res = client.post(
        "/upload",
        files={"file": ("clean_data.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["data"]["dataset_id"]

    # Step 2: Quality check via query param
    quality_res = client.post(f"/quality?dataset_id={dataset_id}")
    assert quality_res.status_code == 200
    res = quality_res.json()
    assert res["status"] == "success"
    assert res["dataset_id"] == dataset_id
    assert res["data"]["total_rows"] == 3
    assert res["data"]["total_columns"] == 2
    assert res["data"]["has_issues"] is False


# 3. Successful quality check via JSON body
def test_quality_via_dataset_id_json_body():
    """Verify data quality check using dataset_id provided in JSON body."""
    csv_content = b"val1,val2\n100,200\n300,400\n"

    upload_res = client.post(
        "/upload",
        files={"file": ("json_test.csv", csv_content, "text/csv")},
    )
    dataset_id = upload_res.json()["data"]["dataset_id"]

    quality_res = client.post("/quality", json={"dataset_id": dataset_id})
    assert quality_res.status_code == 200
    assert quality_res.json()["data"]["total_rows"] == 2


# 4. Empty CSV
def test_quality_empty_csv():
    """Verify 400 Bad Request when an empty CSV is uploaded."""
    response = client.post(
        "/quality",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# 5. Invalid file format
def test_quality_invalid_file_format():
    """Verify 400 Bad Request when non-CSV file is uploaded."""
    response = client.post(
        "/quality",
        files={"file": ("notes.txt", b"just text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


# 6. Corrupted CSV
def test_quality_corrupted_csv():
    """Verify 400 Bad Request when corrupted non-UTF-8 bytes are sent."""
    corrupted_bytes = b"\xff\xfe\x00\x00bad\x80\x81\x82bytes"
    response = client.post(
        "/quality",
        files={"file": ("corrupt.csv", corrupted_bytes, "text/csv")},
    )
    assert response.status_code == 400


# 7. Missing file and dataset_id
def test_quality_missing_inputs():
    """Verify 400 Bad Request when neither file nor dataset_id is provided."""
    response = client.post("/quality")
    assert response.status_code == 400
    assert "No file or dataset_id provided" in response.json()["detail"]


# 8. Nonexistent dataset_id
def test_quality_nonexistent_dataset_id():
    """Verify 404 Not Found for a valid UUID that does not exist in storage."""
    fake_uuid = "11111111-2222-3333-4444-555555555555"
    response = client.post(f"/quality?dataset_id={fake_uuid}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# 9. Invalid dataset_id format
def test_quality_invalid_dataset_id_format():
    """Verify 400 Bad Request when dataset_id format is invalid."""
    response = client.post("/quality?dataset_id=../../traversal")
    assert response.status_code == 400
    assert "invalid dataset_id format" in response.json()["detail"].lower()


# 10. Oversized file rejected
def test_quality_oversized_file_rejected(monkeypatch):
    """Verify 413 Content Too Large when uploaded file exceeds size limit."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)
    large_content = b"a,b\n" + (b"12345,67890\n" * 150)
    response = client.post(
        "/quality",
        files={"file": ("large.csv", large_content, "text/csv")},
    )
    assert response.status_code == 413


# 11. JSON serialization
def test_quality_json_serialization():
    """Verify the entire endpoint response is strictly JSON-serializable."""
    csv_content = b"score,ratio\n1.5,NaN\nNaN,inf\n"
    response = client.post(
        "/quality",
        files={"file": ("infs.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    dumped = json.dumps(response.json())
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"
