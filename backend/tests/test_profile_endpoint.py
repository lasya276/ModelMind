"""
API tests for the Dataset Profiling endpoint (backend/app/routes/profile.py).
"""

import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.data_engine.storage import get_dataset

client = TestClient(app)


# 1. Successful profiling
def test_profile_successful():
    """Verify successful profiling of a valid CSV file returning complete report."""
    csv_content = (
        b"name,age,salary,department\n"
        b"Alice,30,70000,Engineering\n"
        b"Bob,25,50000,Marketing\n"
        b"Charlie,35,90000,Engineering\n"
        b"Diana,28,60000,Sales\n"
    )
    response = client.post(
        "/profile",
        files={"file": ("employees.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["filename"] == "employees.csv"
    assert "dataset_id" in res_data
    assert len(res_data["dataset_id"]) > 0

    profile = res_data["data"]
    assert profile["total_rows"] == 4
    assert profile["total_columns"] == 4
    assert "columns" in profile
    assert "numeric_summary" in profile
    assert "categorical_summary" in profile
    assert "missing_values" in profile
    assert "unique_values" in profile
    assert "duplicate_rows" in profile

    # Verify numeric summary contains stats for age and salary
    assert "age" in profile["numeric_summary"]
    assert "salary" in profile["numeric_summary"]
    assert profile["numeric_summary"]["age"]["mean"] is not None

    # Verify categorical summary contains department
    assert "department" in profile["categorical_summary"]
    assert profile["categorical_summary"]["department"]["top_value"] == "Engineering"


# 2. Correct row and column counts
def test_profile_correct_counts():
    """Verify that returned row and column counts match the dataset structure."""
    csv_content = b"col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n"
    response = client.post(
        "/profile",
        files={"file": ("matrix.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_rows"] == 3
    assert data["total_columns"] == 3
    assert data["columns"] == ["col1", "col2", "col3"]


# 3. Empty CSV
def test_profile_empty_csv():
    """Verify 400 Bad Request when an empty CSV is uploaded."""
    response = client.post(
        "/profile",
        files={"file": ("empty.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


# 4. Invalid file format
def test_profile_invalid_file_format():
    """Verify 400 Bad Request when non-CSV file is provided."""
    response = client.post(
        "/profile",
        files={"file": ("document.pdf", b"%PDF-1.4 dummy content", "application/pdf")},
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


# 5. Corrupted CSV
def test_profile_corrupted_csv():
    """Verify 400 Bad Request when CSV file has invalid encoding or corrupted content."""
    # Invalid non-UTF-8 bytes
    corrupted_bytes = b"\xff\xfe\x00\x00corrupted\x80\x81\x82data"
    response = client.post(
        "/profile",
        files={"file": ("corrupt.csv", corrupted_bytes, "text/csv")},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


# 6. Missing file
def test_profile_missing_file():
    """Verify 400 Bad Request when no file or dataset_id is sent."""
    response = client.post("/profile")

    assert response.status_code == 400
    assert "No file or dataset_id provided" in response.json()["detail"]


# 7. Error handling: empty or missing filename
def test_profile_empty_filename():
    """Verify 400 or 422 error when an uploaded file has an empty filename."""
    response = client.post(
        "/profile",
        files={"file": ("", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code in (400, 422)


# 8. JSON serialization
def test_profile_json_serialization():
    """Verify the entire endpoint response is strictly JSON-serializable."""
    csv_content = b"val1,val2,label\n1.5,NaN,ClassA\nNaN,3.2,ClassB\n"
    response = client.post(
        "/profile",
        files={"file": ("nan_data.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()

    # Verify standard json dumps and loads work without TypeError
    dumped = json.dumps(body)
    assert isinstance(dumped, str)
    reloaded = json.loads(dumped)
    assert reloaded["status"] == "success"
    assert reloaded["data"]["total_rows"] == 2


# 9. Profile via dataset_id
def test_profile_via_dataset_id():
    """Verify profiling a dataset using an existing dataset_id from /upload."""
    csv_content = b"x,y\n10,20\n30,40\n50,60\n"

    # Step 1: Upload via /upload
    upload_res = client.post(
        "/upload",
        files={"file": ("points.csv", csv_content, "text/csv")},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["data"]["dataset_id"]
    assert dataset_id is not None

    # Step 2: Profile via dataset_id
    profile_res = client.post(f"/profile?dataset_id={dataset_id}")
    assert profile_res.status_code == 200
    data = profile_res.json()
    assert data["status"] == "success"
    assert data["dataset_id"] == dataset_id
    assert data["data"]["total_rows"] == 3
    assert data["data"]["columns"] == ["x", "y"]


# 10. Profile via nonexistent dataset_id
def test_profile_nonexistent_dataset_id():
    """Verify 404 Not Found when a valid-format UUID does not exist in storage."""
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    response = client.post(f"/profile?dataset_id={fake_uuid}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# 11. Profile via invalid dataset_id format
def test_profile_invalid_dataset_id_format():
    """Verify 400 Bad Request when an invalid dataset_id format is passed."""
    response = client.post("/profile?dataset_id=../../traversal")

    assert response.status_code == 400
    assert "invalid dataset_id format" in response.json()["detail"].lower()


# 12. Preserves original uploaded data
def test_profile_preserves_original_data():
    """Verify the stored CSV bytes match the exact uploaded bytes."""
    raw_content = b"id,score\n1,95.5\n2,88.0\n"
    response = client.post(
        "/profile",
        files={"file": ("scores.csv", raw_content, "text/csv")},
    )

    assert response.status_code == 200
    dataset_id = response.json()["dataset_id"]

    stored_bytes, stored_filename = get_dataset(dataset_id)
    assert stored_bytes == raw_content
    assert stored_filename == "scores.csv"


# 13. Oversized file upload rejected with HTTP 413 on /profile
def test_profile_oversized_file_rejected(monkeypatch):
    """Verify HTTP 413 is returned when a file exceeds the maximum upload size on /profile."""
    # Temporarily set max size to 1 KB to test without allocating 50 MB in memory
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)

    large_content = b"col1,col2\n" + (b"12345,67890\n" * 150)  # ~1.8 KB > 1 KB
    response = client.post(
        "/profile",
        files={"file": ("large.csv", large_content, "text/csv")},
    )

    assert response.status_code == 413
    assert "exceeds the maximum" in response.json()["detail"].lower()


# 14. Oversized file upload rejected with HTTP 413 on /upload
def test_upload_oversized_file_rejected(monkeypatch):
    """Verify HTTP 413 is returned when a file exceeds the maximum upload size on /upload."""
    monkeypatch.setattr("app.data_engine.storage.MAX_UPLOAD_SIZE_BYTES", 1024)

    large_content = b"col1,col2\n" + (b"12345,67890\n" * 150)
    response = client.post(
        "/upload",
        files={"file": ("large.csv", large_content, "text/csv")},
    )

    assert response.status_code == 413
    assert "exceeds the maximum" in response.json()["detail"].lower()


# 15. Automatic 48-hour dataset expiration cleanup
def test_dataset_expiration_cleanup():
    """Verify datasets older than 48 hours are automatically purged while fresh ones remain."""
    import time
    from app.data_engine.storage import (
        save_dataset,
        cleanup_expired_datasets,
        get_storage_dir,
        UUID_PATTERN,
    )

    # 1. Create a fresh dataset
    fresh_id = save_dataset(b"a,b\n1,2\n", "fresh.csv")
    s_dir = get_storage_dir()

    # 2. Create an expired dataset (simulating 50 hours ago)
    expired_id = save_dataset(b"x,y\n3,4\n", "expired.csv")
    meta_path = s_dir / f"{expired_id}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["created_at"] = time.time() - (50 * 3600)  # 50 hours ago
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    # Run cleanup with 48-hour threshold
    cleaned_count = cleanup_expired_datasets(max_age_hours=48)
    assert cleaned_count >= 1

    # Verify expired dataset files are removed
    assert not (s_dir / f"{expired_id}.csv").exists()
    assert not (s_dir / f"{expired_id}.json").exists()

    # Verify fresh dataset files remain intact
    assert (s_dir / f"{fresh_id}.csv").exists()
    assert (s_dir / f"{fresh_id}.json").exists()


# 16. Safe atomic file writes
def test_safe_atomic_writes():
    """Verify safe atomic writes leave no temporary .tmp files and record created_at metadata."""
    from app.data_engine.storage import save_dataset, get_storage_dir, get_dataset_metadata

    content = b"header1,header2\nalpha,beta\n"
    dataset_id = save_dataset(content, "atomic_test.csv")

    s_dir = get_storage_dir()

    # Verify destination files exist
    assert (s_dir / f"{dataset_id}.csv").exists()
    assert (s_dir / f"{dataset_id}.json").exists()

    # Verify no .tmp files were left behind
    assert not (s_dir / f"{dataset_id}.csv.tmp").exists()
    assert not (s_dir / f"{dataset_id}.json.tmp").exists()

    # Verify metadata contains size_bytes and created_at timestamp
    meta = get_dataset_metadata(dataset_id)
    assert meta["dataset_id"] == dataset_id
    assert meta["filename"] == "atomic_test.csv"
    assert meta["size_bytes"] == len(content)
    assert "created_at" in meta
    assert isinstance(meta["created_at"], (int, float))

