"""
Basic tests for DataSift API endpoints.
Can be executed with pytest once dependencies are installed.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify that root endpoint returns status online."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_endpoint():
    """Verify health check endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_upload_invalid_extension():
    """Verify rejecting non-CSV files."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"some text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_upload_valid_csv():
    """Verify parsing a valid CSV file."""
    csv_content = b"name,age,city\nAlice,30,New York\nBob,25,San Francisco\n"
    response = client.post(
        "/upload",
        files={"file": ("users.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["row_count"] == 2
    assert data["data"]["column_count"] == 3
    assert data["data"]["columns"] == ["name", "age", "city"]


def test_upload_empty_csv():
    """Verify handling empty CSV files gracefully."""
    response = client.post(
        "/upload",
        files={"file": ("empty.csv", b"", "text/csv")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
