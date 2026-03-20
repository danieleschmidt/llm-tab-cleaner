"""Tests for FastAPI endpoints."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tab_cleaner.api import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_clean_endpoint():
    csv_data = "age,salary\n25,50000\n,60000\n30,\n28,55000\n"
    response = client.post("/clean", json={"csv_data": csv_data})
    assert response.status_code == 200
    data = response.json()
    assert "cleaned_csv" in data
    assert "audit_json" in data
    assert data["rows_in"] == 4
    assert data["changes"] >= 2  # at least the two NaN imputes


def test_api_clean_invalid_csv():
    response = client.post("/clean", json={"csv_data": "not,a\nvalid\ncsv\n\n\n"})
    # May succeed or fail depending on pandas — just check it doesn't 500
    assert response.status_code in (200, 400)


def test_api_clean_audit_is_valid_json():
    csv_data = "x,y\n1,2\n3,4\n"
    response = client.post("/clean", json={"csv_data": csv_data})
    assert response.status_code == 200
    data = response.json()
    # Audit should be parseable JSON
    audit_entries = json.loads(data["audit_json"])
    assert isinstance(audit_entries, list)
