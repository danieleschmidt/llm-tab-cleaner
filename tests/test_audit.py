"""Tests for AuditTrail."""

import json

import pytest

from tab_cleaner.audit import AuditTrail


def test_audit_trail_records_changes():
    trail = AuditTrail()
    trail.record(
        row_idx=0,
        col="age",
        old_value=None,
        new_value=30.0,
        action="impute_missing",
        confidence=0.9,
    )
    assert len(trail) == 1
    entry = trail.entries[0]
    assert entry["row_idx"] == 0
    assert entry["col"] == "age"
    assert entry["old_value"] is None
    assert entry["new_value"] == 30.0
    assert entry["action"] == "impute_missing"
    assert entry["confidence"] == 0.9
    assert "timestamp" in entry


def test_audit_trail_json_roundtrip():
    trail = AuditTrail()
    trail.record(1, "salary", 0, 50000, "impute_missing", confidence=0.85)
    trail.record(2, "age", 200, 75.0, "clip_outlier", confidence=0.7)

    json_str = trail.to_json()
    # Valid JSON
    parsed = json.loads(json_str)
    assert len(parsed) == 2

    # Round-trip via from_json
    restored = AuditTrail.from_json(json_str)
    assert len(restored) == 2
    assert restored.entries[0]["col"] == "salary"
    assert restored.entries[1]["action"] == "clip_outlier"


def test_audit_summary():
    trail = AuditTrail()
    trail.record(0, "a", None, 1, "impute_missing")
    trail.record(1, "a", None, 2, "impute_missing")
    trail.record(2, "b", 999, 75, "clip_outlier")

    summary = trail.summary()
    assert summary["impute_missing"] == 2
    assert summary["clip_outlier"] == 1


def test_audit_empty():
    trail = AuditTrail()
    assert len(trail) == 0
    assert trail.summary() == {}
    assert trail.to_json() == "[]"


def test_audit_record_change_helper():
    trail = AuditTrail()
    change = {"row": 5, "col": "x", "old_val": 10, "new_val": 8, "action": "clip_outlier"}
    trail.record_change(change, confidence=0.8)
    assert len(trail) == 1
    assert trail.entries[0]["confidence"] == 0.8
