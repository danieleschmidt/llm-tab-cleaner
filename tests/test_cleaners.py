"""Tests for individual cleaners."""

import numpy as np
import pandas as pd
import pytest

from tab_cleaner.cleaners import (
    DuplicateRemover,
    MissingValueImputer,
    OutlierDetector,
    TypeCoercer,
)


def make_numeric_df_with_nans():
    return pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [10.0, 20.0, np.nan, 40.0]})


def test_missing_value_imputer_numeric():
    df = make_numeric_df_with_nans()
    imp = MissingValueImputer()
    cleaned, changes = imp.fit_transform(df)

    # No NaNs left
    assert cleaned.isna().sum().sum() == 0

    # Changes recorded for each NaN
    assert len(changes) == 2
    for ch in changes:
        assert ch["action"] == "impute_missing"
        assert ch["old_val"] is None

    # Imputed value for 'a' is mean of [1, 3, 4] = 8/3
    a_change = next(c for c in changes if c["col"] == "a")
    assert abs(a_change["new_val"] - (1 + 3 + 4) / 3) < 1e-6


def test_missing_value_imputer_string():
    df = pd.DataFrame({"name": ["alice", "bob", None, "alice"]})
    imp = MissingValueImputer()
    cleaned, changes = imp.fit_transform(df)

    assert cleaned["name"].isna().sum() == 0
    assert len(changes) == 1
    assert changes[0]["action"] == "impute_missing"
    # Mode of ["alice", "bob", "alice"] = "alice"
    assert changes[0]["new_val"] == "alice"


def test_type_coercer_converts_numeric():
    df = pd.DataFrame({"values": ["1", "2", "3", "4", "not_a_number"]})
    coercer = TypeCoercer(threshold=0.8)
    cleaned, changes = coercer.fit_transform(df)

    # Column should now be numeric (4 of 5 = 80% parseable)
    assert pd.api.types.is_numeric_dtype(cleaned["values"])
    # The non-numeric value becomes NaN
    assert cleaned["values"].isna().sum() == 1
    # Changes recorded (at least for the parseable rows that changed type)
    assert len(changes) >= 1


def test_type_coercer_skips_mostly_non_numeric():
    df = pd.DataFrame({"text": ["hello", "world", "foo", "bar", "1"]})
    coercer = TypeCoercer(threshold=0.8)
    cleaned, changes = coercer.fit_transform(df)

    # Only 20% parseable → column should NOT be coerced
    assert not pd.api.types.is_numeric_dtype(cleaned["text"])
    assert len(changes) == 0


def test_outlier_detector_iqr():
    # Q1=2, Q3=4, IQR=2, lower=-1, upper=7 → 100 is outlier
    data = [1, 2, 3, 4, 5, 100]
    df = pd.DataFrame({"x": data})
    detector = OutlierDetector(strategy="clip")
    cleaned, changes = detector.fit_transform(df)

    assert len(changes) == 1
    assert changes[0]["action"] == "clip_outlier"
    assert changes[0]["old_val"] == 100
    assert cleaned["x"].max() < 100


def test_outlier_detector_remove_strategy():
    data = [1, 2, 3, 4, 5, 100]
    df = pd.DataFrame({"x": data})
    detector = OutlierDetector(strategy="remove")
    cleaned, changes = detector.fit_transform(df)

    assert len(changes) == 1
    assert changes[0]["action"] == "remove_outlier_row"
    assert len(cleaned) == len(data) - 1


def test_duplicate_remover():
    df = pd.DataFrame({"a": [1, 2, 1, 3], "b": ["x", "y", "x", "z"]})
    remover = DuplicateRemover()
    cleaned, changes = remover.fit_transform(df)

    assert len(cleaned) == 3
    assert len(changes) == 1
    assert changes[0]["action"] == "remove_duplicate"


def test_duplicate_remover_no_dupes():
    df = pd.DataFrame({"a": [1, 2, 3]})
    remover = DuplicateRemover()
    cleaned, changes = remover.fit_transform(df)

    assert len(cleaned) == 3
    assert len(changes) == 0
