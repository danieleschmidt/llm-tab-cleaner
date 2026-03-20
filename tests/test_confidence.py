"""Tests for ConfidenceScorer."""

import numpy as np
import pandas as pd
import pytest

from tab_cleaner.confidence import ConfidenceScorer


def test_confidence_scorer_unchanged_cells():
    df_orig = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    df_clean = df_orig.copy()
    scorer = ConfidenceScorer()
    scores = scorer.score_dataframe(df_orig, df_clean, changes=[])

    # All unchanged → all scores should be 1.0
    assert scores.shape == df_clean.shape
    assert (scores == 1.0).all().all()


def test_confidence_scorer_changed_cells():
    df_orig = pd.DataFrame({"x": [1.0, 2.0, 100.0]})
    df_clean = pd.DataFrame({"x": [1.0, 2.0, 5.0]})  # 100 → 5 (clipped)
    changes = [{"row": 2, "col": "x", "old_val": 100.0, "new_val": 5.0, "action": "clip_outlier"}]

    scorer = ConfidenceScorer()
    scores = scorer.score_dataframe(df_orig, df_clean, changes)

    # Unchanged cells stay at 1.0
    assert scores.loc[0, "x"] == 1.0
    assert scores.loc[1, "x"] == 1.0
    # Changed cell is < 1.0
    assert scores.loc[2, "x"] < 1.0


def test_confidence_scorer_imputed_null():
    df_orig = pd.DataFrame({"y": [1.0, np.nan, 3.0]})
    df_clean = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    changes = [{"row": 1, "col": "y", "old_val": None, "new_val": 2.0, "action": "impute_missing"}]

    scorer = ConfidenceScorer()
    scores = scorer.score_dataframe(df_orig, df_clean, changes)

    # Imputed cell — old_val is None so no relative-change penalty, stays at 0.9
    assert 0.0 < scores.loc[1, "y"] <= 1.0


def test_confidence_scorer_removed_cell():
    df_orig = pd.DataFrame({"z": [1.0, 2.0, 3.0]})
    df_clean = pd.DataFrame({"z": [1.0, 2.0, 3.0]})
    # Simulate a removal change (new_val=None)
    changes = [{"row": 0, "col": "z", "old_val": 1.0, "new_val": None, "action": "remove_outlier_row"}]

    scorer = ConfidenceScorer()
    scores = scorer.score_dataframe(df_orig, df_clean, changes)

    assert scores.loc[0, "z"] == 0.0


def test_confidence_scores_bounded():
    """All scores must be in [0, 1]."""
    df_orig = pd.DataFrame({"a": [1.0, 2.0, 300.0, 4.0]})
    df_clean = pd.DataFrame({"a": [1.0, 2.0, 10.0, 4.0]})
    changes = [{"row": 2, "col": "a", "old_val": 300.0, "new_val": 10.0, "action": "clip_outlier"}]

    scorer = ConfidenceScorer()
    scores = scorer.score_dataframe(df_orig, df_clean, changes)

    assert (scores >= 0.0).all().all()
    assert (scores <= 1.0).all().all()
