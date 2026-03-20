"""ConfidenceScorer: assigns a 0-1 confidence score to every cell after cleaning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ConfidenceScorer:
    """Score each cell in a cleaned dataframe relative to the original.

    Score semantics:
        1.0  - cell was not changed at all
        0.9  - cell was changed but the new value is within the column's normal range
        0.5  - cell was changed and the relative change was large
        0.0  - cell was removed / set to None

    The score is blended from three components:
        1. Was the cell changed?         (1.0 if unchanged, else partial)
        2. Relative change magnitude     (for numeric cells)
        3. Is the new value in-range?    (within [mean ± 2*std])
    """

    def score_dataframe(
        self,
        df_original: pd.DataFrame,
        df_cleaned: pd.DataFrame,
        changes: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Return a dataframe of the same shape as *df_cleaned* with float confidence scores."""

        # Build a lookup of (row, col) -> change
        change_map: dict[tuple[int, str], dict[str, Any]] = {}
        for ch in changes:
            key = (ch["row"], ch["col"])
            change_map[key] = ch

        # Pre-compute per-column stats on the cleaned data for range checks
        col_stats: dict[str, tuple[float, float]] = {}
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            mu = df_cleaned[col].mean()
            sigma = df_cleaned[col].std()
            col_stats[col] = (mu, float(sigma) if not np.isnan(sigma) else 0.0)

        scores = pd.DataFrame(
            np.ones((len(df_cleaned), len(df_cleaned.columns))),
            index=df_cleaned.index,
            columns=df_cleaned.columns,
            dtype=float,
        )

        for (row, col), ch in change_map.items():
            if row not in scores.index or col not in scores.columns:
                continue

            old_val = ch["old_val"]
            new_val = ch["new_val"]

            # Removed cell / row
            if new_val is None:
                scores.loc[row, col] = 0.0
                continue

            # Start with base confidence for a changed cell
            score = 0.9

            # For numeric columns compute relative change penalty
            if col in col_stats and old_val is not None:
                try:
                    old_f = float(old_val)
                    new_f = float(new_val)
                    denominator = abs(old_f) if abs(old_f) > 1e-9 else 1.0
                    rel_change = abs(new_f - old_f) / denominator
                    # Penalty up to -0.4 for large relative changes
                    penalty = min(0.4, rel_change * 0.4)
                    score -= penalty

                    # In-range bonus: is new value within mean ± 2*std?
                    mu, sigma = col_stats[col]
                    if sigma > 0 and abs(new_f - mu) <= 2 * sigma:
                        score = min(1.0, score + 0.05)
                except (TypeError, ValueError):
                    pass

            scores.loc[row, col] = max(0.0, min(1.0, score))

        return scores
