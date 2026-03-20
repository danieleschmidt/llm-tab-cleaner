"""Cleaning rules: MissingValueImputer, TypeCoercer, OutlierDetector, DuplicateRemover.

Each cleaner implements the BaseCleaningRule interface, making it trivial to plug in
an LLM-based cleaner that follows the same contract.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import pandas as pd


Change = dict[str, Any]  # {row, col, old_val, new_val, action}


class BaseCleaningRule(abc.ABC):
    """Abstract base class for all cleaning rules.

    Implementing this interface is all that's needed to plug in an
    LLM-based cleaner (or any other strategy).
    """

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseCleaningRule":
        """Learn statistics / parameters from *df* (training split or full dataset)."""
        ...

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Change]]:
        """Apply the rule to *df*.

        Returns:
            (cleaned_df, changes) where *changes* is a list of dicts with keys:
            ``row``, ``col``, ``old_val``, ``new_val``, ``action``.
        """
        ...

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Change]]:
        self.fit(df)
        return self.transform(df)


# ---------------------------------------------------------------------------
# Concrete cleaners
# ---------------------------------------------------------------------------


class MissingValueImputer(BaseCleaningRule):
    """Fill NaN values with column mean (numeric) or mode (string/object)."""

    def __init__(self) -> None:
        self._fill_values: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "MissingValueImputer":
        self._fill_values = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                val = df[col].mean()
                self._fill_values[col] = val
            else:
                mode = df[col].mode()
                self._fill_values[col] = mode.iloc[0] if len(mode) > 0 else None
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Change]]:
        df = df.copy()
        changes: list[Change] = []
        for col, fill_val in self._fill_values.items():
            if col not in df.columns:
                continue
            if fill_val is None:
                continue
            mask = df[col].isna()
            for row_idx in df.index[mask]:
                changes.append(
                    {
                        "row": int(row_idx),
                        "col": col,
                        "old_val": None,
                        "new_val": fill_val,
                        "action": "impute_missing",
                    }
                )
            df[col] = df[col].fillna(fill_val)
        return df, changes


class TypeCoercer(BaseCleaningRule):
    """Convert columns to numeric when ≥80 % of non-null values are parseable."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self._coerce_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "TypeCoercer":
        self._coerce_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            converted = pd.to_numeric(non_null, errors="coerce")
            parseable_ratio = converted.notna().sum() / len(non_null)
            if parseable_ratio >= self.threshold:
                self._coerce_cols.append(col)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Change]]:
        df = df.copy()
        changes: list[Change] = []
        for col in self._coerce_cols:
            if col not in df.columns:
                continue
            original = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            for row_idx in df.index:
                old = original.loc[row_idx]
                new = df.loc[row_idx, col]
                if old != new and not (
                    pd.isna(old) and pd.isna(new)
                ):
                    changes.append(
                        {
                            "row": int(row_idx),
                            "col": col,
                            "old_val": old,
                            "new_val": new,
                            "action": "type_coerce",
                        }
                    )
        return df, changes


class OutlierDetector(BaseCleaningRule):
    """Detect and handle outliers using the IQR method.

    Parameters
    ----------
    strategy:
        ``"clip"`` — replace with IQR fence values (default).
        ``"remove"`` — drop the entire row.
    iqr_factor:
        Multiplier for IQR to compute fences (default 1.5).
    """

    def __init__(self, strategy: str = "clip", iqr_factor: float = 1.5) -> None:
        if strategy not in ("clip", "remove"):
            raise ValueError("strategy must be 'clip' or 'remove'")
        self.strategy = strategy
        self.iqr_factor = iqr_factor
        self._fences: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "OutlierDetector":
        self._fences = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
            self._fences[col] = (lower, upper)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Change]]:
        df = df.copy()
        changes: list[Change] = []

        if self.strategy == "clip":
            for col, (lower, upper) in self._fences.items():
                if col not in df.columns:
                    continue
                mask = (df[col] < lower) | (df[col] > upper)
                for row_idx in df.index[mask]:
                    old_val = df.loc[row_idx, col]
                    new_val = float(np.clip(old_val, lower, upper))
                    changes.append(
                        {
                            "row": int(row_idx),
                            "col": col,
                            "old_val": old_val,
                            "new_val": new_val,
                            "action": "clip_outlier",
                        }
                    )
                df[col] = df[col].clip(lower=lower, upper=upper)

        elif self.strategy == "remove":
            rows_to_drop: set[int] = set()
            for col, (lower, upper) in self._fences.items():
                if col not in df.columns:
                    continue
                outlier_mask = (df[col] < lower) | (df[col] > upper)
                for row_idx in df.index[outlier_mask]:
                    if int(row_idx) not in rows_to_drop:
                        rows_to_drop.add(int(row_idx))
                        changes.append(
                            {
                                "row": int(row_idx),
                                "col": col,
                                "old_val": df.loc[row_idx, col],
                                "new_val": None,
                                "action": "remove_outlier_row",
                            }
                        )
            df = df.drop(index=list(rows_to_drop)).reset_index(drop=True)

        return df, changes


class DuplicateRemover(BaseCleaningRule):
    """Remove exact duplicate rows (keeps first occurrence)."""

    def __init__(self) -> None:
        self._duplicate_indices: list[int] = []

    def fit(self, df: pd.DataFrame) -> "DuplicateRemover":
        self._duplicate_indices = list(
            df[df.duplicated(keep="first")].index.astype(int)
        )
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[Change]]:
        changes: list[Change] = []
        dup_mask = df.duplicated(keep="first")
        for row_idx in df.index[dup_mask]:
            changes.append(
                {
                    "row": int(row_idx),
                    "col": "__row__",
                    "old_val": df.loc[row_idx].to_dict(),
                    "new_val": None,
                    "action": "remove_duplicate",
                }
            )
        df = df[~dup_mask].reset_index(drop=True)
        return df, changes
