"""CleaningPipeline: orchestrates cleaners, builds audit trail and confidence scores."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .audit import AuditTrail
from .cleaners import (
    BaseCleaningRule,
    DuplicateRemover,
    MissingValueImputer,
    OutlierDetector,
    TypeCoercer,
)
from .confidence import ConfidenceScorer


class CleaningPipeline:
    """Run a sequence of :class:`BaseCleaningRule` instances and collect results.

    Parameters
    ----------
    cleaners:
        Ordered list of cleaners to apply.  When *None* a default pipeline
        (imputer → coercer → outlier detector → duplicate remover) is created.
    config:
        Optional dict passed to built-in cleaners when building the default
        pipeline.  Recognised keys:
            ``outlier_strategy`` – ``"clip"`` (default) or ``"remove"``
            ``outlier_iqr_factor`` – float, default 1.5
            ``type_coerce_threshold`` – float, default 0.8
    """

    def __init__(
        self,
        cleaners: list[BaseCleaningRule] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or {}
        if cleaners is not None:
            self._cleaners: list[BaseCleaningRule] = list(cleaners)
        else:
            self._cleaners = self._default_cleaners()

    def _default_cleaners(self) -> list[BaseCleaningRule]:
        return [
            MissingValueImputer(),
            TypeCoercer(
                threshold=self._config.get("type_coerce_threshold", 0.8)
            ),
            OutlierDetector(
                strategy=self._config.get("outlier_strategy", "clip"),
                iqr_factor=self._config.get("outlier_iqr_factor", 1.5),
            ),
            DuplicateRemover(),
        ]

    def add_cleaner(self, cleaner: BaseCleaningRule) -> "CleaningPipeline":
        self._cleaners.append(cleaner)
        return self

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, AuditTrail]:
        """Apply all cleaners in order.

        Returns:
            (cleaned_df, audit_trail)
        """
        df_original = df.copy()
        current_df = df.copy()
        audit = AuditTrail()
        scorer = ConfidenceScorer()

        all_changes: list[dict[str, Any]] = []

        for cleaner in self._cleaners:
            cleaner.fit(current_df)
            current_df, changes = cleaner.transform(current_df)
            all_changes.extend(changes)

        # Compute confidence for all accumulated changes
        # (we use the original df for comparison)
        scores = scorer.score_dataframe(df_original, current_df, all_changes)

        for change in all_changes:
            row = change["row"]
            col = change["col"]
            if col == "__row__":
                confidence = 0.0
            elif row in scores.index and col in scores.columns:
                confidence = float(scores.loc[row, col])
            else:
                confidence = 1.0
            audit.record_change(change, confidence=confidence)

        return current_df, audit
