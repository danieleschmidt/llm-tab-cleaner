"""AuditTrail: records every cleaning change with confidence and timestamp."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


class AuditTrail:
    """Immutable log of all cleaning operations.

    Each entry:
        row_idx   - integer row index in the original dataframe
        col       - column name (or '__row__' for row-level ops)
        old_value - value before cleaning
        new_value - value after cleaning (None = removed)
        action    - string describing the operation
        confidence- float 0-1 confidence score
        timestamp - ISO 8601 UTC string
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        row_idx: int,
        col: str,
        old_value: Any,
        new_value: Any,
        action: str,
        confidence: float = 1.0,
        timestamp: str | None = None,
    ) -> None:
        self._entries.append(
            {
                "row_idx": row_idx,
                "col": col,
                "old_value": old_value,
                "new_value": new_value,
                "action": action,
                "confidence": confidence,
                "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            }
        )

    def record_change(self, change: dict[str, Any], confidence: float = 1.0) -> None:
        """Convenience wrapper that accepts a change dict from a cleaner."""
        self.record(
            row_idx=change["row"],
            col=change["col"],
            old_value=change["old_val"],
            new_value=change["new_val"],
            action=change["action"],
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def summary(self) -> dict[str, int]:
        """Return counts grouped by *action*."""
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry["action"]] = counts.get(entry["action"], 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self._entries, indent=indent, default=str)

    @classmethod
    def from_json(cls, data: str) -> "AuditTrail":
        trail = cls()
        trail._entries = json.loads(data)
        return trail

    def __repr__(self) -> str:  # pragma: no cover
        return f"AuditTrail(entries={len(self._entries)})"
