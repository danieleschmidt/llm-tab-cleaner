# tab-cleaner

LLM-ready tabular data cleaning pipeline with full audit trails and per-cell confidence scoring.

Rule-based cleaners ship out of the box. The `BaseCleaningRule` interface lets you drop in an LLM-backed cleaner with zero pipeline changes.

## Features

- **MissingValueImputer** — fills NaN with column mean (numeric) or mode (string)
- **TypeCoercer** — converts columns to numeric when ≥80 % of values are parseable
- **OutlierDetector** — IQR-based outlier detection; clip or remove strategy
- **DuplicateRemover** — drops exact duplicate rows
- **ConfidenceScorer** — assigns a 0–1 confidence score to every cleaned cell
- **AuditTrail** — JSON-serializable log of every change (row, col, old→new, action, confidence, timestamp)
- **FastAPI** endpoint for HTTP access
- **CLI** for file-based workflows

## Install

```bash
pip install -e .
```

## CLI Usage

```bash
# Clean a CSV file
tab-cleaner clean --input data.csv --output cleaned.csv --audit audit.json

# Show stats about a file
tab-cleaner info --input data.csv
```

## API Usage

Start the server:

```bash
uvicorn tab_cleaner.api:app --reload
```

POST to `/clean`:

```bash
curl -X POST http://localhost:8000/clean \
  -H "Content-Type: application/json" \
  -d '{"csv_data": "age,salary\n25,50000\n,60000\n30,99999\n"}'
```

Response:

```json
{
  "cleaned_csv": "age,salary\n...",
  "audit_json": "[{\"row_idx\": 1, \"col\": \"age\", ...}]",
  "rows_in": 3,
  "rows_out": 3,
  "changes": 2
}
```

## Extending with an LLM Cleaner

```python
from tab_cleaner.cleaners import BaseCleaningRule
import pandas as pd

class LLMCleaner(BaseCleaningRule):
    def fit(self, df: pd.DataFrame) -> "LLMCleaner":
        return self  # optionally sample rows for context

    def transform(self, df: pd.DataFrame):
        # call your LLM here, return (cleaned_df, changes)
        ...

from tab_cleaner.pipeline import CleaningPipeline
pipeline = CleaningPipeline(cleaners=[LLMCleaner()])
cleaned_df, audit = pipeline.fit_transform(df)
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
