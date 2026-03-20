"""FastAPI application for tab-cleaner."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .pipeline import CleaningPipeline

app = FastAPI(
    title="tab-cleaner",
    description="LLM-ready data cleaning pipeline",
    version="1.0.0",
)


class CleanRequest(BaseModel):
    csv_data: str
    outlier_strategy: str = "clip"


class CleanResponse(BaseModel):
    cleaned_csv: str
    audit_json: str
    rows_in: int
    rows_out: int
    changes: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.post("/clean", response_model=CleanResponse)
def clean(request: CleanRequest) -> CleanResponse:
    try:
        df = pd.read_csv(io.StringIO(request.csv_data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc

    config = {"outlier_strategy": request.outlier_strategy}
    pipeline = CleaningPipeline(config=config)
    cleaned_df, audit = pipeline.fit_transform(df)

    return CleanResponse(
        cleaned_csv=cleaned_df.to_csv(index=False),
        audit_json=audit.to_json(),
        rows_in=len(df),
        rows_out=len(cleaned_df),
        changes=len(audit),
    )
