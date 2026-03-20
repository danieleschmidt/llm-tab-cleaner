"""CLI entry point for tab-cleaner."""

from __future__ import annotations

import sys

import click
import pandas as pd

from .pipeline import CleaningPipeline


@click.group()
def cli() -> None:
    """tab-cleaner — LLM-ready data cleaning pipeline."""


@cli.command()
@click.option("--input", "input_path", required=True, help="Input CSV file")
@click.option("--output", "output_path", required=True, help="Output CSV file")
@click.option("--audit", "audit_path", default=None, help="Write audit trail to JSON file")
@click.option(
    "--outlier-strategy",
    default="clip",
    show_default=True,
    type=click.Choice(["clip", "remove"]),
    help="How to handle outliers",
)
def clean(
    input_path: str,
    output_path: str,
    audit_path: str | None,
    outlier_strategy: str,
) -> None:
    """Clean a CSV file and write the result."""
    df = pd.read_csv(input_path)
    pipeline = CleaningPipeline(config={"outlier_strategy": outlier_strategy})
    cleaned_df, audit = pipeline.fit_transform(df)
    cleaned_df.to_csv(output_path, index=False)
    click.echo(
        f"Cleaned {len(df)} → {len(cleaned_df)} rows, {len(audit)} changes applied."
    )
    if audit_path:
        with open(audit_path, "w") as f:
            f.write(audit.to_json())
        click.echo(f"Audit trail written to {audit_path}")


@cli.command()
@click.option("--input", "input_path", required=True, help="Input CSV file")
def info(input_path: str) -> None:
    """Show basic statistics about a CSV file."""
    df = pd.read_csv(input_path)
    click.echo(f"Rows:    {len(df)}")
    click.echo(f"Columns: {len(df.columns)}")
    click.echo(f"Missing: {df.isna().sum().sum()} cells")
    click.echo(f"Dupes:   {df.duplicated().sum()} rows")
    click.echo("\nColumn types:")
    for col, dtype in df.dtypes.items():
        missing = df[col].isna().sum()
        click.echo(f"  {col}: {dtype}  (missing={missing})")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
