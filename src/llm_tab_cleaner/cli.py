"""Command line interface for llm-tab-cleaner."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import click
import pandas as pd

from .core import TableCleaner
from .cleaning_rule import create_default_rules

__version__ = "0.3.0"


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(level=level, format=format_string)


def load_file(file_path: Path) -> pd.DataFrame:
    """Load data file (CSV, Parquet, Excel, etc.)."""
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix in ['.parquet', '.pq']:
            return pd.read_parquet(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.jsonl':
            return pd.read_json(file_path, lines=True)
        else:
            # Try CSV as fallback
            return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)


def save_file(df: pd.DataFrame, file_path: Path) -> None:
    """Save DataFrame to file."""
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.csv':
            df.to_csv(file_path, index=False)
        elif suffix in ['.parquet', '.pq']:
            df.to_parquet(file_path, index=False)
        elif suffix in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False)
        elif suffix == '.json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            # Default to CSV
            df.to_csv(file_path, index=False)
        
        print(f"✓ Saved cleaned data to {file_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")
        sys.exit(1)


def save_report(report: Any, file_path: Path) -> None:
    """Save cleaning report to JSON file."""
    try:
        report_data = {
            "total_fixes": report.total_fixes,
            "quality_score": report.quality_score,
            "processing_time": report.processing_time,
            "profile_summary": report.profile_summary,
            "fixes": [
                {
                    "column": fix.column,
                    "row_index": fix.row_index,
                    "original": str(fix.original),
                    "cleaned": str(fix.cleaned),
                    "confidence": fix.confidence,
                    "reasoning": fix.reasoning,
                    "rule_applied": fix.rule_applied
                }
                for fix in report.fixes
            ],
            "audit_trail": report.audit_trail
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"✓ Saved cleaning report to {file_path}")
    except Exception as e:
        print(f"Error saving report {file_path}: {e}")


def print_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, report: Any) -> None:
    """Print cleaning summary."""
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    
    print(f"📊 Dataset: {len(df_before)} rows × {len(df_before.columns)} columns")
    print(f"⏱️  Processing time: {report.processing_time:.2f} seconds")
    print(f"🔧 Total fixes applied: {report.total_fixes}")
    print(f"📈 Quality score: {report.quality_score:.2%}")
    
    if report.profile_summary:
        print(f"📋 Original quality: {report.profile_summary['overall_quality_score']:.2%}")
        print(f"🚨 Total issues found: {report.profile_summary['total_issues']}")
        if report.profile_summary['duplicate_percentage'] > 0:
            print(f"🔄 Duplicate rows: {report.profile_summary['duplicate_percentage']:.1f}%")
    
    # Show top fixes by column
    if report.fixes:
        print(f"\n📝 Top fixes by column:")
        column_counts = {}
        for fix in report.fixes:
            if fix.confidence >= 0.85:  # Only count high-confidence fixes
                column_counts[fix.column] = column_counts.get(fix.column, 0) + 1
        
        for column, count in sorted(column_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {column}: {count} fixes")
    
    print("="*60)


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """LLM-powered data cleaning pipeline.
    
    Clean messy data using Large Language Models with confidence-gated corrections
    and comprehensive audit trails for production ETL pipelines.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file path (defaults to input_cleaned.ext)')
@click.option('--provider', type=click.Choice(['anthropic', 'openai', 'local']),
              default='local', help='LLM provider to use (default: local)')
@click.option('--confidence-threshold', type=float, default=0.85,
              help='Minimum confidence threshold for applying fixes (default: 0.85)')
@click.option('--columns', help='Comma-separated list of columns to clean (default: all)')
@click.option('--sample-rate', type=float, default=1.0,
              help='Fraction of data to process (0.0-1.0, default: 1.0)')
@click.option('--use-rules', is_flag=True,
              help='Use built-in cleaning rules in addition to LLM')
@click.option('--save-report', is_flag=True,
              help='Save detailed cleaning report to JSON file')
@click.option('--no-profiling', is_flag=True,
              help='Disable data profiling (faster but less context)')
@click.option('--max-fixes-per-column', type=int, default=1000,
              help='Maximum fixes to attempt per column (default: 1000)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--rules', type=click.Path(exists=True, path_type=Path),
              help='Custom cleaning rules YAML file')
@click.option('--batch', is_flag=True, help='Batch processing mode')
@click.option('--output-dir', type=click.Path(path_type=Path),
              help='Output directory for batch processing')
def clean(input_file: Path, output: Optional[Path], provider: str, confidence_threshold: float,
          columns: Optional[str], sample_rate: float, use_rules: bool, save_report: bool,
          no_profiling: bool, max_fixes_per_column: int, verbose: bool, 
          rules: Optional[Path], batch: bool, output_dir: Optional[Path]):
    """Clean a data file using LLM-powered data quality improvements.
    
    Examples:
      llm-clean clean data.csv
      llm-clean clean data.csv --provider openai --confidence-threshold 0.9
      llm-clean clean data.parquet --columns name,email --sample-rate 0.1
    """
    # Setup logging
    setup_logging(verbose)
    
    # Validate parameters
    if not 0.0 <= sample_rate <= 1.0:
        click.echo(f"❌ Error: Sample rate must be between 0.0 and 1.0", err=True)
        raise click.Abort()
    
    if not 0.0 <= confidence_threshold <= 1.0:
        click.echo(f"❌ Error: Confidence threshold must be between 0.0 and 1.0", err=True)
        raise click.Abort()
    
    # Handle batch processing
    if batch:
        if not input_file.is_dir():
            click.echo(f"❌ Error: Input must be a directory for batch processing", err=True)
            raise click.Abort()
        
        if not output_dir:
            click.echo(f"❌ Error: --output-dir required for batch processing", err=True)
            raise click.Abort()
            
        # Batch processing not yet implemented
        click.echo(f"❌ Error: Batch processing not implemented yet", err=True)
        raise click.Abort()
    
    # Set output path if not specified
    if not output:
        output_path = input_file.with_name(
            f"{input_file.stem}_cleaned{input_file.suffix}"
        )
    else:
        output_path = output
    
    click.echo(f"🧹 LLM Tab Cleaner v{__version__}")
    click.echo(f"📁 Input: {input_file}")
    click.echo(f"📁 Output: {output_path}")
    click.echo(f"🤖 Provider: {provider}")
    click.echo(f"🎯 Confidence threshold: {confidence_threshold}")
    if columns:
        click.echo(f"📊 Columns: {columns}")
    if sample_rate < 1.0:
        click.echo(f"🎲 Sample rate: {sample_rate:.1%}")
    
    # Load data
    click.echo(f"\n📖 Loading data from {input_file}...")
    try:
        df = load_file(input_file)
        click.echo(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    except Exception as e:
        click.echo(f"❌ Error loading file: {e}", err=True)
        raise click.Abort()
    
    # Prepare cleaning configuration
    cleaning_rules = None
    if use_rules or rules:
        if rules:
            # TODO: Load custom rules from YAML file
            click.echo(f"⚠️  Custom rules loading not implemented yet, using defaults")
        cleaning_rules = create_default_rules()
    
    column_list = columns.split(',') if columns else None
    
    try:
        # Initialize cleaner
        click.echo(f"\n🔧 Initializing cleaner...")
        cleaner = TableCleaner(
            llm_provider=provider,
            confidence_threshold=confidence_threshold,
            rules=cleaning_rules,
            enable_profiling=not no_profiling,
            max_fixes_per_column=max_fixes_per_column
        )
        
        # Clean the data
        click.echo(f"🚀 Starting cleaning process...")
        
        cleaned_df, report = cleaner.clean(
            df=df,
            columns=column_list,
            sample_rate=sample_rate
        )
        
        # Print summary
        print_summary(df, cleaned_df, report)
        
        # Save cleaned data
        save_file(cleaned_df, output_path)
        
        # Save report if requested
        if save_report:
            report_path = output_path.with_suffix('.report.json')
            save_report(report, report_path)
        
        click.echo(f"\n✅ Cleaning completed successfully!")
        
        # Provide feedback
        if report.total_fixes > 0:
            click.echo(f"💡 {report.total_fixes} issues were fixed. Review the changes before using in production.")
        else:
            click.echo(f"✨ No issues found - your data looks clean!")
        
    except KeyboardInterrupt:
        click.echo(f"\n⏹️  Cleaning interrupted by user")
        raise click.Abort()
    except Exception as e:
        click.echo(f"\n❌ Error during cleaning: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


def main() -> None:
    """Entry point for console script."""
    cli()


if __name__ == "__main__":
    main()