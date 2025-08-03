"""Command line interface for llm-tab-cleaner."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .core import TableCleaner
from .cleaning_rule import create_default_rules


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


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-powered data cleaning pipeline",
        epilog="Examples:\n"
               "  llm-clean data.csv\n"
               "  llm-clean data.csv --provider openai --confidence-threshold 0.9\n"
               "  llm-clean data.parquet --columns name,email --sample-rate 0.1",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        help="Input file to clean (CSV, Parquet, Excel, JSON)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (defaults to input_cleaned.ext)"
    )
    
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "local"],
        default="local",  # Default to local for demo
        help="LLM provider to use (default: local)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Minimum confidence threshold for applying fixes (default: 0.85)"
    )
    
    parser.add_argument(
        "--columns",
        help="Comma-separated list of columns to clean (default: all)"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Fraction of data to process (0.0-1.0, default: 1.0)"
    )
    
    parser.add_argument(
        "--use-rules",
        action="store_true",
        help="Use built-in cleaning rules in addition to LLM"
    )
    
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save detailed cleaning report to JSON file"
    )
    
    parser.add_argument(
        "--no-profiling",
        action="store_true",
        help="Disable data profiling (faster but less context)"
    )
    
    parser.add_argument(
        "--max-fixes-per-column",
        type=int,
        default=1000,
        help="Maximum fixes to attempt per column (default: 1000)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    # Set output path if not specified
    if not args.output:
        output_path = input_path.with_name(
            f"{input_path.stem}_cleaned{input_path.suffix}"
        )
    else:
        output_path = Path(args.output)
    
    # Validate parameters
    if not 0.0 <= args.sample_rate <= 1.0:
        print(f"❌ Error: Sample rate must be between 0.0 and 1.0")
        sys.exit(1)
    
    if not 0.0 <= args.confidence_threshold <= 1.0:
        print(f"❌ Error: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    print(f"🧹 LLM Tab Cleaner")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    print(f"🤖 Provider: {args.provider}")
    print(f"🎯 Confidence threshold: {args.confidence_threshold}")
    if args.columns:
        print(f"📊 Columns: {args.columns}")
    if args.sample_rate < 1.0:
        print(f"🎲 Sample rate: {args.sample_rate:.1%}")
    
    # Load data
    print(f"\n📖 Loading data from {input_path}...")
    df = load_file(input_path)
    print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")
    
    # Prepare cleaning configuration
    rules = create_default_rules() if args.use_rules else None
    columns = args.columns.split(',') if args.columns else None
    
    try:
        # Initialize cleaner
        print(f"\n🔧 Initializing cleaner...")
        cleaner = TableCleaner(
            llm_provider=args.provider,
            confidence_threshold=args.confidence_threshold,
            rules=rules,
            enable_profiling=not args.no_profiling,
            max_fixes_per_column=args.max_fixes_per_column
        )
        
        # Clean the data
        print(f"🚀 Starting cleaning process...")
        start_time = time.time()
        
        cleaned_df, report = cleaner.clean(
            df=df,
            columns=columns,
            sample_rate=args.sample_rate
        )
        
        # Print summary
        print_summary(df, cleaned_df, report)
        
        # Save cleaned data
        save_file(cleaned_df, output_path)
        
        # Save report if requested
        if args.save_report:
            report_path = output_path.with_suffix('.report.json')
            save_report(report, report_path)
        
        print(f"\n✅ Cleaning completed successfully!")
        
        # Exit with appropriate code
        if report.total_fixes > 0:
            print(f"💡 {report.total_fixes} issues were fixed. Review the changes before using in production.")
        else:
            print(f"✨ No issues found - your data looks clean!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Cleaning interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during cleaning: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()