"""Command line interface for llm-tab-cleaner."""

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-powered data cleaning pipeline"
    )
    
    parser.add_argument(
        "input_file",
        help="Input CSV/Parquet file to clean"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (defaults to input_cleaned.csv)"
    )
    
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "local"],
        default="anthropic",
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Minimum confidence threshold for applying fixes"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
        
    # Set output path if not specified
    if not args.output:
        output_path = input_path.with_name(
            f"{input_path.stem}_cleaned{input_path.suffix}"
        )
    else:
        output_path = Path(args.output)
        
    print(f"Cleaning {input_path} -> {output_path}")
    print(f"Provider: {args.provider}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    # TODO: Implement actual cleaning logic
    print("Cleaning functionality not yet implemented")


if __name__ == "__main__":
    main()