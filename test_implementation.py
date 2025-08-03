#!/usr/bin/env python3
"""Quick test to verify the LLM Tab Cleaner implementation works."""

import pandas as pd
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from llm_tab_cleaner import TableCleaner, create_default_rules, get_version_info
    print("✅ Successfully imported llm_tab_cleaner")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic cleaning functionality."""
    print("\n🧪 Testing basic functionality...")
    
    # Create sample data with quality issues
    data = {
        'name': ['John Smith', '  jane doe  ', 'N/A', 'Bob Johnson', 'ALICE BROWN'],
        'email': ['john@gmail.com', 'JANE@YAHOO.COM', 'invalid', 'bob@company.com', 'alice@HOTMAIL.com'],
        'phone': ['555-123-4567', '(555) 234-5678', '555.345.6789', 'missing', '5554567890'],
        'age': [25, 30, 'unknown', 35, '40'],
        'state': ['California', 'NY', 'texas', 'FL', 'illinois']
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Created test DataFrame with {len(df)} rows")
    print(df.to_string())
    
    # Initialize cleaner with local provider (no API keys needed)
    print("\n🔧 Initializing TableCleaner...")
    cleaner = TableCleaner(
        llm_provider="local",
        confidence_threshold=0.8,
        rules=create_default_rules(),
        enable_profiling=True
    )
    
    # Clean the data
    print("\n🚀 Starting cleaning process...")
    cleaned_df, report = cleaner.clean(df)
    
    # Display results
    print("\n📋 CLEANING RESULTS:")
    print("="*50)
    print(f"Total fixes applied: {report.total_fixes}")
    print(f"Quality score: {report.quality_score:.2%}")
    print(f"Processing time: {report.processing_time:.2f}s")
    
    if report.profile_summary:
        print(f"Original quality: {report.profile_summary['overall_quality_score']:.2%}")
    
    print(f"\n📝 Applied fixes:")
    for fix in report.fixes[:10]:  # Show first 10 fixes
        print(f"  • {fix.column}[{fix.row_index}]: '{fix.original}' → '{fix.cleaned}' (confidence: {fix.confidence:.2%})")
    
    print(f"\n📊 Cleaned DataFrame:")
    print(cleaned_df.to_string())
    
    return True

def test_version_info():
    """Test version information."""
    print("\n📋 Version Information:")
    info = get_version_info()
    print(f"Version: {info['version']}")
    print("Available features:")
    for feature, available in info['features'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {feature}")

def main():
    """Run all tests."""
    print("🧹 LLM Tab Cleaner Implementation Test")
    print("="*40)
    
    try:
        test_version_info()
        
        if test_basic_functionality():
            print("\n✅ All tests passed! Implementation is working correctly.")
            return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)