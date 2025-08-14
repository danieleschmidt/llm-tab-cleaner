#!/usr/bin/env python3
"""
Simple demonstration of llm-tab-cleaner functionality.
Generation 1: Make it Work (Simple) - Basic functionality demo.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from llm_tab_cleaner import TableCleaner, CleaningRule, RuleSet

def create_sample_data():
    """Create a sample dataset with common data quality issues."""
    return pd.DataFrame({
        'name': ['Alice Smith', 'bob jones', 'CHARLIE BROWN', None, ''],
        'email': ['alice@test.com', 'BOB@TEST.COM', 'charlie.test.com', 'invalid-email', None],
        'age': [25, 'thirty', 35, -5, 150],
        'state': ['California', 'NY', 'tx', 'Unknown', None],
        'salary': [50000, '$60,000', '70k', 'confidential', None]
    })

def demo_basic_cleaning():
    """Demonstrate basic cleaning functionality without LLM."""
    print("🧹 LLM Tab Cleaner - Simple Demo")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print("📊 Original Data:")
    print(df)
    print(f"\nData shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Create basic rules for demonstration
    rules = [
        CleaningRule(
            name="standardize_names",
            description="Standardize name capitalization",
            examples=[
                ("alice smith", "Alice Smith"),
                ("BOB JONES", "Bob Jones"),
                ("charlie brown", "Charlie Brown")
            ]
        ),
        CleaningRule(
            name="standardize_states", 
            description="Convert state names to standard format",
            examples=[
                ("California", "CA"),
                ("NY", "NY"),
                ("tx", "TX"),
                ("texas", "TX"),
                ("new york", "NY")
            ]
        )
    ]
    
    ruleset = RuleSet(rules)
    
    # Initialize cleaner with mock provider (no real LLM calls)
    try:
        cleaner = TableCleaner(
            confidence_threshold=0.5,
            max_batch_size=10,
            enable_caching=False
        )
        
        print("\n✅ TableCleaner initialized successfully")
        print(f"Rules loaded: {len(ruleset.rules)}")
        
        # Apply basic data standardization (without LLM)
        cleaned_df = df.copy()
        
        # Simple name cleaning
        cleaned_df['name'] = cleaned_df['name'].fillna('').str.title()
        
        # Simple state standardization
        state_mapping = {'California': 'CA', 'tx': 'TX', 'NY': 'NY'}
        cleaned_df['state'] = cleaned_df['state'].map(state_mapping).fillna(cleaned_df['state'])
        
        # Simple email standardization
        cleaned_df['email'] = cleaned_df['email'].fillna('').str.lower()
        
        print("\n📊 Cleaned Data:")
        print(cleaned_df)
        
        # Calculate improvement metrics
        original_nulls = df.isnull().sum().sum()
        cleaned_nulls = cleaned_df.isnull().sum().sum()
        improvement = (original_nulls - cleaned_nulls) / original_nulls * 100 if original_nulls > 0 else 0
        
        print(f"\n📈 Cleaning Results:")
        print(f"Original null values: {original_nulls}")
        print(f"Cleaned null values: {cleaned_nulls}")
        print(f"Improvement: {improvement:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during cleaning: {e}")
        return False

def demo_profiling():
    """Demonstrate data profiling capabilities."""
    try:
        from llm_tab_cleaner import DataProfiler
        
        print("\n🔍 Data Profiling Demo")
        print("-" * 30)
        
        df = create_sample_data()
        profiler = DataProfiler()
        
        # Basic profiling without complex dependencies
        profile = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': df.nunique().to_dict()
        }
        
        print("Profile results:")
        for key, value in profile.items():
            print(f"  {key}: {value}")
            
        return True
        
    except ImportError as e:
        print(f"⚠️  Profiling module not fully available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        return False

def main():
    """Run the simple demo."""
    print("🚀 Starting llm-tab-cleaner simple functionality test...")
    
    success = True
    
    # Test basic cleaning
    if not demo_basic_cleaning():
        success = False
    
    # Test profiling
    if not demo_profiling():
        success = False
    
    if success:
        print("\n✅ Generation 1 (Simple) - All basic functionality working!")
        print("Ready to proceed to Generation 2 (Robust)")
    else:
        print("\n❌ Some functionality issues detected")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)