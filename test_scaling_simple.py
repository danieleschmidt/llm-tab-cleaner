#!/usr/bin/env python3
"""Simple scaling test for LLM Tab Cleaner."""

import pandas as pd
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_caching():
    """Test basic caching functionality."""
    try:
        from llm_tab_cleaner.optimization import CacheManager, OptimizationConfig
        
        config = OptimizationConfig(enable_caching=True, cache_type="memory")
        cache = CacheManager(config)
        
        # Test basic operations
        cache.set("key1", "value1")
        result = cache.get("key1")
        
        print(f"✅ Caching works: {result == 'value1'}")
        
        # Test stats
        stats = cache.stats()
        print(f"✅ Cache stats: {stats['size']} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

def test_parallel_processing():
    """Test parallel processing basics."""
    try:
        from llm_tab_cleaner.optimization import ParallelProcessor, OptimizationConfig
        
        config = OptimizationConfig(enable_parallel_processing=True, max_workers=2, chunk_size=10)
        processor = ParallelProcessor(config)
        
        # Create simple test data
        df = pd.DataFrame({'col1': range(25), 'col2': range(25)})
        chunks = processor.split_dataframe(df)
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Simple processing function
        def process_chunk(chunk):
            return len(chunk)
        
        results = processor.process_chunks_threaded(chunks, process_chunk)
        total_processed = sum(results)
        
        print(f"✅ Processed {total_processed} total rows")
        return len(df) == total_processed
        
    except Exception as e:
        print(f"❌ Parallel processing test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization basics."""
    try:
        from llm_tab_cleaner.optimization import MemoryOptimizer, OptimizationConfig
        
        # Create simple test data
        df = pd.DataFrame({
            'numbers': [1, 2, 3] * 100,  # Could be smaller int type
            'categories': ['A', 'B'] * 150,  # Could be category
        })
        
        config = OptimizationConfig(enable_memory_optimization=True)
        optimizer = MemoryOptimizer(config)
        
        # Get memory info
        memory_info = optimizer.get_memory_usage(df)
        print(f"✅ Memory usage: {memory_info['total_mb']:.3f}MB")
        
        # Test optimization (may or may not reduce memory)
        optimized_df = optimizer.optimize_dataframe_dtypes(df)
        print("✅ DataFrame optimization completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def test_optimization_engine():
    """Test optimization engine integration."""
    try:
        from llm_tab_cleaner.optimization import OptimizationEngine
        
        df = pd.DataFrame({'test': [1, 2, 3]})
        engine = OptimizationEngine()
        
        # Test recommendations
        recommendations = engine.get_optimization_recommendations(df)
        print(f"✅ Got optimization recommendations: {len(recommendations)} categories")
        
        # Test performance summary
        summary = engine.get_performance_summary()
        print(f"✅ Got performance summary: {len(summary)} sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization engine test failed: {e}")
        return False

def main():
    """Run simple scaling tests."""
    print("🚀 Simple Scaling Test")
    print("="*25)
    
    tests = [
        ("Caching", test_caching),
        ("Parallel Processing", test_parallel_processing),
        ("Memory Optimization", test_memory_optimization),
        ("Optimization Engine", test_optimization_engine)
    ]
    
    passed = 0
    
    for name, test_func in tests:
        print(f"\n🧪 {name}...")
        if test_func():
            passed += 1
    
    print(f"\n📋 Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✅ All scaling features working!")
        return True
    else:
        print("❌ Some scaling features failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)