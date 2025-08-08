#!/usr/bin/env python3
"""Simple robustness test for LLM Tab Cleaner."""

import pandas as pd
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_security_basics():
    """Test basic security features."""
    try:
        from llm_tab_cleaner.security import SecurityManager, SecurityConfig
        
        # Test basic validation
        config = SecurityConfig(max_rows=10, allow_sensitive_columns=True)
        manager = SecurityManager(config)
        
        # Small safe dataset
        df = pd.DataFrame({'name': ['John', 'Jane'], 'age': [25, 30]})
        result = manager.validate_and_prepare_data(df, "test")
        
        print(f"✅ Security validation: {result['passed']}")
        print(f"   Data hash: {result['data_hash'][:8]}...")
        
        # Cleanup
        manager.finalize_operation(result['operation_id'], True)
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

def test_monitoring_basics():
    """Test basic monitoring features."""
    try:
        from llm_tab_cleaner.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Test basic metrics
        collector.record_counter("test_counter", 1.0)
        collector.record_gauge("test_gauge", 42.0)
        
        summary = collector.get_metric_summary("test_counter")
        print(f"✅ Metrics collection working: latest = {summary.get('latest', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False

def test_resilience_basics():
    """Test basic resilience features.""" 
    try:
        from llm_tab_cleaner.core import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=2)
        
        def test_func():
            return "success"
        
        result = cb.call(test_func)
        print(f"✅ Circuit breaker working: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Resilience test failed: {e}")
        return False

def main():
    """Run simple robustness tests.""" 
    print("🛡️ Simple Robustness Test")
    print("="*30)
    
    tests = [
        ("Security Basics", test_security_basics),
        ("Monitoring Basics", test_monitoring_basics), 
        ("Resilience Basics", test_resilience_basics)
    ]
    
    passed = 0
    
    for name, test_func in tests:
        print(f"\n🧪 {name}...")
        if test_func():
            passed += 1
    
    print(f"\n📋 Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("✅ All robustness features working!")
        return True
    else:
        print("❌ Some robustness features failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)