#!/usr/bin/env python3
"""Test robustness features of LLM Tab Cleaner."""

import pandas as pd
import sys
import os
import time
import logging
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from llm_tab_cleaner import TableCleaner
    from llm_tab_cleaner.security import SecurityManager, SecurityConfig, create_secure_cleaner
    from llm_tab_cleaner.monitoring import CleaningMonitor, setup_monitoring
    print("✅ Successfully imported robustness modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_security_validation():
    """Test security validation features."""
    print("\n🛡️ Testing security validation...")
    
    # Test with normal data
    normal_data = {
        'name': ['John', 'Jane', 'Bob'],
        'age': [25, 30, 35],
        'city': ['NYC', 'LA', 'Chicago']
    }
    normal_df = pd.DataFrame(normal_data)
    
    security_config = SecurityConfig(
        max_rows=1000,
        max_columns=10,
        allow_sensitive_columns=False
    )
    
    security_manager = SecurityManager(security_config)
    
    try:
        validation_result = security_manager.validate_and_prepare_data(normal_df, "test_cleaning")
        print(f"✅ Normal data validation passed: {validation_result['passed']}")
        print(f"   Data hash: {validation_result['data_hash']}")
        print(f"   Warnings: {len(validation_result['warnings'])}")
        
        # Finalize operation
        security_manager.finalize_operation(validation_result['operation_id'], True, {
            "test": "completed"
        })
        
    except Exception as e:
        print(f"❌ Security validation failed: {e}")
        return False
    
    # Test with sensitive column names
    sensitive_data = {
        'username': ['user1', 'user2'],
        'password': ['secret1', 'secret2'],  # Should trigger warning
        'email': ['a@b.com', 'c@d.com']
    }
    sensitive_df = pd.DataFrame(sensitive_data)
    
    try:
        validation_result = security_manager.validate_and_prepare_data(sensitive_df, "sensitive_test")
        print(f"❌ Sensitive data validation should have failed!")
        return False
    except Exception as e:
        print(f"✅ Sensitive data correctly rejected: {type(e).__name__}")
    
    return True

def test_monitoring_system():
    """Test monitoring and health check systems."""
    print("\n📊 Testing monitoring system...")
    
    try:
        monitor = setup_monitoring()
        
        # Start operation monitoring
        context = monitor.start_operation_monitoring("test-op-001", "data_cleaning")
        print(f"✅ Started monitoring operation: {context['operation_id']}")
        
        # Simulate some work
        time.sleep(0.01)  # Reduced sleep time
        
        # End operation monitoring
        monitor.end_operation_monitoring(
            context, 
            success=True, 
            rows_processed=100,
            fixes_applied=5
        )
        print("✅ Ended monitoring operation successfully")
        
        # Run health checks (timeout protection)
        health_results = monitor.health.run_all_checks()
        print(f"✅ Health checks completed: {len(health_results)} checks")
        
        # Get basic dashboard data
        dashboard = monitor.get_monitoring_dashboard()
        print(f"✅ Dashboard data generated with {len(dashboard)} sections")
        
        return True
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False

def test_resilience_features():
    """Test circuit breakers and retry mechanisms."""
    print("\n🔄 Testing resilience features...")
    
    from llm_tab_cleaner.core import retry_with_backoff, CircuitBreaker
    
    # Test retry decorator
    attempt_count = 0
    
    @retry_with_backoff(max_retries=3, backoff_factor=0.01)
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise Exception("Simulated failure")
        return "success"
    
    try:
        result = flaky_function()
        print(f"✅ Retry mechanism worked: {result} (attempts: {attempt_count})")
    except Exception as e:
        print(f"❌ Retry mechanism failed: {e}")
        return False
    
    # Test circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=1)
    
    def failing_function():
        raise Exception("Always fails")
    
    # Trigger failures to open circuit
    for i in range(3):
        try:
            circuit_breaker.call(failing_function)
        except:
            pass
    
    try:
        circuit_breaker.call(failing_function)
        print("❌ Circuit breaker should be open!")
        return False
    except Exception as e:
        if "Circuit breaker is OPEN" in str(e):
            print("✅ Circuit breaker opened correctly")
        else:
            print(f"❌ Unexpected circuit breaker error: {e}")
            return False
    
    return True

def test_secure_cleaner_integration():
    """Test secure cleaner integration."""
    print("\n🔐 Testing secure cleaner integration...")
    
    # Create test data
    data = {
        'name': ['  John Smith  ', 'jane doe', 'N/A'],
        'email': ['john@test.com', 'JANE@TEST.COM', 'invalid'],
        'status': ['active', 'inactive', 'unknown']
    }
    df = pd.DataFrame(data)
    
    # Create secure cleaner with restrictive config
    security_config = SecurityConfig(
        max_rows=100,
        max_columns=10,
        allow_sensitive_columns=True,  # Allow for this test
        enable_audit_logging=True
    )
    
    try:
        secure_cleaner = create_secure_cleaner(
            config=security_config,
            llm_provider="local",
            confidence_threshold=0.8
        )
        print("✅ Created secure cleaner")
        
        # Clean the data
        cleaned_df, report = secure_cleaner.clean(df)
        print(f"✅ Secure cleaning completed: {report.total_fixes} fixes applied")
        print(f"   Quality score: {report.quality_score:.2%}")
        print(f"   Processing time: {report.processing_time:.3f}s")
        
        # Verify security manager was used
        if hasattr(secure_cleaner, 'security_manager'):
            print("✅ Security manager integrated correctly")
        else:
            print("❌ Security manager not found")
            return False
        
    except Exception as e:
        print(f"❌ Secure cleaner test failed: {e}")
        return False
    
    return True

def test_error_handling():
    """Test comprehensive error handling."""
    print("\n⚠️ Testing error handling...")
    
    # Test with extremely large data (should be caught by security)
    try:
        large_data = {f'col_{i}': [f'value_{j}' for j in range(100)] for i in range(50)}
        large_df = pd.DataFrame(large_data)
        
        security_config = SecurityConfig(max_rows=10)  # Very restrictive
        secure_cleaner = create_secure_cleaner(
            config=security_config,
            llm_provider="local"
        )
        
        cleaned_df, report = secure_cleaner.clean(large_df)
        print("❌ Should have failed with large dataset!")
        return False
        
    except Exception as e:
        if "exceeds maximum" in str(e):
            print("✅ Large dataset correctly rejected by security")
        else:
            print(f"❌ Unexpected error: {e}")
            return False
    
    # Test with malformed data
    try:
        malformed_data = pd.DataFrame({'test': [None, None, None]})
        cleaner = TableCleaner(llm_provider="local")
        cleaned_df, report = cleaner.clean(malformed_data)
        print(f"✅ Handled malformed data gracefully: {report.total_fixes} fixes")
    except Exception as e:
        print(f"❌ Failed to handle malformed data: {e}")
        return False
    
    return True

def main():
    """Run all robustness tests."""
    print("🛡️ LLM Tab Cleaner Robustness Test Suite")
    print("="*50)
    
    # Configure logging to reduce noise
    logging.getLogger().setLevel(logging.WARNING)
    
    tests = [
        ("Security Validation", test_security_validation),
        ("Monitoring System", test_monitoring_system),
        ("Resilience Features", test_resilience_features),
        ("Secure Cleaner Integration", test_secure_cleaner_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name}...")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 TEST RESULTS SUMMARY")
    print("="*30)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 All robustness tests passed! System is robust and reliable.")
        return True
    else:
        print(f"\n⚠️ {total-passed} tests failed. Robustness needs improvement.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)