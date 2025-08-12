# Quality Gates Report

**Project:** llm-tab-cleaner v0.3.0  
**Generated:** 2025-08-12 23:17:02  
**Overall Result:** ❌ FAILED

## Summary

- **Gates Passed:** 5/6
- **Average Score:** 8.7/10

## Gate Details

### Code Execution
**Status:** ✅ PASSED  
**Score:** 10.0/10

**Details:**
```
✅ Code imports and executes successfully
Basic functionality test passed: 1 rows processed
Quality score: 1.0

```

### Test Coverage
**Status:** ❌ FAILED  
**Score:** 1.4/10

**Details:**
```
❌ Issues: Coverage 13.7% < 75%
Output: t(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_auto_scaling.py:8: in <module>
    from llm_tab_cleaner.auto_scaling import (
src/llm_tab_cleaner/auto_scaling.py:12: in <module>
    import psutil
E   ModuleNotFoundError: No module named 'psutil'
=========================== short test summary info ============================
ERROR tests/test_auto_scaling.py
=============================== 1 error in 0.35s ===============================

```

### Security Scan
**Status:** ✅ PASSED  
**Score:** 10.0/10

**Details:**
```
✅ No security vulnerabilities found
Security features: 3
✅ advanced_security.py implements security controls
✅ validation.py implements security controls
✅ security.py implements security controls
```

### Performance Benchmarks
**Status:** ✅ PASSED  
**Score:** 10.0/10

**Details:**
```
✅ Performance test results:
Size 100: 0.010s, 10379.4 rows/sec
Size 500: 0.006s, 78325.0 rows/sec
Size 1000: 0.008s, 127567.9 rows/sec
Average throughput: 72090.7 rows/sec
Max time per 1000 rows: 0.096s
PERFORMANCE: PASS

```

### Code Quality
**Status:** ✅ PASSED  
**Score:** 10.0/10

**Details:**
```
✅ Code quality assessment:
✅ __init__.py has docstrings
✅ adaptive.py has docstrings
✅ advanced_security.py has docstrings
✅ auto_scaling.py has docstrings
✅ backup.py has docstrings
✅ benchmarks.py has docstrings
✅ caching.py has docstrings
✅ cleaning_rule.py has docstrings
✅ cli.py has docstrings
✅ compliance.py has docstrings
✅ confidence.py has docstrings
✅ core.py has docstrings
✅ deployment.py has docstrings
✅ distributed.py has docstrings
✅ health.py has docstrings
✅ i18n.py has docstrings
✅ incremental.py has docstrings
✅ llm_providers.py has docstrings
✅ monitoring.py has docstrings
✅ optimization.py has docstrings
✅ profiler.py has docstrings
✅ research.py has docstrings
✅ security.py has docstrings
✅ spark.py has docstrings
✅ streaming.py has docstrings
✅ validation.py has docstrings
✅ Total Python files: 26
✅ Total lines of code: 12888
✅ Modular architecture with 26 modules
✅ Key components implemented: 5/5
```

### Documentation
**Status:** ✅ PASSED  
**Score:** 11.0/10

**Details:**
```
✅ Documentation score: 11.0/10
✅ README.md (9.6KB)
✅ CHANGELOG.md (2.7KB)
✅ ARCHITECTURE.md (9.5KB)
✅ API_REFERENCE.md (17.6KB)
✅ DEPLOYMENT_GUIDE.md (15.3KB)
✅ SECURITY.md (2.6KB)
✅ 18 Python files have detailed docstrings
```

