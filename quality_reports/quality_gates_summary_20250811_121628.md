# Quality Gates Report

**Project:** llm-tab-cleaner v0.3.0  
**Generated:** 2025-08-11 12:16:28  
**Overall Result:** ❌ FAILED

## Summary

- **Gates Passed:** 5/6
- **Average Score:** 8.2/10

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
Output:    16  77.42%   56->62, 120->110, 151-163, 167-168, 175->177, 186->185, 208->207, 242->241, 248->247, 259-260, 269, 271, 275, 283, 299, 303-307, 317-319, 354->357, 395-405, 418, 443, 447-468, 492-499, 503-506
---------------------------------------------------------------------------------------
TOTAL                                       5577   4643   1804     30  13.71%
Coverage JSON written to file coverage.json
============================== 21 passed in 8.45s ==============================

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
**Score:** 7.0/10

**Details:**
```
✅ Performance test results:
Size 100: 1.133s, 88.2 rows/sec
Size 500: 1.119s, 446.8 rows/sec
Size 1000: 1.131s, 884.3 rows/sec
Average throughput: 473.1 rows/sec
Max time per 1000 rows: 11.332s
PERFORMANCE: ACCEPTABLE

```

### Code Quality
**Status:** ✅ PASSED  
**Score:** 10.0/10

**Details:**
```
✅ Code quality assessment:
✅ __init__.py has docstrings
✅ adaptive.py has docstrings
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
✅ security.py has docstrings
✅ spark.py has docstrings
✅ streaming.py has docstrings
✅ validation.py has docstrings
✅ advanced_security.py has docstrings
✅ research.py has docstrings
✅ auto_scaling.py has docstrings
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

