# 🔧 LLM Tab Cleaner - Enhanced API Reference

**Version:** 0.3.0  
**Last Updated:** August 10, 2025  
**Coverage:** Production-Ready Features

---

## 📚 Table of Contents

1. [Core Cleaning API](#core-cleaning-api)
2. [Adaptive Learning](#adaptive-learning)
3. [Streaming Processing](#streaming-processing)  
4. [Distributed Computing](#distributed-computing)
5. [Caching Systems](#caching-systems)
6. [Security & Validation](#security--validation)
7. [Health & Monitoring](#health--monitoring)
8. [Performance Optimization](#performance-optimization)
9. [Backup & Recovery](#backup--recovery)
10. [Deployment APIs](#deployment-apis)

---

## 🧹 Core Cleaning API

### TableCleaner

**Main interface for LLM-powered table cleaning with enhanced capabilities.**

```python
from llm_tab_cleaner import TableCleaner

cleaner = TableCleaner(
    llm_provider="anthropic",           # LLM provider: "anthropic", "openai", "local"
    confidence_threshold=0.85,          # Minimum confidence for applying fixes
    rules=None,                        # Custom cleaning rules
    enable_profiling=True,             # Data profiling for quality assessment
    max_fixes_per_column=1000,         # Maximum fixes per column
    enable_monitoring=True,            # Performance monitoring
    max_concurrent_operations=8,       # Concurrent operations limit
    circuit_breaker_config=None,      # Circuit breaker configuration
    enable_security=True,              # Security validation and sanitization
    enable_backup=True,                # Automatic backup before operations
    backup_dir="./backups",           # Backup storage directory
    **provider_kwargs                 # Additional LLM provider arguments
)
```

#### Methods

##### `clean(df, columns=None, sample_rate=1.0) → tuple[pd.DataFrame, CleaningReport]`

Clean a pandas DataFrame with comprehensive security and backup.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame to clean
- `columns` (List[str], optional): Specific columns to clean
- `sample_rate` (float): Fraction of data to process (0.0-1.0)

**Returns:**
- `cleaned_df` (pd.DataFrame): Cleaned DataFrame
- `report` (CleaningReport): Detailed cleaning report

**Enhanced Features:**
- ✅ Automatic input validation and sanitization
- ✅ Pre-processing backup creation with restore capability
- ✅ Health monitoring with system resource checks
- ✅ Circuit breaker protection against failures
- ✅ Comprehensive audit trails

**Example:**
```python
import pandas as pd
from llm_tab_cleaner import TableCleaner

# Sample messy data
df = pd.DataFrame({
    'name': ['John Doe', 'N/A', 'Jane Smith', ''],
    'email': ['john@test.com', 'invalid-email', 'jane@test.com', 'missing'],
    'age': [25, -1, 30, 999],
    'salary': ['$50,000', 'N/A', '75000', 'unknown']
})

# Initialize with enhanced features
cleaner = TableCleaner(
    llm_provider="anthropic",
    confidence_threshold=0.80,
    enable_security=True,
    enable_backup=True
)

# Clean with comprehensive protection
cleaned_df, report = cleaner.clean(df)

print(f"Applied {report.total_fixes} fixes")
print(f"Quality score: {report.quality_score:.2%}")
print(f"Processing time: {report.processing_time:.2f}s")

# Access backup information
if hasattr(report, 'backup_id'):
    print(f"Backup created: {report.backup_id}")
```

### CleaningReport

**Comprehensive report of cleaning operations.**

```python
@dataclass
class CleaningReport:
    total_fixes: int                    # Number of fixes applied
    quality_score: float                # Overall quality improvement score
    fixes: List[Fix]                   # Detailed list of all fixes
    processing_time: float             # Total processing time in seconds
    profile_summary: Dict[str, Any]    # Data profiling summary
    audit_trail: List[Dict[str, Any]]  # Complete audit trail
```

### Fix

**Individual data fix record with enhanced metadata.**

```python
@dataclass
class Fix:
    column: str                        # Column name where fix was applied
    row_index: int                     # Row index of the fix
    original: Any                      # Original value before cleaning
    cleaned: Any                       # Cleaned value after processing
    confidence: float                  # Confidence score (0.0-1.0)
    reasoning: str                     # Explanation of the fix
    rule_applied: str                  # Rule or method used for cleaning
```

---

## 🤖 Adaptive Learning

### AdaptiveCache

**Intelligent caching system that learns from usage patterns.**

```python
from llm_tab_cleaner.adaptive import AdaptiveCache

cache = AdaptiveCache(
    max_size=10000,                    # Maximum cached entries
    ttl=3600                          # Time-to-live in seconds
)

# Store cleaning result
cache.put(
    value="N/A",
    column="status", 
    context={"data_type": "string"},
    output_value="Unknown",
    confidence=0.95
)

# Retrieve cached result
result = cache.get("N/A", "status", {"data_type": "string"})
if result:
    cleaned_value, confidence = result
    print(f"Cached: {cleaned_value} (confidence: {confidence:.2f})")

# Get performance statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
print(f"Total entries: {stats['cache_size']}")
```

### PatternLearner

**Learns cleaning patterns from successful operations.**

```python
from llm_tab_cleaner.adaptive import PatternLearner
from llm_tab_cleaner import Fix

learner = PatternLearner(max_patterns=2000)

# Learn from successful fix
fix = Fix(
    column="state",
    row_index=0, 
    original="calif",
    cleaned="CA",
    confidence=0.90,
    reasoning="State standardization",
    rule_applied="state_codes"
)

learner.learn_from_fix(fix, {"data_type": "string"})

# Get suggestion for similar value
suggestion = learner.suggest_fix("texas", "state", {"data_type": "string"})
if suggestion:
    suggested_value, confidence = suggestion
    print(f"Suggestion: {suggested_value} (confidence: {confidence:.2f})")

# View learning statistics
stats = learner.get_stats()
print(f"Patterns learned: {stats['pattern_count']}")
print(f"Average confidence: {stats['average_confidence']:.2f}")
```

### AutoScalingProcessor

**Automatically scales processing based on workload.**

```python
from llm_tab_cleaner.adaptive import AutoScalingProcessor

processor = AutoScalingProcessor(
    initial_batch_size=100,
    max_batch_size=5000
)

# Process items with automatic batch optimization
def clean_batch(items):
    # Your batch processing logic
    return [f"cleaned_{item}" for item in items]

items = list(range(10000))
results = processor.process_batch(items, clean_batch)

# Check performance statistics  
stats = processor.get_stats()
print(f"Optimal batch size: {stats['current_batch_size']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average throughput: {stats['average_throughput']:.1f} items/sec")
```

---

## 🌊 Streaming Processing

### StreamingCleaner

**Real-time streaming data cleaner with adaptive capabilities.**

```python
import asyncio
from llm_tab_cleaner.streaming import StreamingCleaner, StreamRecord
from llm_tab_cleaner import TableCleaner

# Initialize base cleaner
base_cleaner = TableCleaner(
    llm_provider="local",
    confidence_threshold=0.75
)

# Create streaming cleaner
streaming_cleaner = StreamingCleaner(
    base_cleaner=base_cleaner,
    batch_size=1000,                   # Records per batch
    batch_timeout=5.0,                 # Batch timeout in seconds
    max_queue_size=10000,              # Maximum queue size
    enable_adaptive=True,              # Enable adaptive features
    checkpoint_interval=100            # Batches between checkpoints
)

async def process_stream():
    # Generate sample stream
    async def generate_records():
        for i in range(5000):
            record = StreamRecord(
                id=f"record_{i}",
                data={
                    "name": f"User {i}" if i % 10 != 0 else "N/A",
                    "value": i if i % 20 != 0 else "invalid"
                },
                timestamp=time.time()
            )
            yield record
            await asyncio.sleep(0.001)  # Simulate real-time arrival
    
    # Process streaming data
    async for cleaned_record in streaming_cleaner.clean_batch_stream(generate_records()):
        print(f"Cleaned: {cleaned_record.id}")
    
    # Get processing statistics
    stats = streaming_cleaner.get_stats()
    print(f"Throughput: {stats['throughput']:.1f} records/sec")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Pattern matches: {stats['pattern_matches']}")

# Run streaming example
asyncio.run(process_stream())
```

### StreamRecord

**Individual record in a data stream.**

```python
from llm_tab_cleaner.streaming import StreamRecord

record = StreamRecord(
    id="unique_record_id",             # Unique identifier
    data={"column1": "value1"},        # Record data as dictionary
    timestamp=time.time(),             # Record timestamp
    metadata={"source": "api"}         # Optional metadata
)
```

---

## ⚡ Distributed Computing

### DistributedCleaner

**Distributed data cleaning with auto-scaling capabilities.**

```python
from llm_tab_cleaner.distributed import DistributedCleaner
import pandas as pd

# Base cleaner configuration
base_config = {
    'llm_provider': 'local',
    'confidence_threshold': 0.80,
    'enable_security': True,
    'enable_backup': True
}

# Initialize distributed cleaner
distributed_cleaner = DistributedCleaner(
    base_cleaner_config=base_config,
    max_workers=8,                     # Maximum worker processes/threads
    chunk_size=5000,                   # Records per chunk
    enable_process_pool=True,          # Use multiprocessing
    load_balancer_strategy="adaptive"  # Load balancing strategy
)

# Large dataset for distributed processing
large_df = pd.DataFrame({
    'id': range(100000),
    'category': ['A', 'B', 'C', 'N/A'] * 25000,
    'value': range(100000)
})

# Process with distributed computing
report = distributed_cleaner.clean_distributed(
    large_df,
    columns=['category'],
    sample_rate=0.1,                   # Process 10% sample
    priority=1                         # Task priority
)

print(f"Distributed processing: {report.total_fixes} fixes")
print(f"Processing time: {report.processing_time:.2f}s")

# Get processing statistics
stats = distributed_cleaner.get_processing_stats()
print(f"Workers used: {stats['max_workers']}")
print(f"Average throughput: {stats['average_throughput']:.1f} records/sec")
print(f"Worker utilization: {stats['worker_utilization']:.1%}")
```

### LoadBalancer

**Intelligent load balancer for distributed processing.**

```python
from llm_tab_cleaner.distributed import LoadBalancer, ProcessingNode

# Initialize load balancer
balancer = LoadBalancer(strategy="adaptive")

# Register processing nodes
node1 = ProcessingNode(
    node_id="worker-1",
    capacity=100,
    current_load=25,
    last_heartbeat=time.time(),
    status="active",
    performance_metrics={
        "cpu_usage": 0.3,
        "memory_usage": 0.4,
        "throughput": 1200
    }
)

balancer.register_node(node1)

# Get cluster status
status = balancer.get_cluster_status()
print(f"Active nodes: {status['active_nodes']}")
print(f"Total capacity: {status['total_capacity']}")
print(f"Utilization: {status['utilization']:.1%}")
```

### AutoScaler

**Automatic scaling based on workload and system resources.**

```python
from llm_tab_cleaner.distributed import AutoScaler

scaler = AutoScaler(
    min_workers=2,                     # Minimum workers
    max_workers=16,                    # Maximum workers  
    target_utilization=0.70            # Target CPU utilization
)

# Get scaling recommendation
metrics = {
    "cpu_utilization": 0.85,
    "memory_utilization": 0.70,
    "queue_length": 150
}

recommendation = scaler.recommend_scaling(metrics)
print(f"Recommendation: {recommendation['action']}")
print(f"Current workers: {recommendation['current_workers']}")
print(f"Recommended workers: {recommendation['recommended_workers']}")
print(f"Reason: {recommendation['reason']}")

# Apply scaling decision
if scaler.apply_scaling(recommendation):
    print("Scaling applied successfully")

# Get scaling statistics
stats = scaler.get_scaling_stats()
print(f"Scale up events: {stats['scale_ups']}")
print(f"Scale down events: {stats['scale_downs']}")
```

---

## 💾 Caching Systems

### MultiLevelCache

**Multi-level cache with different storage tiers.**

```python
from llm_tab_cleaner.caching import MultiLevelCache

cache = MultiLevelCache(
    l1_size=100,                       # L1 cache size (fast, in-memory)
    l2_size=500,                       # L2 cache size (compressed)
    l3_size=2000,                      # L3 cache size (disk-based)
    enable_disk_cache=True,            # Enable disk cache
    cache_dir="./cache"                # Cache directory
)

# Store value (automatically promotes through levels)
cache.put("data_key", "processed_value")

# Retrieve value (promotes from lower levels)
value = cache.get("data_key")
print(f"Retrieved: {value}")

# Get comprehensive statistics
stats = cache.get_stats()
print(f"L1 hit rate: {stats['hit_rates']['l1_rate']:.1%}")
print(f"L2 hit rate: {stats['hit_rates']['l2_rate']:.1%}")
print(f"L3 hit rate: {stats['hit_rates']['l3_rate']:.1%}")
print(f"Overall hit rate: {stats['hit_rates']['overall_hit_rate']:.1%}")
print(f"Disk usage: {stats['cache_levels']['l3']['disk_usage_mb']:.1f} MB")
```

### CompressedCache

**Cache with automatic compression for large values.**

```python
from llm_tab_cleaner.caching import CompressedCache

cache = CompressedCache(
    max_size=500,                      # Maximum entries
    compression_threshold=1024         # Compress values > 1KB
)

# Store large value (automatically compressed)
large_data = "x" * 10000  # 10KB string
cache.put("large_key", large_data)

# Retrieve (automatically decompressed)
retrieved = cache.get("large_key")
print(f"Original size: {len(large_data)} bytes")

# Check compression statistics
stats = cache.get_stats()
compression = stats["compression"]
print(f"Compressed entries: {compression['compressed_entries']}")
print(f"Compression ratio: {compression['compression_ratio']:.1%}")
print(f"Space saved: {compression['total_uncompressed_size'] - compression['total_compressed_size']} bytes")
```

### CacheKeyGenerator

**Generate consistent cache keys for complex objects.**

```python
from llm_tab_cleaner.caching import CacheKeyGenerator
import pandas as pd

# Generate keys for different data types
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
params = {'threshold': 0.8, 'method': 'llm'}

key_gen = CacheKeyGenerator()

# Generate consistent key
cache_key = key_gen.generate_key(df, params)
print(f"Cache key: {cache_key}")

# Same inputs always generate same key
same_key = key_gen.generate_key(df, params)
assert cache_key == same_key  # True
```

---

## 🔒 Security & Validation

### InputValidator

**Comprehensive input validation and security checking.**

```python
from llm_tab_cleaner.validation import InputValidator
import pandas as pd

validator = InputValidator()

# Validate DataFrame
df = pd.DataFrame({
    'safe_column': ['value1', 'value2'],
    'suspicious': ['<script>alert("xss")</script>', 'DROP TABLE users;']
})

result = validator.validate_dataframe(df)
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")

# Validate column names
column_result = validator.validate_column_name("user_email")
print(f"Column name valid: {column_result.is_valid}")

# Validate individual values
value_result = validator.validate_value("../../../etc/passwd")
print(f"Value safe: {value_result.is_valid}")
print(f"Sanitized: {value_result.sanitized_value}")

# Validate file paths
path_result = validator.validate_file_path("/safe/data/file.csv")
print(f"Path secure: {path_result.is_valid}")

# Validate configuration
config = {
    'llm_provider': 'anthropic',
    'confidence_threshold': 0.85,
    'batch_size': 1000
}

config_result = validator.validate_cleaning_config(config)
print(f"Configuration valid: {config_result.is_valid}")
```

### DataSanitizer

**Sanitizes data for safe processing.**

```python
from llm_tab_cleaner.validation import DataSanitizer
import pandas as pd

sanitizer = DataSanitizer()

# Sanitize DataFrame
messy_df = pd.DataFrame({
    'html_content': ['<b>Bold</b>', '<script>evil()</script>'],
    'sql_like': ['SELECT * FROM users', 'normal data'],
    'column with spaces!': ['data1', 'data2']
})

sanitized_df, warnings = sanitizer.sanitize_dataframe(messy_df)
print(f"Warnings: {warnings}")
print("Sanitized DataFrame:")
print(sanitized_df.head())

# Sanitize configuration
unsafe_config = {
    'data_path': '../../../secret/data.csv',
    'threshold': '<script>',
    'batch_size': 1000
}

clean_config, warnings = sanitizer.sanitize_config(unsafe_config)
print(f"Clean config: {clean_config}")
print(f"Config warnings: {warnings}")
```

---

## 🏥 Health & Monitoring  

### HealthMonitor

**Comprehensive health monitoring system.**

```python
import asyncio
from llm_tab_cleaner.health import HealthMonitor

# Initialize health monitor
monitor = HealthMonitor(check_interval=30.0)

# Register custom health check
def database_check():
    """Check database connection."""
    try:
        # Simulate database check
        return "healthy", "Database connection active", {"connections": 10}
    except Exception as e:
        return "unhealthy", f"Database error: {e}", {}

monitor.register_check("database", database_check)

# Run health checks
async def check_health():
    health_status = await monitor.run_checks()
    
    print(f"Overall status: {health_status.status}")
    print(f"Uptime: {health_status.uptime:.1f} seconds")
    
    for check in health_status.checks:
        print(f"- {check.name}: {check.status} ({check.message})")

# Start continuous monitoring
async def start_monitoring():
    await monitor.start_monitoring()

asyncio.run(check_health())
```

### AlertManager

**Manages health alerts and notifications.**

```python
from llm_tab_cleaner.health import AlertManager, HealthCheck

alert_manager = AlertManager()

# Add custom alert handler
def email_alert_handler(health_check):
    """Send email alert for critical issues."""
    if health_check.status == "unhealthy":
        print(f"🚨 CRITICAL ALERT: {health_check.name} - {health_check.message}")
        # Send email notification here

alert_manager.add_handler(email_alert_handler)

# Simulate health check
check = HealthCheck(
    name="memory_usage",
    status="unhealthy", 
    message="Memory usage 95%",
    timestamp=time.time(),
    duration=0.1,
    metadata={"usage_percent": 95}
)

# Process alert (triggers handlers)
alert_manager.process_health_check(check)

# Get recent alerts
recent_alerts = alert_manager.get_recent_alerts(hours=24)
print(f"Alerts in last 24h: {len(recent_alerts)}")
```

---

## ⚡ Performance Optimization

### ResourceMonitor

**Real-time system resource monitoring.**

```python
from llm_tab_cleaner.optimization import ResourceMonitor
import time

monitor = ResourceMonitor(monitoring_interval=1.0)

# Start monitoring
monitor.start_monitoring()

# Add threshold callback
def resource_alert(violation_type, metrics):
    """Alert on resource threshold violations."""
    print(f"🚨 {violation_type}: CPU {metrics.cpu_percent}%, Memory {metrics.memory_percent}%")

monitor.add_threshold_callback(resource_alert)

# Get current metrics
time.sleep(2)  # Let monitoring collect data
current = monitor.get_current_metrics()

if current:
    print(f"CPU: {current.cpu_percent:.1f}%")
    print(f"Memory: {current.memory_percent:.1f}%")
    print(f"Disk I/O: {current.disk_io_read_mb:.1f} MB read, {current.disk_io_write_mb:.1f} MB write")

# Get average metrics
avg_metrics = monitor.get_average_metrics(minutes=5)
if avg_metrics:
    print(f"5-min average CPU: {avg_metrics['cpu_percent']:.1f}%")
    print(f"5-min average Memory: {avg_metrics['memory_percent']:.1f}%")

# Stop monitoring
monitor.stop_monitoring()
```

### PerformanceOptimizer

**Automatic performance optimization.**

```python
from llm_tab_cleaner.optimization import PerformanceOptimizer, ResourceMonitor

monitor = ResourceMonitor()
monitor.start_monitoring()

optimizer = PerformanceOptimizer(monitor)

# Trigger automatic optimization
time.sleep(1)  # Let monitor collect baseline
optimizations = optimizer.auto_optimize(trigger_threshold=80.0)

for opt in optimizations:
    print(f"Applied {opt.optimization_type}: {opt.improvement_percent:+.1f}% improvement")
    print(f"Description: {opt.description}")

# Apply specific optimization
result = optimizer.optimize_specific("garbage_collection")
if result:
    print(f"Garbage collection: {result.improvement_percent:+.1f}% improvement")

# Get optimization history
stats = optimizer.get_optimization_stats()
print(f"Total optimizations: {stats['total_optimizations']}")
print(f"Average improvement: {stats['average_improvement_percent']:+.1f}%")

monitor.stop_monitoring()
```

### DataFrameOptimizer

**Specialized DataFrame optimization utilities.**

```python
from llm_tab_cleaner.optimization import DataFrameOptimizer
import pandas as pd
import numpy as np

# Create DataFrame with inefficient dtypes
df = pd.DataFrame({
    'id': range(1000),                 # int64 by default
    'category': ['A', 'B', 'C'] * 334, # object type
    'flag': [True, False] * 500,       # bool
    'small_num': np.random.randint(0, 100, 1000),  # int64
    'description': ['desc'] * 1000     # object type
})

print(f"Original memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize dtypes
optimized_df, stats = DataFrameOptimizer.optimize_dtypes(df, aggressive=True)

print(f"Optimized memory usage: {stats['memory_after_mb']:.2f} MB")
print(f"Memory saved: {stats['memory_saved_mb']:.2f} MB ({stats['memory_saved_percent']:.1f}%)")
print(f"Columns optimized: {stats['columns_optimized']}")

for optimization in stats['optimizations']:
    print(f"  - {optimization}")

# Get chunking strategy for large DataFrames
chunking_advice = DataFrameOptimizer.suggest_chunking_strategy(
    df, 
    memory_limit_mb=100
)

if chunking_advice['chunking_needed']:
    print(f"Recommended chunks: {chunking_advice['suggested_chunks']}")
    print(f"Chunk size: {chunking_advice['chunk_size']}")
    print(f"Code example: {chunking_advice['chunking_code']}")
```

---

## 💾 Backup & Recovery

### DataBackupManager

**Manages data backups and recovery operations.**

```python
from llm_tab_cleaner.backup import DataBackupManager
import pandas as pd

# Initialize backup manager
backup_manager = DataBackupManager(backup_dir="./backups")

# Create test data
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'SF', 'LA']
})

# Create backup
backup_id = backup_manager.create_backup(
    data=df,
    description="Customer data before cleaning",
    format="parquet"
)
print(f"Backup created: {backup_id}")

# List all backups
backups = backup_manager.list_backups()
for backup in backups[:5]:  # Show first 5
    print(f"- {backup.backup_id}: {backup.description} ({backup.size_bytes} bytes)")

# Create restore point
restore_point_id = backup_manager.create_restore_point(
    backup_id=backup_id,
    operation_type="data_cleaning",
    affected_records=len(df),
    metadata={"confidence_threshold": 0.85}
)
print(f"Restore point: {restore_point_id}")

# Restore from backup
restored_df = backup_manager.restore_backup(backup_id)
print(f"Restored {len(restored_df)} records")

# Get backup statistics
stats = backup_manager.get_backup_stats()
print(f"Total backups: {stats['total_backups']}")
print(f"Total size: {stats['total_size_mb']:.2f} MB")
print(f"Oldest backup: {stats['oldest_backup']['age_hours']:.1f} hours old")
```

### AutoBackupWrapper

**Wrapper that automatically creates backups before operations.**

```python
from llm_tab_cleaner.backup import DataBackupManager, AutoBackupWrapper
import pandas as pd

backup_manager = DataBackupManager()
auto_backup = AutoBackupWrapper(backup_manager)

# Enable automatic backups
auto_backup.enable()

# This will automatically create a backup
df = pd.DataFrame({'data': [1, 2, 3]})
backup_id = auto_backup.backup_before_operation(df, "preprocessing")

if backup_id:
    print(f"Auto-backup created: {backup_id}")
    
    # Perform your operation here
    # If something goes wrong, you can restore:
    # restored_df = backup_manager.restore_backup(backup_id)

# Disable auto-backup if needed
auto_backup.disable()
```

---

## 🚀 Global Access Functions

### Global Instance Access

```python
# Get global instances with default configurations
from llm_tab_cleaner.caching import get_global_cache
from llm_tab_cleaner.optimization import get_global_resource_monitor
from llm_tab_cleaner.health import get_global_health_monitor
from llm_tab_cleaner.backup import get_global_backup_manager

# Global cache (multi-level by default)
cache = get_global_cache()
cache.put("key", "value")

# Global resource monitoring
monitor = get_global_resource_monitor()  # Auto-starts monitoring
current_metrics = monitor.get_current_metrics()

# Global health monitoring
health_monitor = get_global_health_monitor()
health_status = await health_monitor.run_checks()

# Global backup manager
backup_mgr = get_global_backup_manager()
backup_id = backup_mgr.create_backup(df, "Global backup")
```

### Version Information

```python
import llm_tab_cleaner

# Get version and feature info
info = llm_tab_cleaner.get_version_info()
print(f"Version: {info['version']}")

# Check available features
features = info['features']
for feature, available in features.items():
    status = "✅" if available else "❌"
    print(f"{status} {feature}: {available}")
```

---

## 🛠️ Advanced Configuration Examples

### Production Configuration

```python
from llm_tab_cleaner import TableCleaner
from llm_tab_cleaner.adaptive import AdaptiveCache, PatternLearner
from llm_tab_cleaner.distributed import DistributedCleaner

# Production-ready configuration
production_config = {
    'llm_provider': 'anthropic',
    'confidence_threshold': 0.85,
    'enable_profiling': True,
    'enable_monitoring': True,
    'enable_security': True,
    'enable_backup': True,
    'max_concurrent_operations': 16,
    'circuit_breaker_config': {
        'failure_threshold': 5,
        'timeout': 60
    }
}

# Initialize production cleaner
cleaner = TableCleaner(**production_config)

# With distributed processing
distributed_config = {
    **production_config,
    'enable_security': True,  # Enhanced security for distributed
    'enable_backup': True     # Backup before distributed ops
}

distributed_cleaner = DistributedCleaner(
    base_cleaner_config=distributed_config,
    max_workers=32,
    chunk_size=10000,
    enable_process_pool=True,
    load_balancer_strategy="adaptive"
)
```

### Development Configuration

```python
# Development/testing configuration
dev_config = {
    'llm_provider': 'local',
    'confidence_threshold': 0.70,
    'enable_profiling': True,
    'enable_monitoring': False,  # Disable for faster testing
    'enable_security': False,    # Skip security for dev data
    'enable_backup': False,      # Skip backup for testing
    'max_concurrent_operations': 4
}

dev_cleaner = TableCleaner(**dev_config)
```

---

## 📊 Error Handling & Best Practices

### Exception Handling

```python
import pandas as pd
from llm_tab_cleaner import TableCleaner
from llm_tab_cleaner.validation import ValidationError

try:
    df = pd.DataFrame({'col': ['data']})
    cleaner = TableCleaner(enable_security=True)
    
    cleaned_df, report = cleaner.clean(df)
    print(f"Success: {report.total_fixes} fixes applied")
    
except ValidationError as e:
    print(f"Validation error: {e}")
    # Handle validation issues
    
except MemoryError as e:
    print(f"Memory error: {e}")
    # Use chunking or distributed processing
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Check logs and health status
```

### Performance Best Practices

```python
# For large datasets
if len(df) > 100000:
    # Use distributed processing
    distributed_cleaner = DistributedCleaner(
        base_cleaner_config=config,
        max_workers=8,
        chunk_size=10000
    )
    report = distributed_cleaner.clean_distributed(df, sample_rate=0.1)

# For real-time processing
elif streaming_data:
    # Use streaming cleaner
    streaming_cleaner = StreamingCleaner(base_cleaner)
    # Process with async iterator

# For repeated operations
else:
    # Use caching and pattern learning
    cleaner = TableCleaner(enable_monitoring=True)
    # Cache will improve performance over time
```

---

## 🔗 Integration Examples

### Apache Airflow Integration

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from llm_tab_cleaner import TableCleaner
import pandas as pd

def clean_data_task(**context):
    """Airflow task for data cleaning."""
    df = pd.read_csv(context['params']['input_file'])
    
    cleaner = TableCleaner(
        llm_provider='anthropic',
        enable_backup=True,
        backup_dir=context['params']['backup_dir']
    )
    
    cleaned_df, report = cleaner.clean(df)
    cleaned_df.to_csv(context['params']['output_file'], index=False)
    
    return {
        'total_fixes': report.total_fixes,
        'quality_score': report.quality_score,
        'processing_time': report.processing_time
    }

# DAG definition
dag = DAG('data_cleaning_pipeline', ...)

clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data_task,
    params={
        'input_file': '/data/raw/customers.csv',
        'output_file': '/data/clean/customers.csv', 
        'backup_dir': '/data/backups'
    },
    dag=dag
)
```

### FastAPI Web Service

```python
from fastapi import FastAPI, HTTPException
from llm_tab_cleaner import TableCleaner
import pandas as pd
import io

app = FastAPI(title="LLM Tab Cleaner API", version="0.3.0")

# Global cleaner instance
cleaner = TableCleaner(
    llm_provider="anthropic",
    enable_monitoring=True,
    enable_security=True
)

@app.post("/clean")
async def clean_data(data: dict):
    """Clean data via REST API."""
    try:
        df = pd.DataFrame(data['data'])
        
        cleaned_df, report = cleaner.clean(
            df, 
            columns=data.get('columns'),
            sample_rate=data.get('sample_rate', 1.0)
        )
        
        return {
            'cleaned_data': cleaned_df.to_dict('records'),
            'report': {
                'total_fixes': report.total_fixes,
                'quality_score': report.quality_score,
                'processing_time': report.processing_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from llm_tab_cleaner.health import get_global_health_monitor
    
    monitor = get_global_health_monitor()
    health_status = await monitor.run_checks()
    
    return {
        'status': health_status.status,
        'uptime': health_status.uptime,
        'checks': len(health_status.checks)
    }
```

---

This enhanced API reference provides comprehensive coverage of all production-ready features in LLM Tab Cleaner v0.3.0, including adaptive learning, streaming processing, distributed computing, advanced caching, security validation, health monitoring, and performance optimization capabilities.