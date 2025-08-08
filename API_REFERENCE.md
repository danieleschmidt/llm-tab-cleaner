# 📚 LLM Tab Cleaner - Complete API Reference

## 🚀 Quick Start

```python
from llm_tab_cleaner import TableCleaner
import pandas as pd

# Initialize cleaner
cleaner = TableCleaner(llm_provider="local")

# Clean data
df = pd.DataFrame({'name': ['john doe', 'N/A'], 'email': ['JOHN@TEST.COM', 'invalid']})
cleaned_df, report = cleaner.clean(df)

print(f"Fixes applied: {report.total_fixes}")
print(f"Quality score: {report.quality_score:.2%}")
```

## 🏗️ Core Classes

### TableCleaner

Main class for data cleaning operations.

#### Constructor
```python
TableCleaner(
    llm_provider: str = "local",
    confidence_threshold: float = 0.85,
    rules: Optional[RuleSet] = None,
    enable_profiling: bool = True,
    max_fixes_per_column: int = 1000,
    parallel_processing: bool = False,
    cache_enabled: bool = True,
    **provider_kwargs
)
```

**Parameters:**
- `llm_provider`: LLM provider ("local", "anthropic", "openai")
- `confidence_threshold`: Minimum confidence for applying fixes (0.0-1.0)
- `rules`: Custom cleaning rules
- `enable_profiling`: Enable data profiling
- `max_fixes_per_column`: Maximum fixes per column
- `parallel_processing`: Enable parallel processing
- `cache_enabled`: Enable result caching
- `**provider_kwargs`: Additional provider arguments

#### Methods

##### `clean(df, columns=None, skip_profiling=False)`
Clean a pandas DataFrame.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame
- `columns` (List[str], optional): Specific columns to clean
- `skip_profiling` (bool): Skip data profiling step

**Returns:**
- `Tuple[pd.DataFrame, CleaningReport]`: Cleaned DataFrame and report

**Example:**
```python
cleaned_df, report = cleaner.clean(df, columns=['name', 'email'])
```

##### `profile_data(df)`
Profile data quality and structure.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to profile

**Returns:**
- `DataProfile`: Detailed data profile

**Example:**
```python
profile = cleaner.profile_data(df)
print(f"Quality score: {profile.overall_quality:.2%}")
```

##### `validate_data(df)`
Validate data against quality rules.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to validate

**Returns:**
- `ValidationReport`: Validation results

##### `get_cleaning_suggestions(df, column=None)`
Get cleaning suggestions without applying fixes.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame  
- `column` (str, optional): Specific column to analyze

**Returns:**
- `Dict[str, List[CleaningSuggestion]]`: Suggestions per column

### CleaningReport

Contains results of cleaning operation.

#### Attributes
```python
@dataclass
class CleaningReport:
    total_fixes: int                    # Number of fixes applied
    quality_score: float               # Overall quality (0.0-1.0)
    processing_time: float             # Processing time in seconds
    fixes: List[Fix]                   # List of specific fixes
    profile_summary: Dict[str, Any]    # Data profiling summary
    audit_trail: List[AuditEntry]      # Detailed audit trail
```

#### Methods

##### `to_dict()`
Convert report to dictionary.

##### `to_json(indent=2)`
Convert report to JSON string.

##### `save(file_path)`
Save report to file.

**Example:**
```python
report.save("cleaning_report.json")
```

### Fix

Represents a single data fix.

```python
@dataclass
class Fix:
    column: str                 # Column name
    row_index: int             # Row index
    original_value: Any        # Original value
    fixed_value: Any           # Fixed value
    confidence: float          # Fix confidence (0.0-1.0)
    reasoning: str             # Explanation
    rule_applied: Optional[str] # Rule that was applied
    timestamp: datetime        # When fix was applied
```

## 🔌 LLM Providers

### LocalProvider

Rule-based cleaning without external API calls.

```python
from llm_tab_cleaner.llm_providers import LocalProvider

provider = LocalProvider()
cleaned_value, confidence = provider.clean_value("  john doe  ", "name", {})
# Returns: ("john doe", 0.9)
```

### AnthropicProvider

Uses Anthropic's Claude models.

```python
from llm_tab_cleaner.llm_providers import AnthropicProvider

provider = AnthropicProvider(
    api_key="your_key",  # or use ANTHROPIC_API_KEY env var
    model="claude-3-haiku-20240307"
)

cleaned_value, confidence = provider.clean_value("joh smith", "name", {})
# Returns: ("John Smith", 0.95)
```

### OpenAIProvider

Uses OpenAI's GPT models.

```python
from llm_tab_cleaner.llm_providers import OpenAIProvider

provider = OpenAIProvider(
    api_key="your_key",  # or use OPENAI_API_KEY env var  
    model="gpt-3.5-turbo"
)

cleaned_value, confidence = provider.clean_value("JANE.DOE@GMAIL.COM", "email", {})
# Returns: ("jane.doe@gmail.com", 0.92)
```

### Provider Factory

```python
from llm_tab_cleaner.llm_providers import get_provider

# Get provider by name
provider = get_provider("anthropic", api_key="your_key", model="claude-3-sonnet")
provider = get_provider("openai", api_key="your_key", model="gpt-4")
provider = get_provider("local")  # No additional args needed
```

## 📏 Data Profiling

### DataProfiler

Analyzes data quality and structure.

```python
from llm_tab_cleaner.profiler import DataProfiler

profiler = DataProfiler()
profile = profiler.profile_dataframe(df)

print(f"Data quality: {profile.overall_quality:.2%}")
print(f"Completeness: {profile.completeness:.2%}")
print(f"Validity: {profile.validity:.2%}")
```

### DataProfile

Contains profiling results.

```python
@dataclass
class DataProfile:
    overall_quality: float              # Overall quality score
    completeness: float                 # Completeness score
    validity: float                     # Validity score  
    consistency: float                  # Consistency score
    uniqueness: float                   # Uniqueness score
    column_profiles: Dict[str, Any]     # Per-column profiles
    statistics: Dict[str, Any]          # Overall statistics
    issues: List[DataQualityIssue]      # Identified issues
    recommendations: List[str]          # Improvement recommendations
```

### ColumnProfile

Per-column analysis results.

```python
@dataclass
class ColumnProfile:
    column_name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    most_common_values: List[Tuple[Any, int]]
    data_type_consistency: float
    format_patterns: List[str]
    outliers: List[Any]
    quality_issues: List[str]
    quality_score: float
```

## 🎯 Cleaning Rules

### RuleSet

Container for cleaning rules.

```python
from llm_tab_cleaner.cleaning_rule import RuleSet, CleaningRule

# Create custom rule
rule = CleaningRule(
    name="standardize_phone",
    description="Standardize phone number format",
    pattern=r"[\(\)\-\s]",
    transform=lambda x: re.sub(r"[\(\)\-\s]", "", str(x)),
    confidence=0.95,
    column_patterns=["phone", "telephone", "mobile"],
    data_types=["object"]
)

# Create rule set
rules = RuleSet([rule])
cleaner = TableCleaner(rules=rules)
```

### CleaningRule

Individual cleaning rule.

```python
@dataclass
class CleaningRule:
    name: str
    description: str
    pattern: str                        # Regex pattern to match
    transform: Callable[[Any], Any]     # Transform function
    confidence: float                   # Rule confidence
    column_patterns: List[str]          # Column name patterns
    data_types: List[str]              # Applicable data types
    examples: List[Tuple[str, str]]    # Before/after examples
    conditions: List[Callable]         # Additional conditions
```

### Built-in Rules

```python
from llm_tab_cleaner import create_default_rules

rules = create_default_rules()
# Includes rules for:
# - Email standardization
# - Phone number cleaning  
# - Date format standardization
# - Text trimming and casing
# - Null value standardization
```

## 🛡️ Security & Compliance

### SecurityManager

Manages security and validation.

```python
from llm_tab_cleaner.security import SecurityManager, SecurityConfig

config = SecurityConfig(
    max_data_size=100_000_000,    # 100MB limit
    max_rows=1_000_000,           # 1M rows limit
    allow_sensitive_columns=False,
    enable_audit_logging=True
)

security_manager = SecurityManager(config)
validation_result = security_manager.validate_and_prepare_data(df, "cleaning_op")
```

### ComplianceManager

Handles regulatory compliance.

```python
from llm_tab_cleaner.compliance import ComplianceManager, create_gdpr_config

config = create_gdpr_config()
compliance_manager = ComplianceManager(config)

# Classify data
classifications = compliance_manager.classify_data(df)

# Record consent
from llm_tab_cleaner.compliance import ConsentRecord
consent = ConsentRecord(
    subject_id="user123",
    consent_type="processing",
    purpose="data_cleaning", 
    granted=True,
    timestamp=datetime.now(),
    expiry_date=datetime.now() + timedelta(days=365)
)
compliance_manager.record_consent(consent)
```

## 📊 Monitoring & Observability

### CleaningMonitor

Comprehensive monitoring system.

```python
from llm_tab_cleaner.monitoring import CleaningMonitor

monitor = CleaningMonitor()

# Start monitoring operation
context = monitor.start_operation_monitoring("op001", "data_cleaning")

# ... perform cleaning ...

# End monitoring
monitor.end_operation_monitoring(context, success=True, rows_processed=1000)

# Get health status
health = monitor.health.get_overall_health()
print(f"System status: {health['status']}")
```

### MetricsCollector

Collects performance metrics.

```python
from llm_tab_cleaner.monitoring import MetricsCollector

collector = MetricsCollector()

# Record metrics
collector.record_counter("operations_total", 1.0)
collector.record_gauge("memory_usage_mb", 512.0)
collector.record_histogram("processing_time_ms", 1500.0)

# Get metrics
summary = collector.get_metric_summary("processing_time_ms")
print(f"Average processing time: {summary['mean']:.2f}ms")
```

## ⚡ Optimization & Scaling

### OptimizationEngine

Performance optimization coordinator.

```python
from llm_tab_cleaner.optimization import OptimizationEngine, OptimizationConfig

config = OptimizationConfig(
    enable_caching=True,
    cache_type="redis",
    enable_parallel_processing=True,
    max_workers=8,
    enable_memory_optimization=True
)

optimizer = OptimizationEngine(config)

# Optimize DataFrame
optimized_df = optimizer.optimize_dataframe(df)

# Get recommendations
recommendations = optimizer.get_optimization_recommendations(df)
```

### CacheManager

Intelligent caching system.

```python
from llm_tab_cleaner.optimization import CacheManager, OptimizationConfig

config = OptimizationConfig(cache_type="memory", max_cache_size=1000)
cache = CacheManager(config)

# Manual cache operations
cache.set("key", "value", ttl=3600)
value = cache.get("key")

# Cache decorator
from llm_tab_cleaner.optimization import cached

@cached(cache)
def expensive_operation(data):
    # Heavy computation
    return processed_data
```

### ParallelProcessor

Parallel processing utilities.

```python
from llm_tab_cleaner.optimization import ParallelProcessor, OptimizationConfig

config = OptimizationConfig(max_workers=4, chunk_size=1000)
processor = ParallelProcessor(config)

# Split data into chunks
chunks = processor.split_dataframe(large_df)

# Process in parallel
results = processor.process_chunks_threaded(chunks, processing_function)

# Combine results
final_df, combined_report = processor.combine_results(results)
```

## 🌍 Internationalization

### I18n System

Multi-language support.

```python
from llm_tab_cleaner.i18n import setup_i18n, t, set_locale

# Setup i18n
setup_i18n(locale="en")

# Translate messages
message = t("cleaning.started")  # "Started data cleaning process"

# Change locale
set_locale("es")
message = t("cleaning.started")  # "Proceso de limpieza de datos iniciado"

# Format with variables
message = t("cleaning.fixes_applied", count=5)  # "5 fixes applied"
```

### Supported Locales

| Locale | Language | Coverage |
|--------|----------|----------|
| `en` | English | 100% (base) |
| `es` | Spanish | 80% |
| `fr` | French | 80% |
| `de` | German | 80% |
| `ja` | Japanese | 70% |
| `zh` | Chinese (Simplified) | 70% |

## 🔥 Spark Integration

### SparkCleaner

Distributed processing with Apache Spark.

```python
from llm_tab_cleaner.spark import SparkCleaner
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DataCleaning").getOrCreate()

cleaner = SparkCleaner(
    spark=spark,
    llm_provider="anthropic",
    batch_size=10000,
    parallelism=100
)

# Clean large distributed dataset
cleaned_df = cleaner.clean_distributed(
    spark_df,
    output_path="s3://bucket/cleaned-data/",
    audit_log="s3://bucket/audit-logs/"
)
```

### StreamingCleaner

Real-time data cleaning.

```python
from llm_tab_cleaner.spark import StreamingCleaner

stream_cleaner = StreamingCleaner(
    spark=spark,
    checkpoint_location="/tmp/streaming_checkpoint"
)

query = stream_cleaner.clean_stream(
    input_stream=kafka_stream,
    output_path="s3://bucket/cleaned-stream/",
    trigger_interval="10 seconds"
)

query.awaitTermination()
```

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

# Performance
LLM_CLEANER_MAX_WORKERS=8
LLM_CLEANER_CHUNK_SIZE=5000
LLM_CLEANER_CACHE_TYPE=redis
REDIS_URL=redis://localhost:6379

# Security
COMPLIANCE_REGIONS=eu,us
ENABLE_AUDIT_LOGGING=true
MAX_DATA_SIZE_MB=1000

# Monitoring  
ENABLE_MONITORING=true
METRICS_ENDPOINT=/metrics
HEALTH_ENDPOINT=/health
```

### Configuration Files

#### `.llm_cleaner_config.json`
```json
{
  "default_provider": "local",
  "confidence_threshold": 0.85,
  "enable_caching": true,
  "cache_type": "memory",
  "max_workers": 4,
  "compliance": {
    "regions": ["global"],
    "audit_logging": true
  },
  "monitoring": {
    "enabled": true,
    "sample_rate": 0.1
  }
}
```

## 🚨 Error Handling

### Exception Types

```python
from llm_tab_cleaner.exceptions import (
    CleaningError,           # General cleaning errors
    ValidationError,         # Data validation errors  
    SecurityException,       # Security violations
    ComplianceError,         # Compliance violations
    ProviderError,          # LLM provider errors
    OptimizationError       # Performance optimization errors
)

try:
    cleaned_df, report = cleaner.clean(df)
except SecurityException as e:
    print(f"Security violation: {e}")
except CleaningError as e:
    print(f"Cleaning failed: {e}")
```

### Error Recovery

```python
from llm_tab_cleaner.core import retry_with_backoff

@retry_with_backoff(max_retries=3, backoff_factor=2.0)
def robust_cleaning(df):
    return cleaner.clean(df)

try:
    result = robust_cleaning(df)
except Exception as e:
    print(f"Failed after retries: {e}")
```

## 📈 Performance Benchmarks

### Expected Performance

| Operation | Speed | Memory |
|-----------|--------|---------|
| Local Cleaning | 20,000+ rows/sec | <100MB |
| LLM Cleaning | 1,000+ rows/sec | <500MB |
| Spark Distributed | 100,000+ rows/sec | Scalable |
| Streaming | 10,000+ rows/sec | <200MB |

### Optimization Tips

1. **Use Local Provider** for simple rules-based cleaning
2. **Enable Caching** for repeated operations
3. **Batch Processing** for large datasets
4. **Parallel Processing** for CPU-intensive tasks
5. **Memory Optimization** for memory-constrained environments

## 🔗 Integration Examples

### REST API Integration
```python
from flask import Flask, request, jsonify
from llm_tab_cleaner import TableCleaner

app = Flask(__name__)
cleaner = TableCleaner()

@app.route('/clean', methods=['POST'])
def clean_data():
    data = request.get_json()
    df = pd.DataFrame(data)
    
    cleaned_df, report = cleaner.clean(df)
    
    return jsonify({
        'cleaned_data': cleaned_df.to_dict('records'),
        'report': report.to_dict()
    })
```

### Airflow DAG Integration
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from llm_tab_cleaner import TableCleaner

def clean_data_task(**context):
    cleaner = TableCleaner(llm_provider="anthropic")
    df = pd.read_csv(context['params']['input_file'])
    cleaned_df, report = cleaner.clean(df)
    cleaned_df.to_csv(context['params']['output_file'])
    return report.to_dict()

dag = DAG('data_cleaning', schedule_interval='@daily')

clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data_task,
    params={'input_file': 's3://raw/data.csv', 'output_file': 's3://clean/data.csv'}
)
```

## 🆘 Troubleshooting

### Common Issues

#### "No module named 'llm_tab_cleaner'"
```bash
pip install llm-tab-cleaner
# or
pip install -e .  # if installing from source
```

#### "API key not found"
```bash
export ANTHROPIC_API_KEY=your_key
# or
export OPENAI_API_KEY=your_key
```

#### "Memory limit exceeded"
```python
# Enable memory optimization
config = OptimizationConfig(enable_memory_optimization=True)
cleaner = TableCleaner(optimization_config=config)

# Or process in smaller chunks
chunks = processor.split_dataframe(df)
results = [cleaner.clean(chunk) for chunk in chunks]
```

#### "Processing too slow"
```python
# Enable parallel processing
config = OptimizationConfig(
    enable_parallel_processing=True,
    max_workers=8,
    enable_caching=True
)
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging
cleaner = TableCleaner(enable_profiling=True)
cleaned_df, report = cleaner.clean(df)
```

## 📚 Additional Resources

- **GitHub Repository**: https://github.com/company/llm-tab-cleaner
- **Documentation**: https://docs.company.com/llm-tab-cleaner  
- **Examples**: https://github.com/company/llm-tab-cleaner/tree/main/examples
- **Support**: support@company.com

---

**🚀 This comprehensive API reference covers all features of LLM Tab Cleaner v0.3.0. For the latest updates, visit our documentation portal.**