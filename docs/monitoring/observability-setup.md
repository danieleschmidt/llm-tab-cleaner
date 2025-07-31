# Observability and Performance Monitoring Setup

This document outlines the recommended monitoring and observability setup for LLM Tab Cleaner in production environments.

## Overview

LLM Tab Cleaner includes built-in support for comprehensive observability through:

- **Metrics**: Performance counters, cleaning statistics, and resource usage
- **Logging**: Structured logging with configurable levels and formats
- **Tracing**: Distributed tracing for complex cleaning pipelines
- **Health Checks**: Application and dependency health monitoring
- **Alerts**: Automated alerting for critical issues

## Metrics Collection

### Built-in Metrics

LLM Tab Cleaner exports the following metrics by default:

```python
# Core cleaning metrics
llm_cleaning_total_records_processed
llm_cleaning_total_fixes_applied
llm_cleaning_confidence_score_histogram
llm_cleaning_processing_duration_seconds
llm_cleaning_api_calls_total
llm_cleaning_api_latency_seconds

# Quality metrics
llm_cleaning_quality_score_histogram
llv_cleaning_error_rate
llm_cleaning_rejected_fixes_total

# Resource metrics
llm_cleaning_memory_usage_bytes
llm_cleaning_cpu_usage_percent
llm_cleaning_api_tokens_consumed
```

### Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-tab-cleaner'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 30s
    metrics_path: /metrics
    scrape_timeout: 10s
```

### Grafana Dashboard

Import the provided dashboard from `docs/monitoring/grafana-dashboard.json`:

Key panels include:
- Cleaning throughput and latency
- LLM API usage and costs
- Data quality trends
- Error rates and alert status
- Resource utilization

## Logging Configuration

### Structured Logging Setup

```python
import logging
from llm_tab_cleaner.monitoring import setup_logging

# Configure structured logging
setup_logging(
    level=logging.INFO,
    format="json",
    include_trace_id=True,
    log_llm_interactions=True
)
```

### Log Levels and Categories

- **ERROR**: Critical failures, API errors, data corruption
- **WARN**: Low confidence fixes, rate limits, retries
- **INFO**: Cleaning summaries, configuration changes
- **DEBUG**: Detailed processing steps, API requests/responses

### ELK Stack Integration

For centralized logging with Elasticsearch, Logstash, and Kibana:

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  llm-cleaner:
    image: llm-tab-cleaner:latest
    environment:
      - LOGGING_ELASTICSEARCH_HOST=elasticsearch:9200
      - LOGGING_FORMAT=json
      - LOGGING_INCLUDE_TRACE_ID=true
    depends_on:
      - elasticsearch
      - logstash

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Distributed Tracing

### OpenTelemetry Setup

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from llm_tab_cleaner.monitoring import setup_tracing

# Configure tracing
setup_tracing(
    service_name="llm-tab-cleaner",
    jaeger_endpoint="http://jaeger:14268/api/traces",
    sample_rate=0.1  # Sample 10% of traces
)
```

### Jaeger Configuration

```yaml
# docker-compose.tracing.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Accept traces
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

### Custom Trace Spans

```python
from llm_tab_cleaner import TableCleaner
from llm_tab_cleaner.monitoring import trace_span

@trace_span("custom_cleaning_pipeline")
def clean_customer_data(df):
    cleaner = TableCleaner()
    
    with trace_span("data_validation"):
        validated_df = validate_input(df)
    
    with trace_span("llm_cleaning"):
        cleaned_df, report = cleaner.clean(validated_df)
    
    with trace_span("quality_check"):
        quality_score = assess_quality(cleaned_df)
    
    return cleaned_df, report, quality_score
```

## Health Checks

### Application Health Endpoint

LLM Tab Cleaner provides a built-in health check endpoint:

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed health with dependencies
curl http://localhost:8080/health/detailed
```

Response format:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-31T10:00:00Z",
  "version": "0.1.0",
  "checks": {
    "database": {"status": "healthy", "latency": "2ms"},
    "llm_api": {"status": "healthy", "latency": "150ms"},
    "cache": {"status": "healthy", "latency": "1ms"}
  },
  "metrics": {
    "uptime": "2h 30m",
    "processed_records": 15000,
    "avg_processing_time": "45ms"
  }
}
```

### Kubernetes Health Checks

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: llm-tab-cleaner
    image: llm-tab-cleaner:latest
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
```

## Alerting

### Alert Rules (Prometheus/AlertManager)

```yaml
# alerts.yml
groups:
  - name: llm-tab-cleaner
    rules:
      - alert: HighErrorRate
        expr: rate(llm_cleaning_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in LLM Tab Cleaner"
          description: "Error rate is {{ $value }} per second"

      - alert: LowConfidenceSpike
        expr: rate(llm_cleaning_low_confidence_total[5m]) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Unusual spike in low-confidence predictions"

      - alert: APILatencyHigh
        expr: histogram_quantile(0.95, llm_cleaning_api_latency_seconds) > 5
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "LLM API latency is high"

      - alert: ServiceDown
        expr: up{job="llm-tab-cleaner"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM Tab Cleaner service is down"
```

### Notification Channels

Configure notifications to:
- **Slack**: Real-time alerts for the team
- **PagerDuty**: Critical issues requiring immediate attention
- **Email**: Summary reports and non-critical alerts
- **Webhook**: Custom integrations with incident management systems

## Performance Monitoring

### Key Performance Indicators (KPIs)

Monitor these critical metrics:

1. **Throughput**: Records processed per second
2. **Latency**: End-to-end processing time (p50, p95, p99)
3. **Quality**: Average confidence score and fix success rate
4. **Cost**: LLM API costs per record processed
5. **Availability**: Service uptime and health check success rate

### Performance Optimization

```python
from llm_tab_cleaner.monitoring import PerformanceProfiler

# Enable performance profiling
profiler = PerformanceProfiler()
with profiler.profile("data_cleaning_batch"):
    cleaned_df, report = cleaner.clean(df)

# Get performance report
performance_report = profiler.get_report()
print(f"Processing time: {performance_report.total_time}s")
print(f"Memory peak: {performance_report.peak_memory}MB")
print(f"API calls: {performance_report.api_calls}")
```

### Custom Dashboards

Create custom Grafana dashboards for different stakeholders:

- **Operations Dashboard**: System health, errors, performance
- **Business Dashboard**: Data quality trends, processing volumes
- **Cost Dashboard**: LLM API usage, cost per record, budget tracking
- **Development Dashboard**: Code coverage, deployment frequency, lead time

## Monitoring in Different Environments

### Development
- Basic console logging
- Local metrics collection
- Simple health checks

### Staging  
- Structured JSON logging
- Full metrics and tracing
- Comprehensive health checks
- Test alert configurations

### Production
- Centralized logging (ELK/Fluentd)
- High-resolution metrics with long retention
- Distributed tracing with sampling
- Full alerting with escalation policies
- SLA monitoring and reporting

## Compliance and Audit Logging

For regulated environments:

```python
from llm_tab_cleaner.monitoring import AuditLogger

audit_logger = AuditLogger(
    compliance_mode="SOX",  # SOX, HIPAA, GDPR
    retention_days=2555,    # 7 years for SOX
    encryption=True,
    immutable=True
)

# Automatic audit logging of all data changes
cleaner = TableCleaner(audit_logger=audit_logger)
```

## Troubleshooting Common Issues

### High Memory Usage
- Enable memory profiling: `PROFILING_MEMORY=true`
- Check batch sizes and adjust accordingly
- Monitor for memory leaks in long-running processes

### API Rate Limits
- Monitor `llm_cleaning_api_rate_limit_errors`
- Implement exponential backoff
- Consider multiple API keys or providers

### Poor Data Quality Scores
- Analyze low-confidence predictions
- Review and update cleaning rules
- Consider model fine-tuning or provider switching

### Performance Degradation
- Check system resource utilization
- Analyze slow query logs
- Review API response times and adjust timeouts

For more detailed troubleshooting, see the [Troubleshooting Guide](../troubleshooting.md).