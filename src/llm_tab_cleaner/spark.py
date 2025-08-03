"""Spark distributed processing for large-scale data cleaning."""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, pandas_udf, struct, lit
    from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
    import pandas as pd
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkSession = None
    SparkDataFrame = None

from .core import TableCleaner, CleaningReport, Fix
from .cleaning_rule import RuleSet
from .llm_providers import get_provider


logger = logging.getLogger(__name__)


class SparkCleaner:
    """Distributed data cleaning using Apache Spark."""
    
    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        llm_provider: str = "local",
        confidence_threshold: float = 0.85,
        rules: Optional[RuleSet] = None,
        batch_size: int = 10000,
        parallelism: int = 100,
        **provider_kwargs
    ):
        """Initialize Spark cleaner.
        
        Args:
            spark: SparkSession instance (if None, creates new session)
            llm_provider: LLM provider to use
            confidence_threshold: Minimum confidence for applying fixes
            rules: Custom cleaning rules
            batch_size: Number of rows per partition
            parallelism: Number of parallel partitions
            **provider_kwargs: Additional LLM provider arguments
        """
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is required for SparkCleaner. Install with: pip install llm-tab-cleaner[spark]")
        
        self.spark = spark or self._create_spark_session()
        self.llm_provider_name = llm_provider
        self.confidence_threshold = confidence_threshold
        self.rules = rules
        self.batch_size = batch_size
        self.parallelism = parallelism
        self.provider_kwargs = provider_kwargs
        
        # Setup broadcast variables for efficient distribution
        self._setup_broadcast_variables()
        
        logger.info(f"Initialized SparkCleaner with {llm_provider} provider")
    
    def clean_distributed(
        self,
        df: SparkDataFrame,
        columns: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        audit_log: Optional[str] = None
    ) -> SparkDataFrame:
        """Clean a Spark DataFrame in distributed fashion.
        
        Args:
            df: Input Spark DataFrame
            columns: Specific columns to clean (if None, cleans all)
            output_path: Optional path to save cleaned data
            checkpoint_dir: Optional checkpoint directory for fault tolerance
            audit_log: Optional path to save audit logs
            
        Returns:
            Cleaned Spark DataFrame
        """
        logger.info(f"Starting distributed cleaning of DataFrame with {df.count()} rows")
        
        # Set checkpoint directory if provided
        if checkpoint_dir:
            self.spark.sparkContext.setCheckpointDir(checkpoint_dir)
        
        # Determine columns to clean
        target_columns = columns if columns is not None else df.columns
        
        # Repartition for optimal processing
        optimal_partitions = max(1, min(self.parallelism, df.count() // self.batch_size))
        df_partitioned = df.repartition(optimal_partitions)
        
        logger.info(f"Processing with {optimal_partitions} partitions")
        
        # Clean each column
        cleaned_df = df_partitioned
        all_fixes = []
        
        for column in target_columns:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found, skipping")
                continue
            
            logger.info(f"Cleaning column: {column}")
            
            # Create UDF for cleaning this column
            cleaning_udf = self._create_cleaning_udf(column)
            
            # Apply cleaning and collect fixes
            result_df = cleaned_df.withColumn(
                f"{column}_cleaned_result",
                cleaning_udf(col(column))
            )
            
            # Extract cleaned values and fixes
            cleaned_df = result_df.withColumn(
                column,
                col(f"{column}_cleaned_result.cleaned_value")
            ).drop(f"{column}_cleaned_result")
            
            # Collect fixes for audit
            if audit_log:
                fixes_df = result_df.select(
                    lit(column).alias("column"),
                    col(f"{column}_cleaned_result.fixes")
                ).filter(
                    col(f"{column}_cleaned_result.fixes").isNotNull()
                )
                
                if fixes_df.count() > 0:
                    all_fixes.append(fixes_df)
        
        # Checkpoint for fault tolerance
        if checkpoint_dir:
            cleaned_df = cleaned_df.checkpoint()
        
        # Save audit logs if requested
        if audit_log and all_fixes:
            audit_df = all_fixes[0]
            for fix_df in all_fixes[1:]:
                audit_df = audit_df.union(fix_df)
            
            audit_df.write.mode("overwrite").json(audit_log)
            logger.info(f"Saved audit logs to {audit_log}")
        
        # Save output if requested
        if output_path:
            cleaned_df.write.mode("overwrite").parquet(output_path)
            logger.info(f"Saved cleaned data to {output_path}")
        
        logger.info("Distributed cleaning completed")
        return cleaned_df
    
    def profile_distributed(self, df: SparkDataFrame) -> Dict[str, Any]:
        """Profile a Spark DataFrame for quality assessment.
        
        Args:
            df: Input Spark DataFrame
            
        Returns:
            Profiling results
        """
        logger.info("Starting distributed profiling")
        
        # Basic statistics
        row_count = df.count()
        column_count = len(df.columns)
        
        # Column-level profiling
        column_profiles = {}
        
        for column in df.columns:
            logger.info(f"Profiling column: {column}")
            
            # Basic stats
            null_count = df.filter(col(column).isNull()).count()
            unique_count = df.select(column).distinct().count()
            
            # Sample for pattern analysis
            sample_values = [
                row[column] for row in 
                df.select(column).filter(col(column).isNotNull()).limit(100).collect()
            ]
            
            column_profiles[column] = {
                "null_count": null_count,
                "null_percentage": (null_count / row_count) * 100 if row_count > 0 else 0,
                "unique_count": unique_count,
                "unique_percentage": (unique_count / row_count) * 100 if row_count > 0 else 0,
                "sample_values": sample_values[:10]  # Limit for serialization
            }
        
        # Duplicate analysis
        duplicate_count = row_count - df.distinct().count()
        
        profile_result = {
            "row_count": row_count,
            "column_count": column_count,
            "duplicate_count": duplicate_count,
            "duplicate_percentage": (duplicate_count / row_count) * 100 if row_count > 0 else 0,
            "columns": column_profiles
        }
        
        logger.info("Distributed profiling completed")
        return profile_result
    
    def _create_spark_session(self) -> SparkSession:
        """Create optimized Spark session for data cleaning."""
        return SparkSession.builder \
            .appName("LLM-Tab-Cleaner") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
    
    def _setup_broadcast_variables(self) -> None:
        """Setup broadcast variables for efficient distribution."""
        # Broadcast cleaning configuration
        self.broadcast_config = self.spark.sparkContext.broadcast({
            "llm_provider": self.llm_provider_name,
            "confidence_threshold": self.confidence_threshold,
            "provider_kwargs": self.provider_kwargs
        })
        
        # Broadcast rules if available
        if self.rules:
            rules_data = {
                "rules": [
                    {
                        "name": rule.name,
                        "description": rule.description,
                        "examples": rule.examples,
                        "pattern": rule.pattern,
                        "transform": rule.transform,
                        "confidence": rule.confidence,
                        "column_patterns": rule.column_patterns,
                        "data_types": rule.data_types
                    }
                    for rule in self.rules.rules
                ]
            }
            self.broadcast_rules = self.spark.sparkContext.broadcast(rules_data)
        else:
            self.broadcast_rules = None
    
    def _create_cleaning_udf(self, column_name: str):
        """Create a pandas UDF for cleaning a specific column."""
        
        # Define return schema
        return_schema = StructType([
            StructField("cleaned_value", StringType(), True),
            StructField("confidence", FloatType(), True),
            StructField("fixes", StringType(), True)  # JSON string of fixes
        ])
        
        @pandas_udf(returnType=return_schema)
        def clean_column(values: pd.Series) -> pd.DataFrame:
            """Pandas UDF to clean column values."""
            # Get broadcast variables
            config = self.broadcast_config.value
            rules_data = self.broadcast_rules.value if self.broadcast_rules else None
            
            # Initialize cleaner (one per partition)
            provider = get_provider(
                config["llm_provider"], 
                **config["provider_kwargs"]
            )
            
            # Reconstruct rules if available
            rules = None
            if rules_data:
                from .cleaning_rule import CleaningRule, RuleSet
                rules_list = []
                for rule_data in rules_data["rules"]:
                    rule = CleaningRule(
                        name=rule_data["name"],
                        description=rule_data["description"],
                        examples=rule_data["examples"],
                        pattern=rule_data["pattern"],
                        transform=rule_data["transform"],
                        confidence=rule_data["confidence"],
                        column_patterns=rule_data["column_patterns"],
                        data_types=rule_data["data_types"]
                    )
                    rules_list.append(rule)
                rules = RuleSet(rules_list)
            
            # Process each value
            results = []
            fixes_list = []
            
            for value in values:
                try:
                    # Apply rules first if available
                    if rules:
                        cleaned_value, confidence, applied_rules = rules.apply_rules(
                            value, column_name
                        )
                        if cleaned_value != value:
                            fix_data = {
                                "original": str(value),
                                "cleaned": str(cleaned_value),
                                "confidence": confidence,
                                "rules": applied_rules
                            }
                            fixes_list.append(json.dumps(fix_data))
                            results.append((str(cleaned_value), confidence, json.dumps(fix_data)))
                            continue
                    
                    # Apply LLM cleaning if no rule applied
                    context = {"column_name": column_name, "data_type": "unknown"}
                    cleaned_value, confidence = provider.clean_value(value, column_name, context)
                    
                    if cleaned_value != value and confidence >= config["confidence_threshold"]:
                        fix_data = {
                            "original": str(value),
                            "cleaned": str(cleaned_value),
                            "confidence": confidence,
                            "provider": config["llm_provider"]
                        }
                        results.append((str(cleaned_value), confidence, json.dumps(fix_data)))
                    else:
                        results.append((str(value), 1.0, None))
                        
                except Exception as e:
                    logger.error(f"Error cleaning value {value}: {e}")
                    results.append((str(value), 0.0, None))
            
            # Convert to DataFrame
            return pd.DataFrame(results, columns=["cleaned_value", "confidence", "fixes"])
        
        return clean_column


class StreamingCleaner:
    """Real-time data cleaning for streaming data."""
    
    def __init__(
        self,
        spark: SparkSession,
        llm_provider: str = "local",
        confidence_threshold: float = 0.85,
        checkpoint_location: str = "/tmp/streaming_cleaner_checkpoint"
    ):
        """Initialize streaming cleaner.
        
        Args:
            spark: SparkSession with streaming enabled
            llm_provider: LLM provider to use
            confidence_threshold: Minimum confidence for applying fixes
            checkpoint_location: Location for streaming checkpoints
        """
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is required for StreamingCleaner")
        
        self.spark = spark
        self.llm_provider = llm_provider
        self.confidence_threshold = confidence_threshold
        self.checkpoint_location = checkpoint_location
        
        logger.info("Initialized StreamingCleaner")
    
    def clean_stream(
        self,
        input_stream,
        output_path: str,
        trigger_interval: str = "10 seconds",
        columns: Optional[List[str]] = None
    ):
        """Clean streaming data in real-time.
        
        Args:
            input_stream: Input streaming DataFrame
            output_path: Output path for cleaned data
            trigger_interval: Processing trigger interval
            columns: Columns to clean
        """
        logger.info("Starting stream cleaning")
        
        # Apply cleaning transformations
        cleaned_stream = input_stream
        
        # For now, implement basic cleaning
        # Full LLM integration would require careful batching
        for column in (columns or input_stream.columns):
            if column in input_stream.columns:
                # Apply basic cleaning rules
                cleaned_stream = cleaned_stream.withColumn(
                    column,
                    # Add basic cleaning transformations here
                    col(column)
                )
        
        # Start streaming query
        query = cleaned_stream.writeStream \
            .outputMode("append") \
            .format("delta") \
            .option("path", output_path) \
            .option("checkpointLocation", self.checkpoint_location) \
            .trigger(processingTime=trigger_interval) \
            .start()
        
        logger.info(f"Started streaming query writing to {output_path}")
        return query


def create_spark_cleaner(
    spark: Optional[SparkSession] = None,
    **kwargs
) -> SparkCleaner:
    """Factory function to create SparkCleaner with optimal configuration."""
    if spark is None:
        spark = SparkSession.builder \
            .appName("LLM-Tab-Cleaner-Distributed") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
    
    return SparkCleaner(spark=spark, **kwargs)