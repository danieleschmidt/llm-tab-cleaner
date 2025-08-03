"""Performance benchmarking and testing utilities."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import statistics

import pandas as pd
import numpy as np

from .core import TableCleaner
from .cleaning_rule import create_default_rules
from .profiler import DataProfiler


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    name: str
    duration: float
    memory_usage: Optional[float] = None
    rows_processed: int = 0
    throughput: float = 0.0  # rows per second
    quality_score: float = 0.0
    fixes_applied: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.rows_processed > 0 and self.duration > 0:
            self.throughput = self.rows_processed / self.duration
        
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    results: List[BenchmarkResult]
    total_duration: float = 0.0
    summary_stats: Dict[str, Any] = None
    
    def __post_init__(self):
        """Calculate summary statistics."""
        if self.results:
            durations = [r.duration for r in self.results]
            throughputs = [r.throughput for r in self.results if r.throughput > 0]
            quality_scores = [r.quality_score for r in self.results if r.quality_score > 0]
            
            self.total_duration = sum(durations)
            
            self.summary_stats = {
                "total_benchmarks": len(self.results),
                "avg_duration": statistics.mean(durations),
                "median_duration": statistics.median(durations),
                "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
                "median_throughput": statistics.median(throughputs) if throughputs else 0,
                "avg_quality_score": statistics.mean(quality_scores) if quality_scores else 0,
                "total_rows_processed": sum(r.rows_processed for r in self.results),
                "total_fixes_applied": sum(r.fixes_applied for r in self.results)
            }


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking for data cleaning operations."""
    
    def __init__(self, enable_memory_tracking: bool = True):
        """Initialize benchmarker.
        
        Args:
            enable_memory_tracking: Whether to track memory usage (requires psutil)
        """
        self.enable_memory_tracking = enable_memory_tracking
        self._memory_tracker = None
        
        if enable_memory_tracking:
            try:
                import psutil
                self._memory_tracker = psutil.Process()
            except ImportError:
                logger.warning("psutil not available, memory tracking disabled")
                self.enable_memory_tracking = False
    
    def benchmark_cleaning_performance(
        self,
        datasets: Dict[str, pd.DataFrame],
        cleaners: Dict[str, TableCleaner],
        iterations: int = 3
    ) -> BenchmarkSuite:
        """Benchmark cleaning performance across datasets and configurations.
        
        Args:
            datasets: Dictionary of dataset_name -> DataFrame
            cleaners: Dictionary of cleaner_name -> TableCleaner
            iterations: Number of iterations per test
            
        Returns:
            Benchmark results
        """
        logger.info(f"Starting cleaning performance benchmark with {len(datasets)} datasets, {len(cleaners)} cleaners")
        
        results = []
        
        for dataset_name, df in datasets.items():
            for cleaner_name, cleaner in cleaners.items():
                logger.info(f"Benchmarking {cleaner_name} on {dataset_name}")
                
                # Run multiple iterations
                iteration_results = []
                
                for iteration in range(iterations):
                    result = self._benchmark_single_cleaning(
                        df, cleaner, f"{cleaner_name}_{dataset_name}_iter{iteration}"
                    )
                    iteration_results.append(result)
                
                # Aggregate results
                avg_result = self._aggregate_results(
                    iteration_results, f"{cleaner_name}_{dataset_name}"
                )
                results.append(avg_result)
        
        return BenchmarkSuite(
            name="cleaning_performance",
            results=results
        )
    
    def benchmark_scalability(
        self,
        base_df: pd.DataFrame,
        cleaner: TableCleaner,
        scale_factors: List[int] = [1, 2, 5, 10]
    ) -> BenchmarkSuite:
        """Benchmark scalability across different data sizes.
        
        Args:
            base_df: Base DataFrame to scale
            cleaner: TableCleaner instance
            scale_factors: Multipliers for data size
            
        Returns:
            Scalability benchmark results
        """
        logger.info(f"Starting scalability benchmark with factors: {scale_factors}")
        
        results = []
        
        for factor in scale_factors:
            # Create scaled dataset
            if factor == 1:
                scaled_df = base_df.copy()
            else:
                # Repeat the data to scale up
                scaled_df = pd.concat([base_df] * factor, ignore_index=True)
            
            logger.info(f"Testing scale factor {factor} ({len(scaled_df)} rows)")
            
            result = self._benchmark_single_cleaning(
                scaled_df, cleaner, f"scale_{factor}x"
            )
            result.metadata["scale_factor"] = factor
            results.append(result)
        
        return BenchmarkSuite(
            name="scalability",
            results=results
        )
    
    def benchmark_provider_comparison(
        self,
        df: pd.DataFrame,
        providers: List[str] = ["local", "anthropic", "openai"],
        iterations: int = 3
    ) -> BenchmarkSuite:
        """Benchmark different LLM providers.
        
        Args:
            df: Test DataFrame
            providers: List of provider names to test
            iterations: Number of iterations per provider
            
        Returns:
            Provider comparison results
        """
        logger.info(f"Starting provider comparison: {providers}")
        
        results = []
        
        for provider in providers:
            try:
                cleaner = TableCleaner(
                    llm_provider=provider,
                    confidence_threshold=0.85,
                    enable_profiling=False  # Disable for fair comparison
                )
                
                iteration_results = []
                
                for iteration in range(iterations):
                    result = self._benchmark_single_cleaning(
                        df, cleaner, f"{provider}_iter{iteration}"
                    )
                    iteration_results.append(result)
                
                # Aggregate results
                avg_result = self._aggregate_results(
                    iteration_results, f"provider_{provider}"
                )
                avg_result.metadata["provider"] = provider
                results.append(avg_result)
                
            except Exception as e:
                logger.error(f"Error benchmarking provider {provider}: {e}")
                # Add failed result
                results.append(BenchmarkResult(
                    name=f"provider_{provider}",
                    duration=0.0,
                    metadata={"provider": provider, "error": str(e)}
                ))
        
        return BenchmarkSuite(
            name="provider_comparison",
            results=results
        )
    
    def benchmark_profiling_performance(
        self,
        datasets: Dict[str, pd.DataFrame]
    ) -> BenchmarkSuite:
        """Benchmark data profiling performance.
        
        Args:
            datasets: Dictionary of dataset_name -> DataFrame
            
        Returns:
            Profiling benchmark results
        """
        logger.info("Starting profiling performance benchmark")
        
        results = []
        profiler = DataProfiler()
        
        for dataset_name, df in datasets.items():
            logger.info(f"Benchmarking profiling on {dataset_name}")
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                profile = profiler.profile_table(df)
                
                duration = time.time() - start_time
                end_memory = self._get_memory_usage()
                memory_usage = end_memory - start_memory if start_memory and end_memory else None
                
                result = BenchmarkResult(
                    name=f"profiling_{dataset_name}",
                    duration=duration,
                    memory_usage=memory_usage,
                    rows_processed=len(df),
                    quality_score=profile.overall_quality_score,
                    metadata={
                        "columns": len(df.columns),
                        "total_issues": profile.total_issues,
                        "duplicate_percentage": profile.duplicate_percentage
                    }
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error profiling {dataset_name}: {e}")
                results.append(BenchmarkResult(
                    name=f"profiling_{dataset_name}",
                    duration=0.0,
                    metadata={"error": str(e)}
                ))
        
        return BenchmarkSuite(
            name="profiling_performance",
            results=results
        )
    
    def generate_synthetic_datasets(
        self,
        sizes: List[int] = [1000, 10000, 100000],
        column_counts: List[int] = [5, 10, 20],
        quality_levels: List[str] = ["high", "medium", "low"]
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic datasets for benchmarking.
        
        Args:
            sizes: Row counts for generated datasets
            column_counts: Number of columns
            quality_levels: Data quality levels
            
        Returns:
            Dictionary of synthetic datasets
        """
        logger.info("Generating synthetic datasets for benchmarking")
        
        datasets = {}
        
        for size in sizes:
            for cols in column_counts:
                for quality in quality_levels:
                    name = f"synthetic_{size}rows_{cols}cols_{quality}quality"
                    df = self._generate_synthetic_data(size, cols, quality)
                    datasets[name] = df
        
        logger.info(f"Generated {len(datasets)} synthetic datasets")
        return datasets
    
    def _benchmark_single_cleaning(
        self,
        df: pd.DataFrame,
        cleaner: TableCleaner,
        name: str
    ) -> BenchmarkResult:
        """Benchmark a single cleaning operation."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            cleaned_df, report = cleaner.clean(df)
            
            duration = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_usage = end_memory - start_memory if start_memory and end_memory else None
            
            return BenchmarkResult(
                name=name,
                duration=duration,
                memory_usage=memory_usage,
                rows_processed=len(df),
                quality_score=report.quality_score,
                fixes_applied=report.total_fixes,
                metadata={
                    "columns": len(df.columns),
                    "processing_time": report.processing_time
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in benchmark {name}: {e}")
            
            return BenchmarkResult(
                name=name,
                duration=duration,
                rows_processed=len(df),
                metadata={"error": str(e)}
            )
    
    def _aggregate_results(
        self,
        results: List[BenchmarkResult],
        name: str
    ) -> BenchmarkResult:
        """Aggregate multiple benchmark results."""
        if not results:
            return BenchmarkResult(name=name, duration=0.0)
        
        # Calculate averages
        avg_duration = statistics.mean([r.duration for r in results])
        avg_memory = None
        if all(r.memory_usage is not None for r in results):
            avg_memory = statistics.mean([r.memory_usage for r in results])
        
        avg_quality = statistics.mean([r.quality_score for r in results if r.quality_score > 0])
        total_fixes = sum([r.fixes_applied for r in results])
        
        # Use first result's metadata as base
        metadata = results[0].metadata.copy() if results[0].metadata else {}
        metadata.update({
            "iterations": len(results),
            "std_duration": statistics.stdev([r.duration for r in results]) if len(results) > 1 else 0,
            "min_duration": min([r.duration for r in results]),
            "max_duration": max([r.duration for r in results])
        })
        
        return BenchmarkResult(
            name=name,
            duration=avg_duration,
            memory_usage=avg_memory,
            rows_processed=results[0].rows_processed,
            quality_score=avg_quality,
            fixes_applied=total_fixes,
            metadata=metadata
        )
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        if not self.enable_memory_tracking or not self._memory_tracker:
            return None
        
        try:
            return self._memory_tracker.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            return None
    
    def _generate_synthetic_data(
        self,
        rows: int,
        cols: int,
        quality: str
    ) -> pd.DataFrame:
        """Generate synthetic data with controlled quality issues."""
        np.random.seed(42)  # For reproducible results
        
        data = {}
        
        # Generate different column types
        column_types = ["name", "email", "phone", "age", "salary", "date", "category"]
        
        for i in range(cols):
            col_type = column_types[i % len(column_types)]
            col_name = f"{col_type}_{i}"
            
            if col_type == "name":
                values = self._generate_names(rows, quality)
            elif col_type == "email":
                values = self._generate_emails(rows, quality)
            elif col_type == "phone":
                values = self._generate_phones(rows, quality)
            elif col_type == "age":
                values = self._generate_ages(rows, quality)
            elif col_type == "salary":
                values = self._generate_salaries(rows, quality)
            elif col_type == "date":
                values = self._generate_dates(rows, quality)
            else:  # category
                values = self._generate_categories(rows, quality)
            
            data[col_name] = values
        
        return pd.DataFrame(data)
    
    def _generate_names(self, rows: int, quality: str) -> List[str]:
        """Generate name data with quality issues."""
        base_names = ["John Smith", "Jane Doe", "Bob Johnson", "Alice Brown", "Charlie Wilson"]
        names = []
        
        for _ in range(rows):
            name = np.random.choice(base_names)
            
            if quality == "low":
                # Add quality issues
                if np.random.random() < 0.2:
                    if np.random.random() < 0.5:
                        name = name.lower()  # Case issues
                    else:
                        name = "N/A"  # Missing data
                elif np.random.random() < 0.1:
                    name = "  " + name + "  "  # Whitespace issues
            elif quality == "medium":
                if np.random.random() < 0.1:
                    name = name.lower()
                elif np.random.random() < 0.05:
                    name = "Unknown"
            
            names.append(name)
        
        return names
    
    def _generate_emails(self, rows: int, quality: str) -> List[str]:
        """Generate email data with quality issues."""
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "company.com"]
        emails = []
        
        for i in range(rows):
            email = f"user{i}@{np.random.choice(domains)}"
            
            if quality == "low":
                if np.random.random() < 0.15:
                    email = email.replace("@", "AT")  # Format issues
                elif np.random.random() < 0.1:
                    email = "invalid"
            elif quality == "medium":
                if np.random.random() < 0.05:
                    email = email.upper()  # Case issues
            
            emails.append(email)
        
        return emails
    
    def _generate_phones(self, rows: int, quality: str) -> List[str]:
        """Generate phone data with quality issues."""
        phones = []
        
        for _ in range(rows):
            # Generate base phone number
            area = np.random.randint(200, 999)
            exchange = np.random.randint(200, 999)
            number = np.random.randint(1000, 9999)
            
            phone = f"{area}-{exchange}-{number}"
            
            if quality == "low":
                format_choice = np.random.choice([
                    f"({area}) {exchange}-{number}",
                    f"{area}.{exchange}.{number}",
                    f"{area}{exchange}{number}",
                    "555-CALL-NOW",  # Invalid
                    "N/A"
                ])
                phone = format_choice
            elif quality == "medium":
                if np.random.random() < 0.1:
                    phone = f"({area}) {exchange}-{number}"
            
            phones.append(phone)
        
        return phones
    
    def _generate_ages(self, rows: int, quality: str) -> List[Any]:
        """Generate age data with quality issues."""
        ages = []
        
        for _ in range(rows):
            age = np.random.randint(18, 80)
            
            if quality == "low":
                if np.random.random() < 0.15:
                    age = np.random.choice([-5, 150, "unknown", "N/A"])  # Invalid ages
            elif quality == "medium":
                if np.random.random() < 0.05:
                    age = "N/A"
            
            ages.append(age)
        
        return ages
    
    def _generate_salaries(self, rows: int, quality: str) -> List[Any]:
        """Generate salary data with quality issues."""
        salaries = []
        
        for _ in range(rows):
            salary = np.random.randint(30000, 150000)
            
            if quality == "low":
                if np.random.random() < 0.1:
                    salary = f"${salary:,}"  # Format with currency
                elif np.random.random() < 0.05:
                    salary = "TBD"
            elif quality == "medium":
                if np.random.random() < 0.03:
                    salary = f"${salary}"
            
            salaries.append(salary)
        
        return salaries
    
    def _generate_dates(self, rows: int, quality: str) -> List[str]:
        """Generate date data with quality issues."""
        dates = []
        
        for _ in range(rows):
            year = np.random.randint(2020, 2024)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            
            date = f"{year}-{month:02d}-{day:02d}"
            
            if quality == "low":
                format_choice = np.random.choice([
                    f"{month}/{day}/{year}",
                    f"{day}-{month}-{year}",
                    f"{month}.{day}.{year}",
                    "TBD",
                    f"{year}/{month}/{day}"
                ])
                date = format_choice
            elif quality == "medium":
                if np.random.random() < 0.1:
                    date = f"{month}/{day}/{year}"
            
            dates.append(date)
        
        return dates
    
    def _generate_categories(self, rows: int, quality: str) -> List[str]:
        """Generate categorical data with quality issues."""
        categories = ["A", "B", "C", "D", "E"]
        values = []
        
        for _ in range(rows):
            category = np.random.choice(categories)
            
            if quality == "low":
                if np.random.random() < 0.1:
                    category = category.lower()  # Case issues
                elif np.random.random() < 0.05:
                    category = "Unknown"
            elif quality == "medium":
                if np.random.random() < 0.03:
                    category = category.lower()
            
            values.append(category)
        
        return values


def run_comprehensive_benchmark(
    custom_datasets: Optional[Dict[str, pd.DataFrame]] = None,
    output_file: Optional[str] = None
) -> Dict[str, BenchmarkSuite]:
    """Run a comprehensive benchmark suite.
    
    Args:
        custom_datasets: Optional custom datasets to include
        output_file: Optional file to save results
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info("Starting comprehensive benchmark suite")
    
    benchmarker = PerformanceBenchmarker()
    results = {}
    
    # Generate or use provided datasets
    if custom_datasets:
        datasets = custom_datasets
    else:
        datasets = benchmarker.generate_synthetic_datasets()
    
    # Benchmark 1: Cleaning performance
    cleaners = {
        "local": TableCleaner(llm_provider="local"),
        "local_with_rules": TableCleaner(llm_provider="local", rules=create_default_rules()),
        "local_no_profiling": TableCleaner(llm_provider="local", enable_profiling=False)
    }
    
    results["cleaning_performance"] = benchmarker.benchmark_cleaning_performance(
        datasets, cleaners, iterations=2
    )
    
    # Benchmark 2: Scalability
    base_df = list(datasets.values())[0]  # Use first dataset
    results["scalability"] = benchmarker.benchmark_scalability(
        base_df, cleaners["local"], scale_factors=[1, 2, 5]
    )
    
    # Benchmark 3: Provider comparison (local only for now)
    results["provider_comparison"] = benchmarker.benchmark_provider_comparison(
        base_df, providers=["local"], iterations=2
    )
    
    # Benchmark 4: Profiling performance
    results["profiling_performance"] = benchmarker.benchmark_profiling_performance(datasets)
    
    # Save results if requested
    if output_file:
        import json
        
        # Convert results to serializable format
        serializable_results = {}
        for suite_name, suite in results.items():
            serializable_results[suite_name] = {
                "name": suite.name,
                "total_duration": suite.total_duration,
                "summary_stats": suite.summary_stats,
                "results": [
                    {
                        "name": r.name,
                        "duration": r.duration,
                        "memory_usage": r.memory_usage,
                        "rows_processed": r.rows_processed,
                        "throughput": r.throughput,
                        "quality_score": r.quality_score,
                        "fixes_applied": r.fixes_applied,
                        "metadata": r.metadata
                    }
                    for r in suite.results
                ]
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved benchmark results to {output_file}")
    
    logger.info("Comprehensive benchmark completed")
    return results