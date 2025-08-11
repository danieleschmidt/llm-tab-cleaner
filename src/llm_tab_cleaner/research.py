"""Research-driven LLM cleaning with experimental algorithms and benchmarking."""

import asyncio
import logging
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy import stats

from .core import TableCleaner, CleaningReport, Fix
from .llm_providers import get_provider


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single cleaning experiment."""
    algorithm_name: str
    dataset_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time: float
    confidence_distribution: List[float]
    fixes_applied: int
    statistical_significance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark datasets and ground truth."""
    name: str
    datasets: List[Tuple[pd.DataFrame, pd.DataFrame]]  # (dirty, clean) pairs
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchAlgorithm(ABC):
    """Abstract base class for research cleaning algorithms."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name for identification."""
        pass
    
    @abstractmethod
    async def clean_async(
        self, 
        df: pd.DataFrame, 
        ground_truth: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean DataFrame asynchronously."""
        pass


class EnsembleLLMCleaner(ResearchAlgorithm):
    """Ensemble of multiple LLM providers with voting mechanism."""
    
    def __init__(
        self, 
        providers: List[str] = ["openai", "anthropic", "local"],
        voting_strategy: str = "majority",
        confidence_weighting: bool = True
    ):
        self.providers = [get_provider(p) for p in providers]
        self.voting_strategy = voting_strategy
        self.confidence_weighting = confidence_weighting
        
    @property
    def name(self) -> str:
        return f"ensemble_llm_{self.voting_strategy}"
    
    async def clean_async(
        self, 
        df: pd.DataFrame, 
        ground_truth: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean using ensemble of LLM providers."""
        start_time = time.time()
        
        # Get predictions from all providers
        provider_results = []
        
        with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
            futures = []
            
            for provider in self.providers:
                cleaner = TableCleaner(llm_provider=provider.name)
                future = executor.submit(cleaner.clean, df)
                futures.append((provider.name, future))
            
            for provider_name, future in futures:
                try:
                    cleaned_df, report = future.result()
                    provider_results.append((provider_name, cleaned_df, report))
                except Exception as e:
                    logger.warning(f"Provider {provider_name} failed: {e}")
        
        # Ensemble voting
        final_df = df.copy()
        ensemble_fixes = []
        
        if self.voting_strategy == "majority":
            final_df, ensemble_fixes = self._majority_voting(df, provider_results)
        elif self.voting_strategy == "confidence_weighted":
            final_df, ensemble_fixes = self._confidence_weighted_voting(df, provider_results)
        
        processing_time = time.time() - start_time
        
        report = CleaningReport(
            total_fixes=len(ensemble_fixes),
            quality_score=self._calculate_ensemble_quality(df, final_df, ensemble_fixes),
            fixes=ensemble_fixes,
            processing_time=processing_time
        )
        
        return final_df, report
    
    def _majority_voting(
        self, 
        original_df: pd.DataFrame, 
        provider_results: List[Tuple[str, pd.DataFrame, CleaningReport]]
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """Apply majority voting across providers."""
        result_df = original_df.copy()
        fixes = []
        
        # For each cell, collect votes from providers
        for row_idx in range(len(original_df)):
            for col_name in original_df.columns:
                votes = {}
                confidences = {}
                
                for provider_name, cleaned_df, report in provider_results:
                    if row_idx < len(cleaned_df) and col_name in cleaned_df.columns:
                        new_value = cleaned_df.iloc[row_idx][col_name]
                        votes[str(new_value)] = votes.get(str(new_value), 0) + 1
                        
                        # Find confidence for this fix
                        matching_fixes = [
                            f for f in report.fixes 
                            if f.row_index == row_idx and f.column == col_name
                        ]
                        if matching_fixes:
                            confidences[str(new_value)] = max(
                                confidences.get(str(new_value), 0),
                                matching_fixes[0].confidence
                            )
                
                # Apply majority vote
                if votes:
                    majority_value = max(votes, key=votes.get)
                    majority_count = votes[majority_value]
                    
                    if majority_count > len(provider_results) // 2:
                        original_value = original_df.iloc[row_idx][col_name]
                        if str(original_value) != majority_value:
                            # Convert back to appropriate type
                            try:
                                if pd.isna(original_value):
                                    converted_value = None if majority_value == 'None' else majority_value
                                else:
                                    converted_value = type(original_value)(majority_value)
                            except (ValueError, TypeError):
                                converted_value = majority_value
                            
                            result_df.iloc[row_idx, result_df.columns.get_loc(col_name)] = converted_value
                            
                            fix = Fix(
                                column=col_name,
                                row_index=row_idx,
                                original=original_value,
                                cleaned=converted_value,
                                confidence=confidences.get(majority_value, 0.8),
                                reasoning=f"Majority vote ({majority_count}/{len(provider_results)} providers)",
                                rule_applied="ensemble_majority"
                            )
                            fixes.append(fix)
        
        return result_df, fixes
    
    def _confidence_weighted_voting(
        self, 
        original_df: pd.DataFrame, 
        provider_results: List[Tuple[str, pd.DataFrame, CleaningReport]]
    ) -> Tuple[pd.DataFrame, List[Fix]]:
        """Apply confidence-weighted voting."""
        result_df = original_df.copy()
        fixes = []
        
        for row_idx in range(len(original_df)):
            for col_name in original_df.columns:
                weighted_votes = {}
                
                for provider_name, cleaned_df, report in provider_results:
                    if row_idx < len(cleaned_df) and col_name in cleaned_df.columns:
                        new_value = cleaned_df.iloc[row_idx][col_name]
                        
                        # Find confidence for this fix
                        matching_fixes = [
                            f for f in report.fixes 
                            if f.row_index == row_idx and f.column == col_name
                        ]
                        confidence = matching_fixes[0].confidence if matching_fixes else 0.5
                        
                        key = str(new_value)
                        if key not in weighted_votes:
                            weighted_votes[key] = {"weight": 0, "value": new_value}
                        weighted_votes[key]["weight"] += confidence
                
                # Apply highest weighted vote
                if weighted_votes:
                    best_vote = max(weighted_votes.values(), key=lambda x: x["weight"])
                    original_value = original_df.iloc[row_idx][col_name]
                    
                    if str(original_value) != str(best_vote["value"]) and best_vote["weight"] > 0.7:
                        result_df.iloc[row_idx, result_df.columns.get_loc(col_name)] = best_vote["value"]
                        
                        fix = Fix(
                            column=col_name,
                            row_index=row_idx,
                            original=original_value,
                            cleaned=best_vote["value"],
                            confidence=min(best_vote["weight"], 1.0),
                            reasoning=f"Confidence-weighted vote (weight: {best_vote['weight']:.3f})",
                            rule_applied="ensemble_weighted"
                        )
                        fixes.append(fix)
        
        return result_df, fixes
    
    def _calculate_ensemble_quality(
        self, 
        original_df: pd.DataFrame, 
        cleaned_df: pd.DataFrame, 
        fixes: List[Fix]
    ) -> float:
        """Calculate quality score for ensemble results."""
        if not fixes:
            return 1.0
        
        avg_confidence = sum(f.confidence for f in fixes) / len(fixes)
        fix_ratio = len(fixes) / (original_df.shape[0] * original_df.shape[1])
        
        return min(1.0, 0.7 + avg_confidence * 0.2 + min(fix_ratio * 10, 0.1))


class AdaptiveLLMCleaner(ResearchAlgorithm):
    """Adaptive LLM cleaner that learns from previous corrections."""
    
    def __init__(
        self, 
        base_provider: str = "anthropic",
        learning_rate: float = 0.1,
        memory_size: int = 1000
    ):
        self.base_provider = get_provider(base_provider)
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.correction_memory: List[Dict[str, Any]] = []
        
    @property
    def name(self) -> str:
        return f"adaptive_llm_{self.base_provider.name}"
    
    async def clean_async(
        self, 
        df: pd.DataFrame, 
        ground_truth: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean with adaptive learning from corrections."""
        start_time = time.time()
        
        # Base cleaning
        cleaner = TableCleaner(llm_provider=self.base_provider.name)
        cleaned_df, base_report = cleaner.clean(df)
        
        # Apply learned corrections
        adaptive_fixes = []
        for fix in base_report.fixes:
            # Check if we have learned a better correction for similar cases
            similar_corrections = self._find_similar_corrections(fix)
            
            if similar_corrections:
                # Apply learned correction with weighted confidence
                learned_confidence = self._calculate_learned_confidence(similar_corrections)
                if learned_confidence > fix.confidence:
                    improved_fix = Fix(
                        column=fix.column,
                        row_index=fix.row_index,
                        original=fix.original,
                        cleaned=similar_corrections[0]["corrected_value"],
                        confidence=learned_confidence,
                        reasoning=f"Adaptive learning (base: {fix.confidence:.3f}, learned: {learned_confidence:.3f})",
                        rule_applied="adaptive_learning"
                    )
                    adaptive_fixes.append(improved_fix)
                    
                    # Apply the improvement
                    cleaned_df.iloc[fix.row_index, cleaned_df.columns.get_loc(fix.column)] = improved_fix.cleaned
                else:
                    adaptive_fixes.append(fix)
            else:
                adaptive_fixes.append(fix)
        
        # Learn from ground truth if available
        if ground_truth is not None:
            self._learn_from_ground_truth(df, ground_truth, adaptive_fixes)
        
        processing_time = time.time() - start_time
        
        report = CleaningReport(
            total_fixes=len(adaptive_fixes),
            quality_score=self._calculate_adaptive_quality(df, cleaned_df, adaptive_fixes, ground_truth),
            fixes=adaptive_fixes,
            processing_time=processing_time
        )
        
        return cleaned_df, report
    
    def _find_similar_corrections(self, fix: Fix) -> List[Dict[str, Any]]:
        """Find similar corrections from memory."""
        similar = []
        
        for memory_item in self.correction_memory:
            # Simple similarity based on column name and original value pattern
            if (memory_item["column"] == fix.column and
                self._values_similar(memory_item["original_value"], fix.original)):
                similar.append(memory_item)
        
        return similar[:5]  # Return top 5 similar corrections
    
    def _values_similar(self, val1: Any, val2: Any) -> bool:
        """Check if two values are similar enough for learning transfer."""
        str1, str2 = str(val1).lower().strip(), str(val2).lower().strip()
        
        # Exact match
        if str1 == str2:
            return True
        
        # Pattern similarity for common cases
        if len(str1) > 0 and len(str2) > 0:
            # Both are numbers
            try:
                float(str1)
                float(str2)
                return True
            except ValueError:
                pass
            
            # Both are email-like
            if "@" in str1 and "@" in str2:
                return True
            
            # Both are phone-like
            if any(c.isdigit() for c in str1) and any(c.isdigit() for c in str2):
                digit_ratio1 = sum(c.isdigit() for c in str1) / len(str1)
                digit_ratio2 = sum(c.isdigit() for c in str2) / len(str2)
                if digit_ratio1 > 0.5 and digit_ratio2 > 0.5:
                    return True
        
        return False
    
    def _calculate_learned_confidence(self, similar_corrections: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on learned corrections."""
        if not similar_corrections:
            return 0.0
        
        success_rate = sum(1 for c in similar_corrections if c.get("was_correct", False)) / len(similar_corrections)
        avg_confidence = sum(c.get("confidence", 0.5) for c in similar_corrections) / len(similar_corrections)
        
        # Weighted combination
        return min(1.0, success_rate * 0.6 + avg_confidence * 0.4)
    
    def _learn_from_ground_truth(
        self, 
        original_df: pd.DataFrame, 
        ground_truth: pd.DataFrame, 
        fixes: List[Fix]
    ):
        """Learn from ground truth corrections."""
        for fix in fixes:
            if fix.row_index < len(ground_truth) and fix.column in ground_truth.columns:
                actual_correct = ground_truth.iloc[fix.row_index][fix.column]
                was_correct = str(fix.cleaned) == str(actual_correct)
                
                correction = {
                    "column": fix.column,
                    "original_value": fix.original,
                    "predicted_value": fix.cleaned,
                    "corrected_value": actual_correct,
                    "confidence": fix.confidence,
                    "was_correct": was_correct,
                    "timestamp": time.time()
                }
                
                self.correction_memory.append(correction)
                
                # Maintain memory size limit
                if len(self.correction_memory) > self.memory_size:
                    self.correction_memory.pop(0)
    
    def _calculate_adaptive_quality(
        self, 
        original_df: pd.DataFrame, 
        cleaned_df: pd.DataFrame, 
        fixes: List[Fix],
        ground_truth: Optional[pd.DataFrame] = None
    ) -> float:
        """Calculate quality score with adaptive learning bonus."""
        base_quality = sum(f.confidence for f in fixes) / len(fixes) if fixes else 1.0
        
        if ground_truth is not None:
            # Calculate actual accuracy
            correct_fixes = 0
            for fix in fixes:
                if fix.row_index < len(ground_truth) and fix.column in ground_truth.columns:
                    actual_correct = ground_truth.iloc[fix.row_index][fix.column]
                    if str(fix.cleaned) == str(actual_correct):
                        correct_fixes += 1
            
            actual_accuracy = correct_fixes / len(fixes) if fixes else 1.0
            return actual_accuracy
        
        # Add learning bonus
        learning_bonus = min(0.1, len(self.correction_memory) / self.memory_size * 0.1)
        return min(1.0, base_quality + learning_bonus)


class ResearchBenchmarker:
    """Comprehensive benchmarking suite for LLM cleaning algorithms."""
    
    def __init__(self, output_dir: str = "./research_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    async def run_comparative_study(
        self, 
        algorithms: List[ResearchAlgorithm],
        benchmark_suite: BenchmarkSuite,
        runs_per_algorithm: int = 3
    ) -> Dict[str, List[ExperimentResult]]:
        """Run comprehensive comparative study."""
        logger.info(f"Starting comparative study with {len(algorithms)} algorithms on {benchmark_suite.name}")
        
        results = {alg.name: [] for alg in algorithms}
        
        for algorithm in algorithms:
            logger.info(f"Testing algorithm: {algorithm.name}")
            
            for dataset_idx, (dirty_df, clean_df) in enumerate(benchmark_suite.datasets):
                dataset_name = f"{benchmark_suite.name}_dataset_{dataset_idx}"
                
                # Run multiple times for statistical significance
                run_results = []
                
                for run in range(runs_per_algorithm):
                    logger.info(f"  Run {run + 1}/{runs_per_algorithm} on {dataset_name}")
                    
                    try:
                        # Clean the dataset
                        start_time = time.time()
                        cleaned_df, report = await algorithm.clean_async(dirty_df, clean_df)
                        processing_time = time.time() - start_time
                        
                        # Calculate metrics
                        metrics = self._calculate_metrics(dirty_df, cleaned_df, clean_df, report.fixes)
                        
                        result = ExperimentResult(
                            algorithm_name=algorithm.name,
                            dataset_name=dataset_name,
                            accuracy=metrics["accuracy"],
                            precision=metrics["precision"],
                            recall=metrics["recall"],
                            f1_score=metrics["f1_score"],
                            processing_time=processing_time,
                            confidence_distribution=[f.confidence for f in report.fixes],
                            fixes_applied=len([f for f in report.fixes if f.confidence >= 0.85]),
                            metadata={
                                "run": run,
                                "total_fixes": len(report.fixes),
                                "quality_score": report.quality_score
                            }
                        )
                        
                        run_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Algorithm {algorithm.name} failed on {dataset_name}, run {run}: {e}")
                        continue
                
                # Calculate statistical significance
                if len(run_results) > 1:
                    accuracies = [r.accuracy for r in run_results]
                    mean_accuracy = statistics.mean(accuracies)
                    std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
                    
                    # Add statistical significance to results
                    for result in run_results:
                        result.statistical_significance = abs(result.accuracy - mean_accuracy) / (std_accuracy + 1e-8)
                
                results[algorithm.name].extend(run_results)
        
        # Save results
        self._save_results(results, benchmark_suite.name)
        
        # Generate statistical report
        self._generate_statistical_report(results, benchmark_suite.name)
        
        return results
    
    def _calculate_metrics(
        self, 
        dirty_df: pd.DataFrame, 
        cleaned_df: pd.DataFrame, 
        ground_truth: pd.DataFrame,
        fixes: List[Fix]
    ) -> Dict[str, float]:
        """Calculate precision, recall, F1, and accuracy metrics."""
        
        # For each cell, determine if it was correctly fixed
        y_true = []  # 1 if cell needed fixing, 0 otherwise
        y_pred = []  # 1 if cell was fixed, 0 otherwise
        y_correct = []  # 1 if fix was correct, 0 otherwise
        
        for row_idx in range(min(len(dirty_df), len(ground_truth))):
            for col_name in dirty_df.columns:
                if col_name in ground_truth.columns:
                    dirty_value = dirty_df.iloc[row_idx][col_name]
                    true_value = ground_truth.iloc[row_idx][col_name]
                    cleaned_value = cleaned_df.iloc[row_idx][col_name] if row_idx < len(cleaned_df) else dirty_value
                    
                    needs_fix = str(dirty_value) != str(true_value)
                    was_fixed = str(dirty_value) != str(cleaned_value)
                    is_correct = str(cleaned_value) == str(true_value)
                    
                    y_true.append(1 if needs_fix else 0)
                    y_pred.append(1 if was_fixed else 0)
                    y_correct.append(1 if is_correct else 0)
        
        if not y_true:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # Calculate metrics
        accuracy = sum(y_correct) / len(y_correct)
        
        if sum(y_pred) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
        else:
            precision = recall = f1 = 0.0
        
        return {
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
    
    def _save_results(self, results: Dict[str, List[ExperimentResult]], benchmark_name: str):
        """Save results to JSON file."""
        results_file = self.output_dir / f"{benchmark_name}_results.json"
        
        serializable_results = {}
        for alg_name, alg_results in results.items():
            serializable_results[alg_name] = [
                {
                    "algorithm_name": r.algorithm_name,
                    "dataset_name": r.dataset_name,
                    "accuracy": r.accuracy,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1_score": r.f1_score,
                    "processing_time": r.processing_time,
                    "confidence_distribution": r.confidence_distribution,
                    "fixes_applied": r.fixes_applied,
                    "statistical_significance": r.statistical_significance,
                    "metadata": r.metadata
                }
                for r in alg_results
            ]
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def _generate_statistical_report(self, results: Dict[str, List[ExperimentResult]], benchmark_name: str):
        """Generate statistical significance report."""
        report_file = self.output_dir / f"{benchmark_name}_statistical_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Statistical Analysis Report: {benchmark_name}\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics for each algorithm
            f.write("## Algorithm Performance Summary\n\n")
            f.write("| Algorithm | Mean Accuracy | Std Accuracy | Mean F1 | Mean Processing Time |\n")
            f.write("|-----------|---------------|--------------|---------|---------------------|\n")
            
            for alg_name, alg_results in results.items():
                if alg_results:
                    accuracies = [r.accuracy for r in alg_results]
                    f1_scores = [r.f1_score for r in alg_results]
                    times = [r.processing_time for r in alg_results]
                    
                    mean_acc = statistics.mean(accuracies)
                    std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
                    mean_f1 = statistics.mean(f1_scores)
                    mean_time = statistics.mean(times)
                    
                    f.write(f"| {alg_name} | {mean_acc:.3f} | {std_acc:.3f} | {mean_f1:.3f} | {mean_time:.2f}s |\n")
            
            # Statistical significance tests
            f.write("\n## Statistical Significance Analysis\n\n")
            
            algorithm_names = list(results.keys())
            for i in range(len(algorithm_names)):
                for j in range(i + 1, len(algorithm_names)):
                    alg1, alg2 = algorithm_names[i], algorithm_names[j]
                    
                    if results[alg1] and results[alg2]:
                        acc1 = [r.accuracy for r in results[alg1]]
                        acc2 = [r.accuracy for r in results[alg2]]
                        
                        if len(acc1) > 1 and len(acc2) > 1:
                            t_stat, p_value = stats.ttest_ind(acc1, acc2)
                            
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                            
                            f.write(f"**{alg1} vs {alg2}**: t-statistic = {t_stat:.3f}, p-value = {p_value:.4f} {significance}\n\n")
            
            f.write("\n*Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant*\n")
        
        logger.info(f"Statistical report saved to {report_file}")


def create_synthetic_benchmark() -> BenchmarkSuite:
    """Create a synthetic benchmark dataset for testing."""
    np.random.seed(42)  # For reproducibility
    
    datasets = []
    
    # Dataset 1: Email cleaning
    dirty_emails = [
        "john.doe@gmail.com",
        "jane@yahoo",  # Missing .com
        "bob@company.co.uk",
        "alice@domain",  # Missing TLD
        "charlie@test.org",
        "N/A",  # Null indicator
        "unknown@email.com"
    ]
    
    clean_emails = [
        "john.doe@gmail.com",
        "jane@yahoo.com",
        "bob@company.co.uk", 
        "alice@domain.com",
        "charlie@test.org",
        None,
        "unknown@email.com"
    ]
    
    dirty_df1 = pd.DataFrame({"email": dirty_emails, "id": range(len(dirty_emails))})
    clean_df1 = pd.DataFrame({"email": clean_emails, "id": range(len(clean_emails))})
    datasets.append((dirty_df1, clean_df1))
    
    # Dataset 2: Phone number cleaning
    dirty_phones = [
        "123-456-7890",
        "123.456.7890",  # Different format
        "(123) 456-7890",
        "1234567890",  # No formatting
        "123-456-789",  # Too short
        "N/A",
        "555-0123"
    ]
    
    clean_phones = [
        "123-456-7890",
        "123-456-7890",
        "123-456-7890",
        "123-456-7890",
        None,  # Invalid phone
        None,
        "555-0123"
    ]
    
    dirty_df2 = pd.DataFrame({"phone": dirty_phones, "id": range(len(dirty_phones))})
    clean_df2 = pd.DataFrame({"phone": clean_phones, "id": range(len(clean_phones))})
    datasets.append((dirty_df2, clean_df2))
    
    return BenchmarkSuite(
        name="synthetic_benchmark",
        datasets=datasets,
        metadata={
            "description": "Synthetic benchmark for testing LLM cleaning algorithms",
            "created_at": time.time()
        }
    )


async def run_research_study():
    """Run a comprehensive research study comparing different algorithms."""
    logger.info("Starting comprehensive research study")
    
    # Initialize algorithms
    algorithms = [
        EnsembleLLMCleaner(providers=["local"], voting_strategy="majority"),
        EnsembleLLMCleaner(providers=["local"], voting_strategy="confidence_weighted"),
        AdaptiveLLMCleaner(base_provider="local"),
    ]
    
    # Create benchmark
    benchmark = create_synthetic_benchmark()
    
    # Run study
    benchmarker = ResearchBenchmarker()
    results = await benchmarker.run_comparative_study(
        algorithms=algorithms,
        benchmark_suite=benchmark,
        runs_per_algorithm=3
    )
    
    logger.info("Research study completed")
    return results


if __name__ == "__main__":
    asyncio.run(run_research_study())