"""Core table cleaning functionality."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import jsonpatch

from .llm_providers import get_provider
from .profiler import DataProfiler


logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""
    total_fixes: int
    quality_score: float
    fixes: List["Fix"]
    processing_time: float = 0.0
    profile_summary: Optional[Dict[str, Any]] = None
    audit_trail: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.audit_trail is None:
            self.audit_trail = []


@dataclass 
class Fix:
    """Individual data fix record."""
    column: str
    row_index: int
    original: Any
    cleaned: Any
    confidence: float
    reasoning: Optional[str] = None
    rule_applied: Optional[str] = None


class TableCleaner:
    """Main interface for LLM-powered table cleaning."""
    
    def __init__(
        self,
        llm_provider: str = "anthropic",
        confidence_threshold: float = 0.85,
        rules: Optional["RuleSet"] = None,
        enable_profiling: bool = True,
        max_fixes_per_column: int = 1000,
        **provider_kwargs
    ):
        """Initialize table cleaner.
        
        Args:
            llm_provider: LLM provider ("anthropic", "openai", "local")
            confidence_threshold: Minimum confidence for applying fixes
            rules: Custom cleaning rules
            enable_profiling: Whether to perform data profiling
            max_fixes_per_column: Maximum fixes to attempt per column
            **provider_kwargs: Additional arguments for LLM provider
        """
        self.llm_provider_name = llm_provider
        self.confidence_threshold = confidence_threshold
        self.rules = rules
        self.enable_profiling = enable_profiling
        self.max_fixes_per_column = max_fixes_per_column
        
        # Initialize LLM provider
        self.llm_provider = get_provider(llm_provider, **provider_kwargs)
        
        # Initialize profiler
        self.profiler = DataProfiler() if enable_profiling else None
        
        logger.info(f"Initialized TableCleaner with {llm_provider} provider")
        
    def clean(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        sample_rate: float = 1.0
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean a pandas DataFrame.
        
        Args:
            df: Input DataFrame to clean
            columns: Specific columns to clean (if None, cleans all)
            sample_rate: Fraction of data to process (0.0-1.0)
            
        Returns:
            Tuple of (cleaned_df, cleaning_report)
        """
        start_time = time.time()
        logger.info(f"Starting cleaning process for DataFrame with {len(df)} rows, {len(df.columns)} columns")
        
        # Create working copy
        cleaned_df = df.copy()
        fixes = []
        audit_trail = []
        
        # Profile the data if enabled
        profile_summary = None
        if self.profiler:
            logger.info("Profiling data for quality assessment")
            table_profile = self.profiler.profile_table(df)
            profile_summary = {
                "overall_quality_score": table_profile.overall_quality_score,
                "total_issues": table_profile.total_issues,
                "duplicate_percentage": table_profile.duplicate_percentage,
                "column_quality_scores": {
                    col: profile.quality_score 
                    for col, profile in table_profile.columns.items()
                }
            }
            logger.info(f"Data quality score: {table_profile.overall_quality_score:.2%}")
        
        # Determine columns to clean
        target_columns = columns if columns is not None else df.columns.tolist()
        
        # Sample data if requested
        if sample_rate < 1.0:
            sample_size = int(len(df) * sample_rate)
            sample_indices = df.sample(n=sample_size).index
            logger.info(f"Sampling {sample_size} rows ({sample_rate:.1%}) for cleaning")
        else:
            sample_indices = df.index
        
        # Clean each column
        for column in target_columns:
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in DataFrame, skipping")
                continue
                
            logger.info(f"Cleaning column: {column}")
            column_fixes, column_audit = self._clean_column(
                cleaned_df, column, sample_indices, table_profile.columns.get(column) if self.profiler else None
            )
            
            fixes.extend(column_fixes)
            audit_trail.extend(column_audit)
            
            # Apply fixes to the DataFrame
            for fix in column_fixes:
                if fix.confidence >= self.confidence_threshold:
                    cleaned_df.at[fix.row_index, fix.column] = fix.cleaned
        
        # Calculate final quality score
        final_quality_score = self._calculate_quality_score(df, cleaned_df, fixes)
        
        processing_time = time.time() - start_time
        
        # Create cleaning report
        report = CleaningReport(
            total_fixes=len([f for f in fixes if f.confidence >= self.confidence_threshold]),
            quality_score=final_quality_score,
            fixes=fixes,
            processing_time=processing_time,
            profile_summary=profile_summary,
            audit_trail=audit_trail
        )
        
        logger.info(f"Cleaning completed: {report.total_fixes} fixes applied in {processing_time:.2f}s")
        
        return cleaned_df, report
    
    def _clean_column(
        self, 
        df: pd.DataFrame, 
        column: str, 
        indices: pd.Index,
        column_profile: Optional[Any] = None
    ) -> Tuple[List[Fix], List[Dict[str, Any]]]:
        """Clean a specific column."""
        fixes = []
        audit_trail = []
        
        # Get column data
        series = df.loc[indices, column]
        
        # Build context for LLM
        context = self._build_column_context(series, column, column_profile)
        
        # Analyze column if profiling is disabled but we need basic info
        if not column_profile and self.profiler:
            column_analysis = self.llm_provider.analyze_column(series.dropna().head(20).tolist(), column)
            context.update(column_analysis)
        
        # Process values that need cleaning
        problematic_indices = self._identify_problematic_values(series, context)
        
        # Limit number of fixes per column
        if len(problematic_indices) > self.max_fixes_per_column:
            logger.warning(f"Column {column} has {len(problematic_indices)} issues, limiting to {self.max_fixes_per_column}")
            problematic_indices = problematic_indices[:self.max_fixes_per_column]
        
        for idx in problematic_indices:
            original_value = series.loc[idx]
            
            # Skip if already null
            if pd.isna(original_value):
                continue
            
            try:
                # Get LLM suggestion
                cleaned_value, confidence = self.llm_provider.clean_value(
                    original_value, column, context
                )
                
                # Check if value actually changed
                if cleaned_value != original_value:
                    fix = Fix(
                        column=column,
                        row_index=idx,
                        original=original_value,
                        cleaned=cleaned_value,
                        confidence=confidence,
                        reasoning=f"LLM cleaning with {self.llm_provider_name}",
                        rule_applied="llm_cleaning"
                    )
                    fixes.append(fix)
                    
                    # Create audit entry
                    audit_entry = {
                        "timestamp": time.time(),
                        "column": column,
                        "row_index": int(idx),
                        "operation": "value_cleaning",
                        "original_value": original_value,
                        "new_value": cleaned_value,
                        "confidence": confidence,
                        "provider": self.llm_provider_name,
                        "patch": jsonpatch.make_patch({column: original_value}, {column: cleaned_value}).patch
                    }
                    audit_trail.append(audit_entry)
                    
            except Exception as e:
                logger.error(f"Error cleaning value {original_value} in column {column}: {e}")
                continue
        
        return fixes, audit_trail
    
    def _build_column_context(
        self, 
        series: pd.Series, 
        column_name: str, 
        column_profile: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Build context information for column cleaning."""
        context = {
            "column_name": column_name,
            "data_type": "unknown",
            "examples": [],
            "patterns": [],
            "common_issues": []
        }
        
        # Use profile information if available
        if column_profile:
            context.update({
                "data_type": column_profile.data_type,
                "patterns": column_profile.patterns,
                "common_issues": column_profile.issues,
                "quality_score": column_profile.quality_score
            })
            
            # Add examples from common values
            if column_profile.common_values:
                context["examples"] = [val for val, _ in column_profile.common_values[:5]]
        else:
            # Basic context building
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                context["examples"] = non_null_values.head(5).tolist()
        
        # Add custom rules context if available
        if self.rules:
            applicable_rules = [
                rule for rule in self.rules.rules 
                if any(keyword in column_name.lower() 
                      for keyword in rule.name.lower().split('_'))
            ]
            if applicable_rules:
                context["custom_rules"] = [
                    {
                        "name": rule.name,
                        "description": rule.description,
                        "examples": rule.examples
                    }
                    for rule in applicable_rules
                ]
        
        return context
    
    def _identify_problematic_values(
        self, 
        series: pd.Series, 
        context: Dict[str, Any]
    ) -> List[int]:
        """Identify values that likely need cleaning."""
        problematic_indices = []
        
        for idx, value in series.items():
            if pd.isna(value):
                continue
                
            # Check against known problematic patterns
            value_str = str(value).strip()
            
            # Common null indicators
            if value_str.lower() in ["n/a", "na", "null", "none", "missing", "", "unknown", "tbd", "tba"]:
                problematic_indices.append(idx)
                continue
            
            # Inconsistent formatting based on data type
            data_type = context.get("data_type", "unknown")
            
            if data_type == "email":
                if "@" not in value_str or "." not in value_str:
                    problematic_indices.append(idx)
            elif data_type == "phone":
                clean_phone = ''.join(c for c in value_str if c.isdigit())
                if len(clean_phone) < 7 or len(clean_phone) > 15:
                    problematic_indices.append(idx)
            elif data_type in ["integer", "float"]:
                # Check for non-numeric content in numeric columns
                try:
                    float(value_str.replace(",", "").replace("$", "").replace("%", ""))
                except ValueError:
                    problematic_indices.append(idx)
            elif data_type == "datetime":
                # Check for inconsistent date formats
                import re
                if not any(re.search(pattern, value_str) for pattern in [
                    r"\d{4}-\d{2}-\d{2}",
                    r"\d{2}/\d{2}/\d{4}",
                    r"\d{2}-\d{2}-\d{4}"
                ]):
                    problematic_indices.append(idx)
        
        return problematic_indices
    
    def _calculate_quality_score(
        self, 
        original_df: pd.DataFrame, 
        cleaned_df: pd.DataFrame, 
        fixes: List[Fix]
    ) -> float:
        """Calculate overall quality improvement score."""
        if len(fixes) == 0:
            return 1.0  # No fixes needed means high quality
        
        # Base score on successful fixes above threshold
        successful_fixes = len([f for f in fixes if f.confidence >= self.confidence_threshold])
        total_cells = original_df.shape[0] * original_df.shape[1]
        
        # Calculate improvement ratio
        improvement_ratio = successful_fixes / total_cells if total_cells > 0 else 0
        
        # Average confidence of applied fixes
        applied_fixes = [f for f in fixes if f.confidence >= self.confidence_threshold]
        avg_confidence = sum(f.confidence for f in applied_fixes) / len(applied_fixes) if applied_fixes else 0
        
        # Combined quality score
        quality_score = min(1.0, 0.8 + improvement_ratio * 0.1 + avg_confidence * 0.1)
        
        return quality_score