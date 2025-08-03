"""Data profiling and anomaly detection for table cleaning."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


logger = logging.getLogger(__name__)


@dataclass
class ColumnProfile:
    """Profile information for a single column."""
    name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Statistical measures
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    
    # Pattern analysis
    patterns: List[str] = None
    common_values: List[Tuple[Any, int]] = None
    anomalies: List[Dict[str, Any]] = None
    
    # Quality metrics
    quality_score: float = 0.0
    issues: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.patterns is None:
            self.patterns = []
        if self.common_values is None:
            self.common_values = []
        if self.anomalies is None:
            self.anomalies = []
        if self.issues is None:
            self.issues = []
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class TableProfile:
    """Profile information for entire table."""
    row_count: int
    column_count: int
    columns: Dict[str, ColumnProfile]
    
    # Cross-column analysis
    duplicate_rows: int = 0
    duplicate_percentage: float = 0.0
    relationships: List[Dict[str, Any]] = None
    
    # Overall quality
    overall_quality_score: float = 0.0
    total_issues: int = 0
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.relationships is None:
            self.relationships = []


class DataProfiler:
    """Advanced data profiling and anomaly detection."""
    
    def __init__(self, sample_size: int = 10000):
        """Initialize profiler.
        
        Args:
            sample_size: Maximum number of rows to analyze for patterns
        """
        self.sample_size = sample_size
        
    def profile_table(self, df: pd.DataFrame) -> TableProfile:
        """Generate comprehensive table profile.
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Complete table profile with column-level and table-level analysis
        """
        logger.info(f"Profiling table with {len(df)} rows and {len(df.columns)} columns")
        
        # Sample data if too large
        sample_df = df.head(self.sample_size) if len(df) > self.sample_size else df
        
        # Profile each column
        column_profiles = {}
        for col in df.columns:
            column_profiles[col] = self._profile_column(df[col], col)
        
        # Analyze duplicates
        duplicate_count = len(df) - len(df.drop_duplicates())
        duplicate_percentage = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        
        # Calculate overall quality score
        column_scores = [profile.quality_score for profile in column_profiles.values()]
        overall_quality = np.mean(column_scores) if column_scores else 0.0
        
        # Count total issues
        total_issues = sum(len(profile.issues) for profile in column_profiles.values())
        
        # Analyze relationships
        relationships = self._analyze_relationships(sample_df)
        
        return TableProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns=column_profiles,
            duplicate_rows=duplicate_count,
            duplicate_percentage=duplicate_percentage,
            relationships=relationships,
            overall_quality_score=overall_quality,
            total_issues=total_issues
        )
    
    def _profile_column(self, series: pd.Series, column_name: str) -> ColumnProfile:
        """Profile a single column."""
        # Basic statistics
        null_count = series.isnull().sum()
        null_percentage = (null_count / len(series)) * 100 if len(series) > 0 else 0
        unique_count = series.nunique()
        unique_percentage = (unique_count / len(series)) * 100 if len(series) > 0 else 0
        
        # Infer data type
        data_type = self._infer_data_type(series)
        
        # Statistical measures (for numeric data)
        mean = median = std = None
        min_val = max_val = None
        
        if data_type in ["integer", "float"]:
            numeric_series = pd.to_numeric(series, errors="coerce")
            if not numeric_series.isnull().all():
                mean = float(numeric_series.mean())
                median = float(numeric_series.median())
                std = float(numeric_series.std())
                min_val = numeric_series.min()
                max_val = numeric_series.max()
        else:
            # For non-numeric, get min/max as string lengths or values
            non_null_series = series.dropna()
            if not non_null_series.empty:
                if data_type == "text":
                    min_val = min(len(str(x)) for x in non_null_series)
                    max_val = max(len(str(x)) for x in non_null_series)
                else:
                    min_val = non_null_series.min()
                    max_val = non_null_series.max()
        
        # Pattern analysis
        patterns = self._detect_patterns(series, data_type)
        
        # Common values
        common_values = self._get_common_values(series)
        
        # Anomaly detection
        anomalies = self._detect_anomalies(series, data_type)
        
        # Quality assessment
        quality_score, issues, suggestions = self._assess_quality(
            series, column_name, data_type, null_percentage, anomalies
        )
        
        return ColumnProfile(
            name=column_name,
            data_type=data_type,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            mean=mean,
            median=median,
            std=std,
            min_val=min_val,
            max_val=max_val,
            patterns=patterns,
            common_values=common_values,
            anomalies=anomalies,
            quality_score=quality_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer the data type of a series."""
        # Remove null values for analysis
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        
        # Check for integer
        try:
            pd.to_numeric(non_null, downcast="integer")
            return "integer"
        except (ValueError, TypeError):
            pass
        
        # Check for float
        try:
            numeric_values = pd.to_numeric(non_null, errors="coerce")
            if not numeric_values.isnull().any():
                return "float"
        except (ValueError, TypeError):
            pass
        
        # Check for datetime
        try:
            pd.to_datetime(non_null, errors="raise")
            return "datetime"
        except (ValueError, TypeError):
            pass
        
        # Check for boolean
        if non_null.dtype == bool or all(str(val).lower() in ["true", "false", "0", "1"] for val in non_null.head(100)):
            return "boolean"
        
        # Check for categorical (low cardinality)
        if len(non_null.unique()) / len(non_null) < 0.1 and len(non_null.unique()) < 50:
            return "categorical"
        
        # Check for specific patterns
        sample_values = [str(val) for val in non_null.head(20)]
        
        # Email pattern
        if any("@" in val and "." in val for val in sample_values):
            return "email"
        
        # Phone pattern
        phone_pattern = re.compile(r"[\+]?[\d\s\-\(\)]{7,}")
        if any(phone_pattern.match(val.replace(" ", "")) for val in sample_values):
            return "phone"
        
        # URL pattern
        if any(val.startswith(("http://", "https://", "www.")) for val in sample_values):
            return "url"
        
        # Default to text
        return "text"
    
    def _detect_patterns(self, series: pd.Series, data_type: str) -> List[str]:
        """Detect common patterns in the data."""
        patterns = []
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return patterns
        
        sample_values = [str(val) for val in non_null.head(100)]
        
        # Date patterns
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}",  # MM-DD-YYYY
            r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            if any(re.match(pattern, val) for val in sample_values):
                patterns.append(f"Date format: {pattern}")
        
        # Number patterns
        if data_type in ["integer", "float"]:
            # Check for currencies
            if any("$" in val or "€" in val or "£" in val for val in sample_values):
                patterns.append("Currency values")
            
            # Check for percentages
            if any("%" in val for val in sample_values):
                patterns.append("Percentage values")
        
        # Text patterns
        if data_type == "text":
            # Check capitalization
            if all(val.isupper() for val in sample_values if val):
                patterns.append("All uppercase")
            elif all(val.islower() for val in sample_values if val):
                patterns.append("All lowercase")
            elif all(val.istitle() for val in sample_values if val):
                patterns.append("Title case")
            
            # Check for codes/IDs
            if all(re.match(r"^[A-Z]{2,4}\d+$", val) for val in sample_values[:10] if val):
                patterns.append("Alphanumeric codes")
        
        # Email patterns
        if data_type == "email":
            domains = [val.split("@")[1] for val in sample_values if "@" in val]
            if domains:
                unique_domains = set(domains)
                if len(unique_domains) == 1:
                    patterns.append(f"Single domain: {list(unique_domains)[0]}")
                elif len(unique_domains) < len(domains) * 0.5:
                    patterns.append("Limited domain variety")
        
        return patterns
    
    def _get_common_values(self, series: pd.Series, top_n: int = 10) -> List[Tuple[Any, int]]:
        """Get most common values and their frequencies."""
        value_counts = series.value_counts()
        return [(val, count) for val, count in value_counts.head(top_n).items()]
    
    def _detect_anomalies(self, series: pd.Series, data_type: str) -> List[Dict[str, Any]]:
        """Detect anomalous values in the series."""
        anomalies = []
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return anomalies
        
        # Numeric anomalies
        if data_type in ["integer", "float"]:
            numeric_series = pd.to_numeric(non_null, errors="coerce")
            if not numeric_series.isnull().all():
                # Z-score based outliers
                z_scores = np.abs(stats.zscore(numeric_series))
                outlier_indices = np.where(z_scores > 3)[0]
                
                for idx in outlier_indices[:10]:  # Limit to 10 outliers
                    original_idx = non_null.index[idx]
                    anomalies.append({
                        "value": series.iloc[original_idx],
                        "issue": "Statistical outlier",
                        "z_score": float(z_scores[idx])
                    })
        
        # Text anomalies
        elif data_type == "text":
            lengths = [len(str(val)) for val in non_null]
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            for i, val in enumerate(non_null):
                val_length = len(str(val))
                if abs(val_length - mean_length) > 3 * std_length:
                    anomalies.append({
                        "value": val,
                        "issue": "Unusual length",
                        "length": val_length,
                        "expected_range": f"{mean_length - 2*std_length:.0f}-{mean_length + 2*std_length:.0f}"
                    })
        
        # Email anomalies
        elif data_type == "email":
            email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
            for val in non_null:
                if not email_pattern.match(str(val)):
                    anomalies.append({
                        "value": val,
                        "issue": "Invalid email format"
                    })
        
        # Phone anomalies
        elif data_type == "phone":
            for val in non_null:
                val_str = str(val)
                # Remove common separators
                clean_val = re.sub(r"[\s\-\(\)]", "", val_str)
                if not clean_val.isdigit() or len(clean_val) < 7 or len(clean_val) > 15:
                    anomalies.append({
                        "value": val,
                        "issue": "Invalid phone format"
                    })
        
        # Categorical anomalies (values that appear very rarely)
        if data_type == "categorical":
            value_counts = non_null.value_counts()
            total_count = len(non_null)
            
            for val, count in value_counts.items():
                if count / total_count < 0.01:  # Less than 1% frequency
                    anomalies.append({
                        "value": val,
                        "issue": "Rare categorical value",
                        "frequency": count,
                        "percentage": (count / total_count) * 100
                    })
        
        return anomalies[:20]  # Limit total anomalies
    
    def _assess_quality(
        self, 
        series: pd.Series, 
        column_name: str, 
        data_type: str, 
        null_percentage: float, 
        anomalies: List[Dict[str, Any]]
    ) -> Tuple[float, List[str], List[str]]:
        """Assess overall column quality and generate suggestions."""
        issues = []
        suggestions = []
        
        # Null value issues
        if null_percentage > 50:
            issues.append(f"High null percentage: {null_percentage:.1f}%")
            suggestions.append("Investigate data source for missing value causes")
        elif null_percentage > 20:
            issues.append(f"Moderate null percentage: {null_percentage:.1f}%")
            suggestions.append("Consider imputation strategies for missing values")
        
        # Anomaly issues
        if len(anomalies) > 0:
            issues.append(f"Found {len(anomalies)} anomalous values")
            suggestions.append("Review and clean anomalous values")
        
        # Data type specific issues
        if data_type == "text":
            # Check for potential encoding issues
            non_null = series.dropna()
            if any("�" in str(val) for val in non_null.head(100)):
                issues.append("Potential encoding issues detected")
                suggestions.append("Check data encoding and fix character issues")
        
        elif data_type in ["integer", "float"]:
            # Check for negative values where they might not make sense
            if "id" in column_name.lower() or "count" in column_name.lower():
                numeric_series = pd.to_numeric(series, errors="coerce")
                if (numeric_series < 0).any():
                    issues.append("Negative values in ID/count column")
                    suggestions.append("Verify negative values are valid")
        
        elif data_type == "categorical":
            # Check for too many categories
            unique_count = series.nunique()
            if unique_count > 100:
                issues.append(f"High cardinality: {unique_count} unique values")
                suggestions.append("Consider grouping categories or using text type")
        
        # Calculate quality score
        quality_score = 1.0
        
        # Deduct for null values
        quality_score -= (null_percentage / 100) * 0.5
        
        # Deduct for anomalies
        anomaly_ratio = len(anomalies) / len(series) if len(series) > 0 else 0
        quality_score -= anomaly_ratio * 0.3
        
        # Deduct for issues
        quality_score -= len(issues) * 0.1
        
        # Ensure score is between 0 and 1
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score, issues, suggestions
    
    def _analyze_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze relationships between columns."""
        relationships = []
        
        if len(df.columns) < 2:
            return relationships
        
        # Check for potential foreign key relationships
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 >= col2:  # Avoid duplicates and self-comparison
                    continue
                
                # Check if one column's values are subset of another
                set1 = set(df[col1].dropna().unique())
                set2 = set(df[col2].dropna().unique())
                
                if set1.issubset(set2) and len(set1) > 1:
                    relationships.append({
                        "type": "potential_foreign_key",
                        "child_column": col1,
                        "parent_column": col2,
                        "coverage": len(set1) / len(set2) if set2 else 0
                    })
                elif set2.issubset(set1) and len(set2) > 1:
                    relationships.append({
                        "type": "potential_foreign_key",
                        "child_column": col2,
                        "parent_column": col1,
                        "coverage": len(set2) / len(set1) if set1 else 0
                    })
        
        # Check for highly correlated numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            correlation_matrix = df[numeric_columns].corr()
            
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i >= j:
                        continue
                    
                    correlation = correlation_matrix.loc[col1, col2]
                    if abs(correlation) > 0.8:  # High correlation
                        relationships.append({
                            "type": "high_correlation",
                            "column1": col1,
                            "column2": col2,
                            "correlation": float(correlation)
                        })
        
        return relationships