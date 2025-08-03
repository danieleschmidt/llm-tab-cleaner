"""LLM provider integrations for data cleaning."""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import requests
from anthropic import Anthropic
from openai import OpenAI


logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def clean_value(
        self, 
        value: Any, 
        column_name: str, 
        context: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Clean a single value using LLM.
        
        Args:
            value: The value to clean
            column_name: Name of the column
            context: Additional context (data type, examples, etc.)
            
        Returns:
            Tuple of (cleaned_value, confidence_score)
        """
        pass
    
    @abstractmethod
    def analyze_column(
        self, 
        values: List[Any], 
        column_name: str
    ) -> Dict[str, Any]:
        """Analyze column for patterns and anomalies.
        
        Args:
            values: Sample of column values
            column_name: Name of the column
            
        Returns:
            Analysis results including patterns, anomalies, suggestions
        """
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider for data cleaning."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (if None, uses ANTHROPIC_API_KEY env var)
            model: Model to use for cleaning
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable or api_key parameter required")
            
        self.model = model
        self.client = Anthropic(api_key=self.api_key)
        
    def clean_value(
        self, 
        value: Any, 
        column_name: str, 
        context: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Clean value using Claude."""
        try:
            prompt = self._build_cleaning_prompt(value, column_name, context)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = self._parse_cleaning_response(response.content[0].text)
            return result.get("cleaned_value", value), result.get("confidence", 0.0)
            
        except Exception as e:
            logger.error(f"Error cleaning value with Anthropic: {e}")
            return value, 0.0
    
    def analyze_column(self, values: List[Any], column_name: str) -> Dict[str, Any]:
        """Analyze column using Claude."""
        try:
            sample_values = values[:20]  # Limit sample size
            prompt = self._build_analysis_prompt(sample_values, column_name)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_analysis_response(response.content[0].text)
            
        except Exception as e:
            logger.error(f"Error analyzing column with Anthropic: {e}")
            return {"patterns": [], "anomalies": [], "suggestions": []}
    
    def _build_cleaning_prompt(self, value: Any, column_name: str, context: Dict[str, Any]) -> str:
        """Build prompt for value cleaning."""
        data_type = context.get("data_type", "unknown")
        examples = context.get("examples", [])
        
        prompt = f"""You are a data quality expert. Clean the following value for column '{column_name}' of type '{data_type}'.

Value to clean: "{value}"

Context:
- Column: {column_name}
- Data type: {data_type}
- Sample valid values: {examples[:5] if examples else "None provided"}

Instructions:
1. If the value is already clean and valid, return it unchanged
2. If it needs cleaning, provide the corrected version
3. Consider common data quality issues: formatting, typos, standardization
4. Preserve the original meaning and data type
5. Return confidence score (0.0-1.0) based on certainty of the correction

Respond in JSON format:
{
  "cleaned_value": "corrected value here",
  "confidence": 0.95,
  "reasoning": "brief explanation of changes made"
}"""
        return prompt
    
    def _build_analysis_prompt(self, values: List[Any], column_name: str) -> str:
        """Build prompt for column analysis."""
        values_str = "\n".join([f"- {v}" for v in values])
        
        prompt = f"""Analyze the following column data for quality issues and patterns.

Column: {column_name}
Sample values:
{values_str}

Identify:
1. Data patterns (formats, structures)
2. Quality issues (missing values, inconsistencies, outliers)
3. Standardization opportunities
4. Recommended cleaning actions

Respond in JSON format:
{
  "patterns": ["pattern1", "pattern2"],
  "anomalies": [{"value": "problematic_value", "issue": "description"}],
  "suggestions": ["suggestion1", "suggestion2"],
  "data_type": "inferred_type",
  "quality_score": 0.85
}"""
        return prompt
    
    def _parse_cleaning_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for cleaning."""
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        return {"cleaned_value": response.strip(), "confidence": 0.5}
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for analysis."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Fallback
        return {
            "patterns": [],
            "anomalies": [],
            "suggestions": [],
            "data_type": "unknown",
            "quality_score": 0.5
        }


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for data cleaning."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: Model to use for cleaning
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter required")
            
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
    def clean_value(
        self, 
        value: Any, 
        column_name: str, 
        context: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Clean value using GPT."""
        try:
            prompt = self._build_cleaning_prompt(value, column_name, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.1
            )
            
            result = self._parse_cleaning_response(response.choices[0].message.content)
            return result.get("cleaned_value", value), result.get("confidence", 0.0)
            
        except Exception as e:
            logger.error(f"Error cleaning value with OpenAI: {e}")
            return value, 0.0
    
    def analyze_column(self, values: List[Any], column_name: str) -> Dict[str, Any]:
        """Analyze column using GPT."""
        try:
            sample_values = values[:20]
            prompt = self._build_analysis_prompt(sample_values, column_name)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.1
            )
            
            return self._parse_analysis_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error analyzing column with OpenAI: {e}")
            return {"patterns": [], "anomalies": [], "suggestions": []}
    
    def _build_cleaning_prompt(self, value: Any, column_name: str, context: Dict[str, Any]) -> str:
        """Build prompt for value cleaning."""
        data_type = context.get("data_type", "unknown")
        examples = context.get("examples", [])
        
        prompt = f"""Clean the following data value for quality issues.

Column: {column_name}
Value: "{value}"
Expected type: {data_type}
Valid examples: {examples[:5] if examples else "None"}

Fix issues like:
- Formatting inconsistencies
- Typos and misspellings  
- Missing or invalid data
- Standardization needs

Return JSON:
{{
  "cleaned_value": "corrected value",
  "confidence": 0.95,
  "reasoning": "what was changed"
}}"""
        return prompt
    
    def _build_analysis_prompt(self, values: List[Any], column_name: str) -> str:
        """Build prompt for column analysis."""
        values_str = "\n".join([f"- {v}" for v in values])
        
        prompt = f"""Analyze this column for data quality patterns and issues.

Column: {column_name}
Values:
{values_str}

Return JSON analysis:
{{
  "patterns": ["format patterns found"],
  "anomalies": [{{"value": "bad_value", "issue": "what's wrong"}}],
  "suggestions": ["cleaning recommendations"],
  "data_type": "inferred type",
  "quality_score": 0.85
}}"""
        return prompt
    
    def _parse_cleaning_response(self, response: str) -> Dict[str, Any]:
        """Parse GPT response for cleaning."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {"cleaned_value": response.strip(), "confidence": 0.5}
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse GPT response for analysis."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {
            "patterns": [],
            "anomalies": [],
            "suggestions": [],
            "data_type": "unknown",
            "quality_score": 0.5
        }


class LocalProvider(LLMProvider):
    """Local/mock provider for testing and development."""
    
    def __init__(self):
        """Initialize local provider."""
        self.rules = self._load_default_rules()
    
    def clean_value(
        self, 
        value: Any, 
        column_name: str, 
        context: Dict[str, Any]
    ) -> Tuple[Any, float]:
        """Clean value using local rules."""
        if value is None or value == "":
            return None, 1.0
            
        value_str = str(value).strip()
        
        # Basic cleaning rules
        if value_str.lower() in ["n/a", "na", "null", "none", "missing"]:
            return None, 0.9
            
        # Date standardization
        if "date" in column_name.lower():
            cleaned_date = self._clean_date(value_str)
            if cleaned_date != value_str:
                return cleaned_date, 0.8
                
        # Phone number cleaning
        if "phone" in column_name.lower():
            cleaned_phone = self._clean_phone(value_str)
            if cleaned_phone != value_str:
                return cleaned_phone, 0.85
        
        # Email cleaning
        if "email" in column_name.lower():
            cleaned_email = self._clean_email(value_str)
            if cleaned_email != value_str:
                return cleaned_email, 0.9
        
        return value, 1.0
    
    def analyze_column(self, values: List[Any], column_name: str) -> Dict[str, Any]:
        """Analyze column using basic rules."""
        non_null_values = [v for v in values if v is not None and str(v).strip()]
        
        patterns = []
        anomalies = []
        suggestions = []
        
        if len(non_null_values) == 0:
            return {
                "patterns": ["All null values"],
                "anomalies": [],
                "suggestions": ["Consider removing column or investigating data source"],
                "data_type": "unknown",
                "quality_score": 0.0
            }
        
        # Detect patterns
        if "date" in column_name.lower():
            patterns.append("Date/time column")
            suggestions.append("Standardize date format")
            
        if "email" in column_name.lower():
            patterns.append("Email column")
            for val in non_null_values[:10]:
                if "@" not in str(val):
                    anomalies.append({"value": val, "issue": "Invalid email format"})
                    
        if "phone" in column_name.lower():
            patterns.append("Phone number column")
            suggestions.append("Standardize phone format")
        
        # Calculate quality score
        null_ratio = (len(values) - len(non_null_values)) / len(values)
        quality_score = max(0.0, 1.0 - null_ratio - len(anomalies) / len(values))
        
        return {
            "patterns": patterns,
            "anomalies": anomalies,
            "suggestions": suggestions,
            "data_type": self._infer_type(non_null_values),
            "quality_score": quality_score
        }
    
    def _load_default_rules(self) -> Dict[str, Any]:
        """Load default cleaning rules."""
        return {
            "null_values": ["n/a", "na", "null", "none", "missing", ""],
            "date_patterns": ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"],
            "phone_pattern": r"[\d\s\-\(\)]+",
            "email_pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        }
    
    def _clean_date(self, value: str) -> str:
        """Basic date cleaning."""
        # Simple transformations
        value = value.replace("/", "-").replace(".", "-")
        return value
    
    def _clean_phone(self, value: str) -> str:
        """Basic phone cleaning."""
        # Remove non-digit characters except spaces and dashes
        import re
        cleaned = re.sub(r"[^\d\s\-]", "", value)
        return cleaned.strip()
    
    def _clean_email(self, value: str) -> str:
        """Basic email cleaning."""
        return value.lower().strip()
    
    def _infer_type(self, values: List[Any]) -> str:
        """Infer data type from values."""
        if not values:
            return "unknown"
            
        sample = values[:10]
        
        # Check if all are numbers
        try:
            [float(v) for v in sample]
            return "numeric"
        except (ValueError, TypeError):
            pass
        
        # Check for dates
        if any("date" in str(v).lower() or "/" in str(v) or "-" in str(v) for v in sample):
            return "date"
            
        # Check for emails
        if any("@" in str(v) for v in sample):
            return "email"
            
        return "text"


def get_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Factory function to get LLM provider.
    
    Args:
        provider_name: Name of provider ("anthropic", "openai", "local")
        **kwargs: Additional arguments for provider initialization
        
    Returns:
        Configured LLM provider
    """
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "local": LocalProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    provider_class = providers[provider_name]
    
    # Local provider doesn't take kwargs
    if provider_name == "local":
        return provider_class()
    
    return provider_class(**kwargs)