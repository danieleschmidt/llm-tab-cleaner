"""Custom cleaning rules and rule sets."""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class CleaningRule:
    """Individual cleaning rule definition."""
    name: str
    description: str
    examples: Optional[List[Tuple[str, str]]] = None
    pattern: Optional[str] = None
    transform: Optional[str] = None
    function: Optional[Callable[[Any], Tuple[Any, float]]] = None
    confidence: float = 0.9
    column_patterns: Optional[List[str]] = None
    data_types: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate rule configuration."""
        if not any([self.pattern, self.transform, self.function, self.examples]):
            raise ValueError(f"Rule '{self.name}' must have at least one of: pattern, transform, function, or examples")
    
    def applies_to_column(self, column_name: str, data_type: str = None) -> bool:
        """Check if rule applies to given column."""
        # Check column name patterns
        if self.column_patterns:
            if not any(pattern.lower() in column_name.lower() for pattern in self.column_patterns):
                return False
        
        # Check data types
        if self.data_types:
            if data_type and data_type not in self.data_types:
                return False
        
        return True
    
    def apply(self, value: Any, context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """Apply the cleaning rule to a value."""
        if pd.isna(value):
            return value, 1.0
        
        value_str = str(value).strip()
        
        # Apply custom function if provided
        if self.function:
            try:
                return self.function(value)
            except Exception as e:
                logger.error(f"Error applying custom function in rule '{self.name}': {e}")
                return value, 0.0
        
        # Apply pattern-based transformation
        if self.pattern and self.transform:
            try:
                if re.search(self.pattern, value_str):
                    # Simple string replacement for now
                    if self.transform.startswith("replace:"):
                        _, replacement = self.transform.split(":", 1)
                        cleaned = re.sub(self.pattern, replacement, value_str)
                        return cleaned, self.confidence
                    else:
                        # Apply transformation logic
                        cleaned = self._apply_transform(value_str, self.transform)
                        return cleaned, self.confidence
            except Exception as e:
                logger.error(f"Error applying pattern rule '{self.name}': {e}")
                return value, 0.0
        
        # Apply example-based matching
        if self.examples:
            for original, corrected in self.examples:
                if str(value).lower().strip() == str(original).lower().strip():
                    return corrected, self.confidence
        
        return value, 1.0  # No change needed
    
    def _apply_transform(self, value: str, transform: str) -> str:
        """Apply transformation string to value."""
        if transform == "lowercase":
            return value.lower()
        elif transform == "uppercase":
            return value.upper()
        elif transform == "title_case":
            return value.title()
        elif transform == "strip_whitespace":
            return value.strip()
        elif transform == "remove_special_chars":
            return re.sub(r"[^a-zA-Z0-9\s]", "", value)
        elif transform == "standardize_phone":
            digits = re.sub(r"[^\d]", "", value)
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == "1":
                return f"1-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
            return value
        elif transform == "standardize_email":
            return value.lower().strip()
        elif transform.startswith("replace:"):
            _, replacement = transform.split(":", 1)
            return replacement
        else:
            logger.warning(f"Unknown transform: {transform}")
            return value


class RuleSet:
    """Collection of cleaning rules with execution engine."""
    
    def __init__(self, rules: List[CleaningRule] = None):
        """Initialize rule set.
        
        Args:
            rules: List of cleaning rules
        """
        self.rules = rules or []
        self._rules_by_name = {rule.name: rule for rule in self.rules}
        
    def get_rule(self, name: str) -> Optional[CleaningRule]:
        """Get rule by name."""
        return self._rules_by_name.get(name)
        
    def add_rule(self, rule: CleaningRule) -> None:
        """Add a new rule."""
        self.rules.append(rule)
        self._rules_by_name[rule.name] = rule
        
    def remove_rule(self, name: str) -> bool:
        """Remove rule by name."""
        if name in self._rules_by_name:
            rule = self._rules_by_name.pop(name)
            self.rules.remove(rule)
            return True
        return False
    
    def get_applicable_rules(self, column_name: str, data_type: str = None) -> List[CleaningRule]:
        """Get rules that apply to a specific column."""
        return [rule for rule in self.rules if rule.applies_to_column(column_name, data_type)]
    
    def apply_rules(
        self, 
        value: Any, 
        column_name: str, 
        data_type: str = None,
        context: Dict[str, Any] = None
    ) -> Tuple[Any, float, List[str]]:
        """Apply all applicable rules to a value."""
        applicable_rules = self.get_applicable_rules(column_name, data_type)
        
        if not applicable_rules:
            return value, 1.0, []
        
        current_value = value
        max_confidence = 0.0
        applied_rules = []
        
        for rule in applicable_rules:
            cleaned_value, confidence = rule.apply(current_value, context)
            
            if cleaned_value != current_value and confidence > max_confidence:
                current_value = cleaned_value
                max_confidence = confidence
                applied_rules = [rule.name]
            elif cleaned_value != current_value and confidence == max_confidence:
                applied_rules.append(rule.name)
        
        return current_value, max_confidence, applied_rules


def create_default_rules() -> RuleSet:
    """Create a set of common cleaning rules."""
    rules = [
        # Null value standardization
        CleaningRule(
            name="standardize_nulls",
            description="Convert common null representations to None",
            examples=[
                ("N/A", None),
                ("n/a", None),
                ("NULL", None),
                ("null", None),
                ("None", None),
                ("none", None),
                ("missing", None),
                ("MISSING", None),
                ("", None),
                ("   ", None),
                ("unknown", None),
                ("UNKNOWN", None),
                ("TBD", None),
                ("TBA", None)
            ],
            confidence=0.95
        ),
        
        # Email standardization
        CleaningRule(
            name="standardize_email",
            description="Standardize email format",
            transform="standardize_email",
            column_patterns=["email", "mail"],
            data_types=["email", "text"],
            confidence=0.9
        ),
        
        # Phone number standardization
        CleaningRule(
            name="standardize_phone",
            description="Standardize phone number format",
            transform="standardize_phone",
            column_patterns=["phone", "tel", "mobile", "cell"],
            data_types=["phone", "text"],
            confidence=0.85
        ),
        
        # State code standardization
        CleaningRule(
            name="standardize_state_codes",
            description="Convert state names to 2-letter codes",
            examples=[
                ("California", "CA"),
                ("california", "CA"),
                ("New York", "NY"),
                ("new york", "NY"),
                ("Texas", "TX"),
                ("texas", "TX"),
                ("Florida", "FL"),
                ("florida", "FL"),
                ("Illinois", "IL"),
                ("illinois", "IL"),
                ("Pennsylvania", "PA"),
                ("pennsylvania", "PA"),
                ("Ohio", "OH"),
                ("ohio", "OH"),
                ("Georgia", "GA"),
                ("georgia", "GA"),
                ("North Carolina", "NC"),
                ("north carolina", "NC"),
                ("Michigan", "MI"),
                ("michigan", "MI")
            ],
            column_patterns=["state", "st"],
            confidence=0.9
        ),
        
        # Boolean standardization
        CleaningRule(
            name="standardize_boolean",
            description="Standardize boolean values",
            examples=[
                ("yes", True),
                ("Yes", True),
                ("YES", True),
                ("y", True),
                ("Y", True),
                ("true", True),
                ("True", True),
                ("TRUE", True),
                ("1", True),
                ("no", False),
                ("No", False),
                ("NO", False),
                ("n", False),
                ("N", False),
                ("false", False),
                ("False", False),
                ("FALSE", False),
                ("0", False)
            ],
            data_types=["boolean"],
            confidence=0.95
        ),
        
        # Whitespace cleanup
        CleaningRule(
            name="trim_whitespace",
            description="Remove leading and trailing whitespace",
            pattern=r"^\s+|\s+$",
            transform="strip_whitespace",
            confidence=0.99
        ),
        
        # Currency standardization
        CleaningRule(
            name="standardize_currency",
            description="Clean currency values",
            pattern=r"[\$,]",
            transform="replace:",
            column_patterns=["price", "cost", "amount", "salary", "wage", "fee"],
            data_types=["float", "integer"],
            confidence=0.85
        ),
        
        # Date format standardization (basic)
        CleaningRule(
            name="standardize_dates_basic",
            description="Basic date format cleanup",
            pattern=r"(\d{1,2})/(\d{1,2})/(\d{4})",
            transform="replace:\\3-\\1-\\2",  # MM/DD/YYYY -> YYYY-MM-DD
            column_patterns=["date", "created", "updated", "birth", "dob"],
            data_types=["datetime", "date"],
            confidence=0.8
        )
    ]
    
    return RuleSet(rules)


def create_custom_rule(
    name: str,
    description: str,
    cleaning_function: Callable[[Any], Tuple[Any, float]],
    column_patterns: List[str] = None,
    data_types: List[str] = None,
    confidence: float = 0.8
) -> CleaningRule:
    """Helper function to create custom cleaning rules."""
    return CleaningRule(
        name=name,
        description=description,
        function=cleaning_function,
        column_patterns=column_patterns,
        data_types=data_types,
        confidence=confidence
    )