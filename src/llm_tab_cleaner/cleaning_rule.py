"""Custom cleaning rules and rule sets."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CleaningRule:
    """Individual cleaning rule definition."""
    name: str
    description: str
    examples: Optional[List[Tuple[str, str]]] = None
    pattern: Optional[str] = None
    transform: Optional[str] = None


class RuleSet:
    """Collection of cleaning rules."""
    
    def __init__(self, rules: List[CleaningRule]):
        """Initialize rule set.
        
        Args:
            rules: List of cleaning rules
        """
        self.rules = rules
        self._rules_by_name = {rule.name: rule for rule in rules}
        
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