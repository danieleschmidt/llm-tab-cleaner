"""Tests for cleaning rules and rule sets."""

import pytest
from llm_tab_cleaner import CleaningRule, RuleSet


class TestCleaningRule:
    """Test cases for CleaningRule class."""
    
    def test_basic_rule_creation(self):
        """Test creating a basic cleaning rule."""
        rule = CleaningRule(
            name="test_rule",
            description="A test rule"
        )
        assert rule.name == "test_rule"
        assert rule.description == "A test rule"
        assert rule.examples is None
        assert rule.pattern is None
        assert rule.transform is None
        
    def test_rule_with_examples(self):
        """Test creating a rule with examples."""
        examples = [("CA", "California"), ("NY", "New York")]
        rule = CleaningRule(
            name="state_expansion",
            description="Expand state codes",
            examples=examples
        )
        assert rule.examples == examples


class TestRuleSet:
    """Test cases for RuleSet class."""
    
    def test_empty_ruleset(self):
        """Test creating an empty rule set."""
        ruleset = RuleSet([])
        assert len(ruleset.rules) == 0
        
    def test_ruleset_with_rules(self):
        """Test creating a rule set with rules."""
        rule1 = CleaningRule("rule1", "First rule")
        rule2 = CleaningRule("rule2", "Second rule")
        
        ruleset = RuleSet([rule1, rule2])
        assert len(ruleset.rules) == 2
        assert ruleset.get_rule("rule1") == rule1
        assert ruleset.get_rule("rule2") == rule2
        
    def test_add_rule(self):
        """Test adding a rule to the set."""
        ruleset = RuleSet([])
        rule = CleaningRule("new_rule", "A new rule")
        
        ruleset.add_rule(rule)
        assert len(ruleset.rules) == 1
        assert ruleset.get_rule("new_rule") == rule
        
    def test_remove_rule(self):
        """Test removing a rule from the set."""
        rule = CleaningRule("temp_rule", "Temporary rule")
        ruleset = RuleSet([rule])
        
        result = ruleset.remove_rule("temp_rule")
        assert result is True
        assert len(ruleset.rules) == 0
        assert ruleset.get_rule("temp_rule") is None
        
    def test_remove_nonexistent_rule(self):
        """Test removing a rule that doesn't exist."""
        ruleset = RuleSet([])
        result = ruleset.remove_rule("nonexistent")
        assert result is False