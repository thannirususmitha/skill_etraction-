"""
Rule Manager Module
===================
Manages skill extraction rules with support for adding, updating, 
removing, and organizing rules dynamically.

Usage:
    from rule_manager import RuleManager
    
    manager = RuleManager('rules/initial_rules.json')
    manager.add_rule(skill="TensorFlow", pattern=r"\\btensorflow\\b", 
                     category="Machine Learning", confidence=0.95)
    manager.save()

Author: Data Engineering Pipeline
Version: 1.0.0
"""

import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RuleValidationError(Exception):
    """Raised when a rule fails validation."""
    pass


class RuleManager:
    """
    Manages skill extraction rules with CRUD operations.
    
    Features:
    - Add/Update/Remove rules dynamically
    - Validate rule patterns
    - Import from JSON/JSONL files
    - Export to various formats
    - Backup and restore functionality
    - Category management
    """
    
    REQUIRED_FIELDS = ['skill', 'pattern', 'category']
    OPTIONAL_FIELDS = {
        'subcategory': 'General',
        'confidence': 0.85,
        'aliases': []
    }
    
    VALID_CATEGORIES = [
        'Python', 'Java', 'Data Science', 'Machine Learning', 
        'Cloud Platforms', 'DevOps', 'Web Development', 'Database',
        'Data Engineering', 'Mobile Development', 'Cybersecurity',
        'Soft Skills', 'Testing & QA', 'Other'
    ]
    
    def __init__(self, rules_file: str, auto_save: bool = False):
        """
        Initialize the rule manager.
        
        Args:
            rules_file: Path to the main rules JSON file
            auto_save: Whether to automatically save after modifications
        """
        self.rules_file = Path(rules_file)
        self.auto_save = auto_save
        self.rules: List[Dict] = []
        self.metadata: Dict = {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'total_rules': 0
        }
        self._skills_index: Dict[str, int] = {}  # skill name -> index in rules list
        
        self._load_or_create()
    
    def _load_or_create(self) -> None:
        """Load existing rules or create new file."""
        if self.rules_file.exists():
            self._load_rules()
        else:
            logger.info(f"Creating new rules file: {self.rules_file}")
            self.rules_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_rules()
    
    def _load_rules(self) -> None:
        """Load rules from the JSON file."""
        try:
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                self.rules = data.get('rules', [])
                self.metadata = {
                    'version': data.get('version', '1.0.0'),
                    'created_at': data.get('created_at', datetime.now().isoformat()),
                    'updated_at': data.get('updated_at', datetime.now().isoformat()),
                    'total_rules': len(self.rules)
                }
            elif isinstance(data, list):
                self.rules = data
            
            # Build index
            self._rebuild_index()
            
            logger.info(f"Loaded {len(self.rules)} rules from {self.rules_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rules file: {e}")
            raise
    
    def _rebuild_index(self) -> None:
        """Rebuild the skills index."""
        self._skills_index = {rule['skill']: i for i, rule in enumerate(self.rules)}
    
    def _save_rules(self) -> None:
        """Save rules to the JSON file."""
        self.metadata['updated_at'] = datetime.now().isoformat()
        self.metadata['total_rules'] = len(self.rules)
        
        # Get unique categories
        categories = sorted(set(r['category'] for r in self.rules))
        
        data = {
            'version': self.metadata['version'],
            'created_at': self.metadata['created_at'],
            'updated_at': self.metadata['updated_at'],
            'total_rules': self.metadata['total_rules'],
            'categories': categories,
            'rules': self.rules
        }
        
        # Ensure directory exists
        self.rules_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.rules_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.rules)} rules to {self.rules_file}")
    
    def save(self) -> None:
        """Manually save rules to file."""
        self._save_rules()
    
    def validate_rule(self, rule: Dict, raise_error: bool = True) -> tuple:
        """
        Validate a rule dictionary.
        
        Args:
            rule: Rule dictionary to validate
            raise_error: Whether to raise exception on invalid rule
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in rule or not rule[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate pattern is valid regex
        if 'pattern' in rule and rule['pattern']:
            try:
                re.compile(rule['pattern'])
            except re.error as e:
                errors.append(f"Invalid regex pattern: {e}")
        
        # Validate confidence range
        if 'confidence' in rule:
            conf = rule['confidence']
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                errors.append(f"Confidence must be between 0 and 1, got: {conf}")
        
        # Validate aliases is a list
        if 'aliases' in rule and not isinstance(rule['aliases'], list):
            errors.append("Aliases must be a list")
        
        is_valid = len(errors) == 0
        error_message = "; ".join(errors) if errors else None
        
        if not is_valid and raise_error:
            raise RuleValidationError(error_message)
        
        return is_valid, error_message
    
    def add_rule(
        self,
        skill: str,
        pattern: str,
        category: str,
        subcategory: str = 'General',
        confidence: float = 0.85,
        aliases: List[str] = None
    ) -> bool:
        """
        Add a new rule.
        
        Args:
            skill: Canonical skill name
            pattern: Regex pattern for matching
            category: Main category
            subcategory: Sub-category (optional)
            confidence: Confidence score 0-1 (optional)
            aliases: List of alternative names (optional)
            
        Returns:
            True if rule was added successfully
        """
        rule = {
            'skill': skill,
            'pattern': pattern,
            'category': category,
            'subcategory': subcategory,
            'confidence': confidence,
            'aliases': aliases or []
        }
        
        self.validate_rule(rule)
        
        # Check for duplicate
        if skill in self._skills_index:
            logger.warning(f"Rule for '{skill}' already exists. Use update_rule() to modify.")
            return False
        
        self.rules.append(rule)
        self._skills_index[skill] = len(self.rules) - 1
        
        logger.info(f"Added rule: {skill} ({category}/{subcategory})")
        
        if self.auto_save:
            self._save_rules()
        
        return True
    
    def add_rule_dict(self, rule: Dict) -> bool:
        """
        Add a rule from a dictionary.
        
        Args:
            rule: Rule dictionary
            
        Returns:
            True if rule was added successfully
        """
        # Apply defaults
        for field, default in self.OPTIONAL_FIELDS.items():
            rule.setdefault(field, default)
        
        return self.add_rule(
            skill=rule['skill'],
            pattern=rule['pattern'],
            category=rule['category'],
            subcategory=rule.get('subcategory', 'General'),
            confidence=rule.get('confidence', 0.85),
            aliases=rule.get('aliases', [])
        )
    
    def add_rules_batch(self, rules: List[Dict], skip_duplicates: bool = True) -> Dict:
        """
        Add multiple rules at once.
        
        Args:
            rules: List of rule dictionaries
            skip_duplicates: Whether to skip duplicate rules
            
        Returns:
            Dictionary with counts of added, skipped, failed rules
        """
        results = {'added': 0, 'skipped': 0, 'failed': 0, 'errors': []}
        
        for rule in rules:
            try:
                # Apply defaults
                for field, default in self.OPTIONAL_FIELDS.items():
                    rule.setdefault(field, default)
                
                is_valid, error = self.validate_rule(rule, raise_error=False)
                
                if not is_valid:
                    results['failed'] += 1
                    results['errors'].append(f"{rule.get('skill', 'unknown')}: {error}")
                    continue
                
                if rule['skill'] in self._skills_index:
                    if skip_duplicates:
                        results['skipped'] += 1
                        continue
                    else:
                        # Update existing
                        self.update_rule(rule['skill'], rule)
                        results['added'] += 1
                        continue
                
                self.rules.append(rule)
                self._skills_index[rule['skill']] = len(self.rules) - 1
                results['added'] += 1
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(str(e))
        
        if self.auto_save and results['added'] > 0:
            self._save_rules()
        
        logger.info(f"Batch add: {results['added']} added, {results['skipped']} skipped, {results['failed']} failed")
        return results
    
    def update_rule(self, skill: str, updates: Dict) -> bool:
        """
        Update an existing rule.
        
        Args:
            skill: Skill name to update
            updates: Dictionary of fields to update
            
        Returns:
            True if rule was updated successfully
        """
        if skill not in self._skills_index:
            logger.warning(f"Rule not found: {skill}")
            return False
        
        idx = self._skills_index[skill]
        rule = self.rules[idx]
        
        # Apply updates
        for key, value in updates.items():
            if key in self.REQUIRED_FIELDS or key in self.OPTIONAL_FIELDS:
                rule[key] = value
        
        # Validate updated rule
        self.validate_rule(rule)
        
        # Handle skill name change
        if 'skill' in updates and updates['skill'] != skill:
            del self._skills_index[skill]
            self._skills_index[updates['skill']] = idx
        
        logger.info(f"Updated rule: {skill}")
        
        if self.auto_save:
            self._save_rules()
        
        return True
    
    def remove_rule(self, skill: str) -> bool:
        """
        Remove a rule by skill name.
        
        Args:
            skill: Skill name to remove
            
        Returns:
            True if rule was removed successfully
        """
        if skill not in self._skills_index:
            logger.warning(f"Rule not found: {skill}")
            return False
        
        idx = self._skills_index[skill]
        del self.rules[idx]
        self._rebuild_index()
        
        logger.info(f"Removed rule: {skill}")
        
        if self.auto_save:
            self._save_rules()
        
        return True
    
    def get_rule(self, skill: str) -> Optional[Dict]:
        """Get a rule by skill name."""
        if skill not in self._skills_index:
            return None
        return self.rules[self._skills_index[skill]]
    
    def find_rules(self, skill: str) -> Optional[Dict]:
        """Alias for get_rule."""
        return self.get_rule(skill)
    
    def get_rules_by_category(self, category: str) -> List[Dict]:
        """Get all rules for a specific category."""
        return [r for r in self.rules if r['category'] == category]
    
    def list_categories(self) -> List[str]:
        """List all categories in current rules."""
        return sorted(set(r['category'] for r in self.rules))
    
    def list_skills(self, category: Optional[str] = None) -> List[str]:
        """List all skill names, optionally filtered by category."""
        if category:
            return [r['skill'] for r in self.rules if r['category'] == category]
        return [r['skill'] for r in self.rules]
    
    def import_from_jsonl(self, filepath: str) -> Dict:
        """
        Import rules from a JSONL file.
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            Import results
        """
        rules = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rules.append(json.loads(line))
        
        return self.add_rules_batch(rules)
    
    def import_from_directory(self, directory: str) -> Dict:
        """
        Import rules from all JSONL files in a directory.
        
        Args:
            directory: Path to directory containing JSONL files
            
        Returns:
            Combined import results
        """
        total_results = {'added': 0, 'skipped': 0, 'failed': 0, 'errors': [], 'files': 0}
        
        dir_path = Path(directory)
        for jsonl_file in sorted(dir_path.glob('*.jsonl')):
            result = self.import_from_jsonl(str(jsonl_file))
            total_results['added'] += result['added']
            total_results['skipped'] += result['skipped']
            total_results['failed'] += result['failed']
            total_results['errors'].extend(result['errors'])
            total_results['files'] += 1
        
        logger.info(f"Imported from {total_results['files']} files: "
                   f"{total_results['added']} added, {total_results['skipped']} skipped")
        
        return total_results
    
    def export_to_jsonl(self, filepath: str, category: Optional[str] = None) -> int:
        """
        Export rules to a JSONL file.
        
        Args:
            filepath: Output file path
            category: Optional category filter
            
        Returns:
            Number of rules exported
        """
        rules = self.get_rules_by_category(category) if category else self.rules
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for rule in rules:
                f.write(json.dumps(rule, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(rules)} rules to {filepath}")
        return len(rules)
    
    def export_by_category(self, output_dir: str) -> Dict[str, int]:
        """
        Export rules to separate JSONL files by category.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Dictionary of category -> count exported
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        for category in self.list_categories():
            filename = category.lower().replace(' ', '_').replace('&', 'and') + '_skills.jsonl'
            filepath = Path(output_dir) / filename
            results[category] = self.export_to_jsonl(str(filepath), category)
        
        return results
    
    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the rules file.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = str(self.rules_file.with_suffix(f'.backup_{timestamp}.json'))
        
        shutil.copy2(self.rules_file, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    def get_statistics(self) -> Dict:
        """Get statistics about the rules."""
        category_counts = {}
        confidence_dist = {'high': 0, 'medium': 0, 'low': 0}
        
        for rule in self.rules:
            cat = rule['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            conf = rule.get('confidence', 0.85)
            if conf >= 0.9:
                confidence_dist['high'] += 1
            elif conf >= 0.8:
                confidence_dist['medium'] += 1
            else:
                confidence_dist['low'] += 1
        
        return {
            'total_rules': len(self.rules),
            'total_categories': len(category_counts),
            'rules_per_category': category_counts,
            'confidence_distribution': confidence_dist,
            'last_updated': self.metadata.get('updated_at')
        }
    
    def __len__(self) -> int:
        return len(self.rules)
    
    def __contains__(self, skill: str) -> bool:
        return skill in self._skills_index


if __name__ == "__main__":
    print("=" * 70)
    print("RULE MANAGER - DEMO")
    print("=" * 70)
    
    # Create a temporary rule manager
    manager = RuleManager('demo_rules.json', auto_save=False)
    
    # Add some rules
    manager.add_rule(
        skill="Python",
        pattern=r"\bpython\b",
        category="Python",
        subcategory="Core Language",
        confidence=0.95,
        aliases=["Python 3"]
    )
    
    manager.add_rule(
        skill="Machine Learning",
        pattern=r"\bmachine\s*learning\b",
        category="Machine Learning",
        subcategory="Core",
        confidence=0.95
    )
    
    # Show statistics
    stats = manager.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total Rules: {stats['total_rules']}")
    print(f"  Categories: {list(stats['rules_per_category'].keys())}")
    
    # Search for a rule
    rule = manager.get_rule("Python")
    if rule:
        print(f"\nPython Rule: {rule}")
    
    # Clean up demo file
    if Path('demo_rules.json').exists():
        os.remove('demo_rules.json')
    
    print("\nDemo completed!")
