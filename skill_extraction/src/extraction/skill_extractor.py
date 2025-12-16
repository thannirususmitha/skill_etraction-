"""
Skill Extractor Module
======================
Extracts skills from text using rule-based NER with regex patterns.

Usage:
    from skill_extractor import SkillExtractor
    
    extractor = SkillExtractor()
    extractor.load_rules('rules/')  # Load from directory
    results = extractor.extract_skills("Looking for Python developer with AWS experience")

Author: Data Engineering Pipeline
Version: 1.0.0
"""

import re
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SkillMatch:
    """Represents a matched skill in text."""
    skill: str
    category: str
    subcategory: str
    confidence: float
    start_pos: int
    end_pos: int
    matched_text: str
    aliases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Contains the complete extraction results for a document."""
    document_id: str
    original_text: str
    skills_found: List[SkillMatch]
    unique_skills: Set[str]
    skills_by_category: Dict[str, List[str]]
    total_matches: int
    
    def to_dict(self) -> Dict:
        return {
            'document_id': self.document_id,
            'skills_found': [s.to_dict() for s in self.skills_found],
            'unique_skills': list(self.unique_skills),
            'skills_by_category': self.skills_by_category,
            'total_matches': self.total_matches
        }


class SkillExtractor:
    """
    Rule-based skill extractor using regex patterns.
    
    Features:
    - Load rules from JSON/JSONL files or directories
    - Add/remove rules dynamically
    - Extract skills with confidence scores and positions
    - Filter by category or confidence threshold
    - Batch processing for multiple documents
    - Export results in various formats
    """
    
    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize the skill extractor.
        
        Args:
            rules_path: Path to rules file (JSON/JSONL) or directory containing rule files
        """
        self.rules: List[Dict] = []
        self.compiled_patterns: Dict[str, re.Pattern] = {}
        self.categories: Set[str] = set()
        self.skills_index: Dict[str, Dict] = {}
        
        if rules_path:
            self.load_rules(rules_path)
    
    def load_rules(self, path: str) -> int:
        """
        Load rules from a file or directory.
        
        Args:
            path: Path to JSON file, JSONL file, or directory containing rule files
            
        Returns:
            Number of rules loaded
        """
        path_obj = Path(path)
        rules_loaded = 0
        
        if path_obj.is_dir():
            # Load all JSONL and JSON files from directory
            for jsonl_file in sorted(path_obj.glob('*.jsonl')):
                rules_loaded += self._load_jsonl_file(str(jsonl_file))
            for json_file in sorted(path_obj.glob('*.json')):
                if json_file.name != 'initial_rules.json':  # Skip consolidated file
                    rules_loaded += self._load_json_file(str(json_file))
        elif path_obj.suffix == '.jsonl':
            rules_loaded = self._load_jsonl_file(path)
        elif path_obj.suffix == '.json':
            rules_loaded = self._load_json_file(path)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")
        
        logger.info(f"Loaded {rules_loaded} rules from {path}")
        return rules_loaded
    
    def _load_json_file(self, filepath: str) -> int:
        """Load rules from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            rules = data
        elif isinstance(data, dict) and 'rules' in data:
            rules = data['rules']
        else:
            raise ValueError("Invalid JSON structure for rules file")
        
        count = 0
        for rule in rules:
            if self.add_rule(rule, compile_pattern=True):
                count += 1
        
        return count
    
    def _load_jsonl_file(self, filepath: str) -> int:
        """Load rules from a JSONL file."""
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        rule = json.loads(line)
                        if self.add_rule(rule, compile_pattern=True):
                            count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {filepath}: {e}")
        return count
    
    def add_rule(self, rule: Dict, compile_pattern: bool = True) -> bool:
        """
        Add a single rule to the extractor.
        
        Args:
            rule: Dictionary with keys: skill, pattern, category, subcategory, confidence, aliases
            compile_pattern: Whether to compile the regex pattern immediately
            
        Returns:
            True if rule was added successfully
        """
        required_fields = ['skill', 'pattern', 'category']
        if not all(field in rule for field in required_fields):
            logger.warning(f"Rule missing required fields: {rule.get('skill', 'unknown')}")
            return False
        
        # Set defaults for optional fields
        rule.setdefault('subcategory', 'General')
        rule.setdefault('confidence', 0.85)
        rule.setdefault('aliases', [])
        
        skill_name = rule['skill']
        
        # Check for duplicate - update if exists
        if skill_name in self.skills_index:
            logger.debug(f"Updating existing rule for '{skill_name}'")
            idx = next(i for i, r in enumerate(self.rules) if r['skill'] == skill_name)
            self.rules[idx] = rule
        else:
            self.rules.append(rule)
        
        self.skills_index[skill_name] = rule
        self.categories.add(rule['category'])
        
        if compile_pattern:
            try:
                self.compiled_patterns[skill_name] = re.compile(
                    rule['pattern'], 
                    re.IGNORECASE
                )
            except re.error as e:
                logger.error(f"Invalid regex pattern for '{skill_name}': {e}")
                return False
        
        return True
    
    def remove_rule(self, skill_name: str) -> bool:
        """
        Remove a rule by skill name.
        
        Args:
            skill_name: Name of the skill to remove
            
        Returns:
            True if rule was removed successfully
        """
        if skill_name not in self.skills_index:
            logger.warning(f"Rule not found: {skill_name}")
            return False
        
        self.rules = [r for r in self.rules if r['skill'] != skill_name]
        del self.skills_index[skill_name]
        
        if skill_name in self.compiled_patterns:
            del self.compiled_patterns[skill_name]
        
        # Update categories
        self.categories = set(r['category'] for r in self.rules)
        
        logger.info(f"Removed rule for '{skill_name}'")
        return True
    
    def extract_skills(
        self,
        text: str,
        document_id: str = "doc_0",
        min_confidence: float = 0.0,
        categories: Optional[List[str]] = None,
        return_positions: bool = True,
        deduplicate: bool = True
    ) -> ExtractionResult:
        """
        Extract skills from text.
        
        Args:
            text: Text to extract skills from
            document_id: Identifier for the document
            min_confidence: Minimum confidence threshold (0.0-1.0)
            categories: List of categories to filter (None = all)
            return_positions: Whether to include match positions
            deduplicate: Whether to remove duplicate skill matches
            
        Returns:
            ExtractionResult object containing all matches
        """
        matches: List[SkillMatch] = []
        seen_skills: Set[str] = set()
        
        for rule in self.rules:
            # Apply filters
            if rule['confidence'] < min_confidence:
                continue
            if categories and rule['category'] not in categories:
                continue
            
            skill_name = rule['skill']
            pattern = self.compiled_patterns.get(skill_name)
            
            if not pattern:
                try:
                    pattern = re.compile(rule['pattern'], re.IGNORECASE)
                    self.compiled_patterns[skill_name] = pattern
                except re.error:
                    continue
            
            # Find all matches
            for match in pattern.finditer(text):
                if deduplicate and skill_name in seen_skills:
                    continue
                
                skill_match = SkillMatch(
                    skill=skill_name,
                    category=rule['category'],
                    subcategory=rule.get('subcategory', 'General'),
                    confidence=rule['confidence'],
                    start_pos=match.start() if return_positions else -1,
                    end_pos=match.end() if return_positions else -1,
                    matched_text=match.group(),
                    aliases=rule.get('aliases', [])
                )
                matches.append(skill_match)
                seen_skills.add(skill_name)
                
                if deduplicate:
                    break
        
        # Group by category
        skills_by_category = defaultdict(list)
        for m in matches:
            if m.skill not in skills_by_category[m.category]:
                skills_by_category[m.category].append(m.skill)
        
        return ExtractionResult(
            document_id=document_id,
            original_text=text[:500] + "..." if len(text) > 500 else text,
            skills_found=matches,
            unique_skills=seen_skills,
            skills_by_category=dict(skills_by_category),
            total_matches=len(matches)
        )
    
    def extract_batch(
        self,
        documents: List[Tuple[str, str]],
        show_progress: bool = True,
        **kwargs
    ) -> List[ExtractionResult]:
        """
        Extract skills from multiple documents.
        
        Args:
            documents: List of (document_id, text) tuples
            show_progress: Whether to show progress
            **kwargs: Arguments passed to extract_skills()
            
        Returns:
            List of ExtractionResult objects
        """
        results = []
        total = len(documents)
        
        for i, (doc_id, text) in enumerate(documents):
            if show_progress and (i + 1) % 500 == 0:
                logger.info(f"Processing document {i + 1}/{total}")
            
            result = self.extract_skills(text, document_id=doc_id, **kwargs)
            results.append(result)
        
        if show_progress:
            logger.info(f"Completed processing {total} documents")
        
        return results
    
    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract skills from a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            id_column: Name of column containing document IDs (optional)
            **kwargs: Arguments passed to extract_skills()
            
        Returns:
            DataFrame with extracted skills
        """
        results = []
        
        for idx, row in df.iterrows():
            doc_id = str(row[id_column]) if id_column else f"doc_{idx}"
            text = str(row[text_column])
            
            extraction = self.extract_skills(text, document_id=doc_id, **kwargs)
            
            results.append({
                'document_id': doc_id,
                'unique_skills': list(extraction.unique_skills),
                'skill_count': extraction.total_matches,
                'skills_by_category': extraction.skills_by_category,
                'skills_list': [m.skill for m in extraction.skills_found]
            })
        
        return pd.DataFrame(results)
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded rules."""
        category_counts = defaultdict(int)
        subcategory_counts = defaultdict(lambda: defaultdict(int))
        confidence_dist = {'high (>=0.9)': 0, 'medium (0.8-0.9)': 0, 'low (<0.8)': 0}
        
        for rule in self.rules:
            category_counts[rule['category']] += 1
            subcategory_counts[rule['category']][rule.get('subcategory', 'General')] += 1
            
            conf = rule['confidence']
            if conf >= 0.9:
                confidence_dist['high (>=0.9)'] += 1
            elif conf >= 0.8:
                confidence_dist['medium (0.8-0.9)'] += 1
            else:
                confidence_dist['low (<0.8)'] += 1
        
        return {
            'total_rules': len(self.rules),
            'total_categories': len(self.categories),
            'categories': list(self.categories),
            'rules_per_category': dict(category_counts),
            'confidence_distribution': confidence_dist
        }
    
    def save_rules(self, filepath: str, format: str = 'json') -> None:
        """
        Save current rules to a file.
        
        Args:
            filepath: Output file path
            format: 'json' or 'jsonl'
        """
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'version': '1.0',
                    'total_rules': len(self.rules),
                    'categories': sorted(list(self.categories)),
                    'rules': self.rules
                }, f, indent=2, ensure_ascii=False)
        elif format == 'jsonl':
            with open(filepath, 'w', encoding='utf-8') as f:
                for rule in self.rules:
                    f.write(json.dumps(rule, ensure_ascii=False) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(self.rules)} rules to {filepath}")
    
    def search_rules(
        self,
        query: str,
        search_in: List[str] = ['skill', 'category', 'subcategory']
    ) -> List[Dict]:
        """
        Search for rules matching a query.
        
        Args:
            query: Search query (case-insensitive)
            search_in: Fields to search in
            
        Returns:
            List of matching rules
        """
        query_lower = query.lower()
        matches = []
        
        for rule in self.rules:
            for field in search_in:
                if field in rule and query_lower in str(rule[field]).lower():
                    matches.append(rule)
                    break
        
        return matches
    
    def get_rules_by_category(self, category: str) -> List[Dict]:
        """Get all rules for a specific category."""
        return [r for r in self.rules if r['category'] == category]
    
    def list_categories(self) -> List[str]:
        """List all available categories."""
        return sorted(list(self.categories))


# Convenience function for quick extraction
def extract_skills_from_text(
    text: str,
    rules_path: str,
    **kwargs
) -> ExtractionResult:
    """
    Quick extraction from text using rules file.
    
    Args:
        text: Text to extract skills from
        rules_path: Path to rules file or directory
        **kwargs: Additional arguments for extract_skills()
        
    Returns:
        ExtractionResult object
    """
    extractor = SkillExtractor(rules_path)
    return extractor.extract_skills(text, **kwargs)


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("SKILL EXTRACTOR - DEMO")
    print("=" * 70)
    
    sample_text = """
    We are looking for a Senior Data Engineer with strong experience in Python, 
    Apache Spark, and AWS. The ideal candidate should have expertise in building 
    ETL pipelines using Airflow and be familiar with machine learning concepts. 
    Experience with Docker, Kubernetes, and CI/CD practices is required. 
    SQL and PostgreSQL expertise required. Knowledge of Snowflake and dbt is a plus.
    Strong communication skills and ability to work in an Agile environment.
    """
    
    # Create extractor
    extractor = SkillExtractor()
    
    # Try to load rules from the rules directory
    rules_dir = Path(__file__).parent.parent / 'rules'
    if rules_dir.exists():
        extractor.load_rules(str(rules_dir))
    else:
        # Add sample rules for demo
        sample_rules = [
            {"skill": "Python", "pattern": r"\bpython\b", "category": "Python", "subcategory": "Core", "confidence": 0.95},
            {"skill": "Apache Spark", "pattern": r"\bspark\b", "category": "Data Engineering", "subcategory": "Processing", "confidence": 0.95},
            {"skill": "AWS", "pattern": r"\baws\b", "category": "Cloud Platforms", "subcategory": "Provider", "confidence": 0.95},
            {"skill": "ETL", "pattern": r"\betl\b", "category": "Data Engineering", "subcategory": "Process", "confidence": 0.90},
            {"skill": "Airflow", "pattern": r"\bairflow\b", "category": "Data Engineering", "subcategory": "Orchestration", "confidence": 0.95},
            {"skill": "Docker", "pattern": r"\bdocker\b", "category": "DevOps", "subcategory": "Container", "confidence": 0.95},
            {"skill": "Kubernetes", "pattern": r"\bkubernetes\b", "category": "DevOps", "subcategory": "Orchestration", "confidence": 0.95},
            {"skill": "SQL", "pattern": r"\bsql\b", "category": "Database", "subcategory": "Language", "confidence": 0.90},
            {"skill": "PostgreSQL", "pattern": r"\bpostgresql\b", "category": "Database", "subcategory": "RDBMS", "confidence": 0.95},
        ]
        for rule in sample_rules:
            extractor.add_rule(rule)
    
    # Extract skills
    result = extractor.extract_skills(sample_text)
    
    # Display results
    print(f"\nSample Text:\n{sample_text[:200]}...")
    print(f"\n{'=' * 70}")
    print(f"EXTRACTION RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Skills Found: {result.total_matches}")
    print(f"\nSkills by Category:")
    for category, skills in sorted(result.skills_by_category.items()):
        print(f"  {category}: {', '.join(skills)}")
    
    print(f"\nDetailed Matches:")
    for match in result.skills_found[:10]:
        print(f"  - {match.skill} ({match.category}/{match.subcategory}) "
              f"[confidence: {match.confidence}]")
    
    # Show statistics
    stats = extractor.get_statistics()
    print(f"\n{'=' * 70}")
    print(f"EXTRACTOR STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total Rules: {stats['total_rules']}")
    print(f"Categories: {stats['total_categories']}")
    print(f"Rules per Category: {stats['rules_per_category']}")
