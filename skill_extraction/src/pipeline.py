#!/usr/bin/env python3
"""
Skill Extraction Pipeline
=========================
Main entry point for the skill extraction system.
Provides CLI interface for extracting skills from job descriptions.

Usage:
    # Extract from CSV file
    python pipeline.py --input job_descriptions.csv --output results.json
    
    # Add new rule
    python pipeline.py --add-rule
    
    # List categories
    python pipeline.py --list-categories
    
    # Show statistics
    python pipeline.py --stats

Author: Data Engineering Pipeline
Version: 1.0.0
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
from skill_extractor import SkillExtractor, ExtractionResult
from rule_manager import RuleManager


class SkillExtractionPipeline:
    """
    Main pipeline for skill extraction from job descriptions.
    
    Features:
    - Load rules from JSON/JSONL files
    - Extract skills from CSV/Excel files
    - Add new rules interactively or programmatically
    - Export results in multiple formats
    - Generate statistics and reports
    """
    
    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            rules_path: Path to rules file or directory (default: rules/)
        """
        self.base_dir = Path(__file__).parent
        self.rules_dir = self.base_dir / 'rules'
        self.output_dir = self.base_dir / 'output'
        
        # Create output directory if needed
        self.output_dir.mkdir(exist_ok=True)
        
        # Determine rules path
        if rules_path:
            self.rules_path = Path(rules_path)
        else:
            # Default to initial_rules.json or rules directory
            initial_rules = self.rules_dir / 'initial_rules.json'
            self.rules_path = initial_rules if initial_rules.exists() else self.rules_dir
        
        # Initialize components
        self.extractor = SkillExtractor()
        self.manager = None
        
        # Load rules
        if self.rules_path.exists():
            self.extractor.load_rules(str(self.rules_path))
            
            # Initialize manager with initial_rules.json
            initial_rules = self.rules_dir / 'initial_rules.json'
            if initial_rules.exists():
                self.manager = RuleManager(str(initial_rules), auto_save=False)
    
    def extract_from_file(
        self,
        input_file: str,
        text_column: str = 'cleaned_description',
        id_column: str = 'job_id',
        output_file: Optional[str] = None,
        output_format: str = 'json',
        min_confidence: float = 0.0,
        categories: Optional[list] = None
    ) -> dict:
        """
        Extract skills from a CSV or Excel file.
        
        Args:
            input_file: Path to input file (CSV or Excel)
            text_column: Name of column containing text
            id_column: Name of column containing document IDs
            output_file: Path to output file (optional)
            output_format: Output format ('json', 'csv', 'detailed')
            min_confidence: Minimum confidence threshold
            categories: List of categories to filter
            
        Returns:
            Summary dictionary
        """
        # Load data
        input_path = Path(input_file)
        if input_path.suffix == '.xlsx':
            df = pd.read_excel(input_file)
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        print(f"Loaded {len(df)} records from {input_file}")
        
        # Verify columns exist
        if text_column not in df.columns:
            # Try alternative column names
            alternatives = ['jd', 'description', 'job_description', 'text', 'content']
            for alt in alternatives:
                if alt in df.columns:
                    text_column = alt
                    break
            else:
                raise ValueError(f"Text column '{text_column}' not found. Available: {list(df.columns)}")
        
        if id_column not in df.columns:
            df['doc_id'] = range(len(df))
            id_column = 'doc_id'
        
        # Prepare documents
        documents = []
        for idx, row in df.iterrows():
            doc_id = str(row[id_column])
            text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            if text.strip():
                documents.append((doc_id, text))
        
        print(f"Processing {len(documents)} valid documents...")
        
        # Extract skills
        results = self.extractor.extract_batch(
            documents,
            min_confidence=min_confidence,
            categories=categories,
            show_progress=True
        )
        
        # Generate summary
        summary = self._generate_summary(results)
        
        # Save results
        if output_file:
            self._save_results(results, output_file, output_format)
        else:
            # Default output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f'extraction_results_{timestamp}.{output_format}'
            self._save_results(results, str(output_file), output_format)
        
        return summary
    
    def _generate_summary(self, results: list) -> dict:
        """Generate extraction summary."""
        all_skills = set()
        skills_count = {}
        category_skills = {}
        doc_skill_counts = []
        
        for result in results:
            doc_skill_counts.append(result.total_matches)
            
            for skill in result.unique_skills:
                all_skills.add(skill)
                skills_count[skill] = skills_count.get(skill, 0) + 1
            
            for category, skills in result.skills_by_category.items():
                if category not in category_skills:
                    category_skills[category] = set()
                category_skills[category].update(skills)
        
        # Top skills
        top_skills = sorted(skills_count.items(), key=lambda x: -x[1])[:20]
        
        summary = {
            'total_documents': len(results),
            'unique_skills_found': len(all_skills),
            'total_skill_mentions': sum(skills_count.values()),
            'avg_skills_per_document': sum(doc_skill_counts) / len(doc_skill_counts) if doc_skill_counts else 0,
            'categories_found': len(category_skills),
            'skills_by_category': {k: len(v) for k, v in category_skills.items()},
            'top_20_skills': top_skills
        }
        
        # Print summary
        print("\n" + "=" * 70)
        print("EXTRACTION SUMMARY")
        print("=" * 70)
        print(f"Documents Processed: {summary['total_documents']}")
        print(f"Unique Skills Found: {summary['unique_skills_found']}")
        print(f"Total Skill Mentions: {summary['total_skill_mentions']}")
        print(f"Avg Skills/Document: {summary['avg_skills_per_document']:.2f}")
        print(f"\nSkills by Category:")
        for cat, count in sorted(summary['skills_by_category'].items()):
            print(f"  {cat}: {count} unique skills")
        print(f"\nTop 10 Skills:")
        for skill, count in top_skills[:10]:
            print(f"  {skill}: {count} mentions")
        
        return summary
    
    def _save_results(self, results: list, output_file: str, format: str) -> None:
        """Save extraction results to file."""
        output_path = Path(output_file)
        
        if format == 'json':
            output_data = {
                'extraction_timestamp': datetime.now().isoformat(),
                'total_documents': len(results),
                'results': [r.to_dict() for r in results]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
        
        elif format == 'csv':
            rows = []
            for result in results:
                rows.append({
                    'document_id': result.document_id,
                    'skill_count': result.total_matches,
                    'skills': ', '.join(result.unique_skills),
                    'categories': ', '.join(result.skills_by_category.keys())
                })
            pd.DataFrame(rows).to_csv(output_path, index=False)
        
        elif format == 'detailed':
            rows = []
            for result in results:
                for match in result.skills_found:
                    rows.append({
                        'document_id': result.document_id,
                        'skill': match.skill,
                        'category': match.category,
                        'subcategory': match.subcategory,
                        'confidence': match.confidence,
                        'matched_text': match.matched_text
                    })
            pd.DataFrame(rows).to_csv(output_path, index=False)
        
        print(f"\nResults saved to: {output_path}")
    
    def add_rule_interactive(self) -> bool:
        """Add a new rule interactively."""
        print("\n" + "=" * 50)
        print("ADD NEW RULE")
        print("=" * 50)
        
        # Get skill name
        skill = input("Skill name: ").strip()
        if not skill:
            print("Skill name is required!")
            return False
        
        # Get pattern
        default_pattern = f"\\b{skill.lower()}\\b"
        pattern_input = input(f"Regex pattern [{default_pattern}]: ").strip()
        pattern = pattern_input if pattern_input else default_pattern
        
        # Validate pattern
        import re
        try:
            re.compile(pattern)
        except re.error as e:
            print(f"Invalid regex pattern: {e}")
            return False
        
        # Get category
        print(f"\nAvailable categories: {', '.join(self.manager.VALID_CATEGORIES)}")
        category = input("Category: ").strip()
        if not category:
            print("Category is required!")
            return False
        
        # Get subcategory
        subcategory = input("Subcategory [General]: ").strip() or 'General'
        
        # Get confidence
        try:
            conf_input = input("Confidence (0.0-1.0) [0.85]: ").strip()
            confidence = float(conf_input) if conf_input else 0.85
            if not 0 <= confidence <= 1:
                raise ValueError()
        except ValueError:
            print("Invalid confidence value!")
            return False
        
        # Get aliases
        aliases_input = input("Aliases (comma-separated) []: ").strip()
        aliases = [a.strip() for a in aliases_input.split(',') if a.strip()] if aliases_input else []
        
        # Add rule
        if self.manager:
            success = self.manager.add_rule(
                skill=skill,
                pattern=pattern,
                category=category,
                subcategory=subcategory,
                confidence=confidence,
                aliases=aliases
            )
            
            if success:
                self.manager.save()
                # Reload extractor
                self.extractor.add_rule({
                    'skill': skill,
                    'pattern': pattern,
                    'category': category,
                    'subcategory': subcategory,
                    'confidence': confidence,
                    'aliases': aliases
                })
                print(f"\n✓ Rule added successfully: {skill}")
                return True
            else:
                print(f"\n✗ Failed to add rule (may already exist)")
                return False
        else:
            print("Rule manager not initialized!")
            return False
    
    def add_rule_programmatic(
        self,
        skill: str,
        pattern: str,
        category: str,
        subcategory: str = 'General',
        confidence: float = 0.85,
        aliases: list = None
    ) -> bool:
        """Add a new rule programmatically."""
        if self.manager:
            success = self.manager.add_rule(
                skill=skill,
                pattern=pattern,
                category=category,
                subcategory=subcategory,
                confidence=confidence,
                aliases=aliases or []
            )
            if success:
                self.manager.save()
                self.extractor.add_rule({
                    'skill': skill,
                    'pattern': pattern,
                    'category': category,
                    'subcategory': subcategory,
                    'confidence': confidence,
                    'aliases': aliases or []
                })
            return success
        return False
    
    def remove_rule(self, skill: str) -> bool:
        """Remove a rule by skill name."""
        if self.manager:
            success = self.manager.remove_rule(skill)
            if success:
                self.manager.save()
                self.extractor.remove_rule(skill)
            return success
        return False
    
    def list_categories(self) -> list:
        """List all available categories."""
        categories = self.extractor.list_categories()
        print("\nAvailable Categories:")
        for cat in categories:
            rules = self.extractor.get_rules_by_category(cat)
            print(f"  {cat}: {len(rules)} rules")
        return categories
    
    def show_statistics(self) -> dict:
        """Show pipeline statistics."""
        stats = self.extractor.get_statistics()
        
        print("\n" + "=" * 70)
        print("PIPELINE STATISTICS")
        print("=" * 70)
        print(f"Total Rules Loaded: {stats['total_rules']}")
        print(f"Total Categories: {stats['total_categories']}")
        print(f"\nRules per Category:")
        for cat, count in sorted(stats['rules_per_category'].items()):
            print(f"  {cat}: {count}")
        print(f"\nConfidence Distribution:")
        for level, count in stats['confidence_distribution'].items():
            print(f"  {level}: {count}")
        
        return stats
    
    def search_rules(self, query: str) -> list:
        """Search for rules matching a query."""
        matches = self.extractor.search_rules(query)
        
        print(f"\nSearch results for '{query}':")
        for rule in matches:
            print(f"  - {rule['skill']} ({rule['category']}/{rule['subcategory']})")
        
        return matches


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Skill Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from Excel file
  python pipeline.py --input job_descriptions.xlsx --output results.json
  
  # Extract with confidence filter
  python pipeline.py --input data.csv --min-confidence 0.8
  
  # Extract specific categories only
  python pipeline.py --input data.csv --categories "Python" "Machine Learning"
  
  # Add new rule interactively
  python pipeline.py --add-rule
  
  # List all categories
  python pipeline.py --list-categories
  
  # Show statistics
  python pipeline.py --stats
        """
    )
    
    parser.add_argument('--input', '-i', help='Input file (CSV or Excel)')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--format', '-f', choices=['json', 'csv', 'detailed'], 
                       default='json', help='Output format')
    parser.add_argument('--text-column', default='cleaned_description',
                       help='Column containing text')
    parser.add_argument('--id-column', default='job_id',
                       help='Column containing document IDs')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                       help='Minimum confidence threshold (0.0-1.0)')
    parser.add_argument('--categories', nargs='+', help='Filter by categories')
    parser.add_argument('--rules', '-r', help='Path to rules file or directory')
    
    parser.add_argument('--add-rule', action='store_true', help='Add new rule interactively')
    parser.add_argument('--remove-rule', help='Remove rule by skill name')
    parser.add_argument('--search', '-s', help='Search for rules')
    parser.add_argument('--list-categories', action='store_true', help='List all categories')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SkillExtractionPipeline(rules_path=args.rules)
    
    # Handle commands
    if args.add_rule:
        pipeline.add_rule_interactive()
    
    elif args.remove_rule:
        pipeline.remove_rule(args.remove_rule)
    
    elif args.search:
        pipeline.search_rules(args.search)
    
    elif args.list_categories:
        pipeline.list_categories()
    
    elif args.stats:
        pipeline.show_statistics()
    
    elif args.input:
        pipeline.extract_from_file(
            input_file=args.input,
            text_column=args.text_column,
            id_column=args.id_column,
            output_file=args.output,
            output_format=args.format,
            min_confidence=args.min_confidence,
            categories=args.categories
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
