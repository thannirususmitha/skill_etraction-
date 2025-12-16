# Skill Extraction Pipeline

A comprehensive rule-based NER (Named Entity Recognition) system for extracting skills from job descriptions.

## 📁 Project Structure

```
skill_extraction_pipeline/
├── pipeline.py              # Main CLI entry point
├── src/
│   ├── __init__.py
│   ├── skill_extractor.py   # Core extraction module
│   └── rule_manager.py      # Dynamic rule management
├── rules/
│   ├── initial_rules.json   # Consolidated rules file (441 rules)
│   ├── python_skills.jsonl
│   ├── java_skills.jsonl
│   ├── data_science_skills.jsonl
│   ├── machine_learning_skills.jsonl
│   ├── cloud_platforms_skills.jsonl
│   ├── devops_skills.jsonl
│   ├── web_development_skills.jsonl
│   ├── database_skills.jsonl
│   ├── data_engineering_skills.jsonl
│   ├── mobile_development_skills.jsonl
│   ├── cybersecurity_skills.jsonl
│   ├── soft_skills.jsonl
│   └── testing_qa_skills.jsonl
└── output/
    ├── extraction_results.json
    └── extraction_results.csv
```

## 🚀 Quick Start

### 1. Extract Skills from a File

```bash
# From CSV file
python pipeline.py --input job_descriptions.csv --output results.json

# From Excel file
python pipeline.py --input job_descriptions.xlsx --output results.json

# With specific text column
python pipeline.py --input data.xlsx --text-column "description" --output results.json
```

### 2. Python API Usage

```python
from src.skill_extractor import SkillExtractor

# Initialize and load rules
extractor = SkillExtractor('rules/')

# Extract skills from text
text = """
Looking for a Senior Data Engineer with Python, Apache Spark, 
and AWS experience. Must have strong SQL skills and knowledge 
of Airflow for ETL pipelines.
"""

result = extractor.extract_skills(text)

print(f"Skills found: {result.unique_skills}")
print(f"By category: {result.skills_by_category}")
```

### 3. Add New Rules

```bash
# Interactive mode
python pipeline.py --add-rule

# Or programmatically
```

```python
from src.rule_manager import RuleManager

manager = RuleManager('rules/initial_rules.json')

# Add a new rule
manager.add_rule(
    skill="FastAPI",
    pattern=r"\bfastapi\b",
    category="Python",
    subcategory="Web Framework",
    confidence=0.95,
    aliases=["Fast API"]
)

manager.save()
```

## 📊 Categories (13 Total, 441 Rules)

| Category | Rules | Description |
|----------|-------|-------------|
| Python | 41 | Python language, frameworks, libraries |
| Java | 30 | Java ecosystem, Spring, enterprise |
| Data Science | 34 | Statistics, analytics, BI tools |
| Machine Learning | 40 | ML/DL, algorithms, frameworks, LLMs |
| Cloud Platforms | 36 | AWS, Azure, GCP services |
| DevOps | 36 | CI/CD, containers, monitoring |
| Web Development | 39 | Frontend, backend, full-stack |
| Database | 34 | SQL, NoSQL, data warehouses |
| Data Engineering | 38 | Spark, Kafka, pipelines |
| Mobile Development | 24 | iOS, Android, cross-platform |
| Cybersecurity | 30 | Security, compliance, identity |
| Soft Skills | 28 | Communication, leadership |
| Testing & QA | 31 | Testing types, automation tools |

## 📝 JSONL Rule Format

Each rule is a JSON object with the following structure:

```json
{
  "skill": "Python",
  "pattern": "\\bpython\\b",
  "category": "Python",
  "subcategory": "Core Language",
  "confidence": 0.95,
  "aliases": ["Python 3", "Python3"]
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `skill` | string | ✓ | Canonical skill name |
| `pattern` | string | ✓ | Regex pattern with word boundaries |
| `category` | string | ✓ | Main category |
| `subcategory` | string | | More specific classification |
| `confidence` | float | | Score 0.0-1.0 (default: 0.85) |
| `aliases` | array | | Alternative names |

## 🔧 CLI Commands

```bash
# Extract from file
python pipeline.py --input data.xlsx --output results.json

# With confidence filter
python pipeline.py --input data.xlsx --min-confidence 0.8

# Filter by specific categories
python pipeline.py --input data.xlsx --categories "Python" "Machine Learning"

# Different output formats
python pipeline.py --input data.xlsx --format csv
python pipeline.py --input data.xlsx --format detailed

# Add new rule interactively
python pipeline.py --add-rule

# Remove a rule
python pipeline.py --remove-rule "SkillName"

# Search rules
python pipeline.py --search "python"

# List all categories
python pipeline.py --list-categories

# Show statistics
python pipeline.py --stats
```

## 💡 Adding New Rules

### Method 1: Interactive CLI

```bash
python pipeline.py --add-rule
```

Follow the prompts to enter:
- Skill name
- Regex pattern
- Category
- Subcategory
- Confidence score
- Aliases

### Method 2: Edit JSONL Files Directly

Add a new line to the appropriate category file:

```bash
echo '{"skill": "NewSkill", "pattern": "\\\\bnewskill\\\\b", "category": "Python", "subcategory": "Library", "confidence": 0.90, "aliases": []}' >> rules/python_skills.jsonl
```

### Method 3: Python API

```python
from src.rule_manager import RuleManager

manager = RuleManager('rules/initial_rules.json')

# Add single rule
manager.add_rule(
    skill="GraphQL",
    pattern=r"\bgraphql\b",
    category="Web Development",
    subcategory="API",
    confidence=0.95
)

# Add multiple rules
new_rules = [
    {"skill": "Rust", "pattern": r"\brust\b", "category": "Other", "subcategory": "Language", "confidence": 0.90},
    {"skill": "Go", "pattern": r"\bgo(lang)?\b", "category": "Other", "subcategory": "Language", "confidence": 0.85},
]
manager.add_rules_batch(new_rules)

manager.save()
```

### Method 4: Import from External JSONL

```python
manager.import_from_jsonl('path/to/new_rules.jsonl')
manager.save()
```

## 📈 Extraction Results

The extraction results include:

```python
{
    'document_id': 'job_123',
    'skill_count': 15,
    'unique_skills': ['Python', 'AWS', 'SQL', ...],
    'skills_by_category': {
        'Python': ['Python', 'Django', 'FastAPI'],
        'Cloud Platforms': ['AWS', 'S3', 'Lambda'],
        'Database': ['SQL', 'PostgreSQL']
    }
}
```

## 🎯 Extraction Statistics (from 7,670 JDs)

**Top 10 Skills Found:**
1. Python: 44.6%
2. SQL: 38.6%
3. Communication: 34.4%
4. AI/ML: 28.8%
5. Java: 26.5%
6. AWS: 26.0%
7. Machine Learning: 21.0%
8. CI/CD: 20.8%
9. Data Science: 19.7%
10. Agile: 18.4%

## 🔄 Updating Rules

### Update Existing Rule

```python
manager.update_rule("Python", {
    "confidence": 0.98,
    "aliases": ["Python 3", "Python3", "Py"]
})
manager.save()
```

### Remove Rule

```python
manager.remove_rule("DeprecatedSkill")
manager.save()
```

## 📦 Dependencies

- Python 3.8+
- pandas
- openpyxl (for Excel files)

```bash
pip install pandas openpyxl
```

## 🧪 Testing the Pipeline

```python
from src.skill_extractor import SkillExtractor

# Load rules
extractor = SkillExtractor('rules/')

# Get statistics
stats = extractor.get_statistics()
print(f"Total rules: {stats['total_rules']}")
print(f"Categories: {stats['categories']}")

# Test extraction
test_text = "Python developer with AWS and Docker experience"
result = extractor.extract_skills(test_text)
print(f"Found: {result.unique_skills}")
```

## 📋 Best Practices

1. **Use word boundaries** in patterns: `\\bskill\\b`
2. **Set appropriate confidence** levels (0.75-0.95)
3. **Include common aliases** for better matching
4. **Test patterns** before adding to production
5. **Backup rules** before major changes: `manager.backup()`

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

**Generated:** December 2024  
**Total Rules:** 441  
**Categories:** 13
