#!/usr/bin/env python3
"""
Rule-Based NER (Named Entity Recognition) for Skill Extraction
===============================================================
Extracts skills from job descriptions using rule-based NER approach.

NER Output includes:
    - Entity text (matched skill)
    - Entity label (SKILL category)
    - Start & End positions
    - Confidence score

Usage:
    python rule_based_ner_skill_extractor.py

Author: Data Engineering Pipeline
Version: 1.0.0
"""

import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Set, Tuple, Optional


# =============================================================================
# NER ENTITY DATA STRUCTURE
# =============================================================================

@dataclass
class NEREntity:
    """
    Named Entity representing a skill found in text.
    
    Attributes:
        text: The exact matched text from the document
        label: Entity label/category (e.g., PYTHON, CLOUD, ML)
        start: Start character position in text
        end: End character position in text
        skill_name: Canonical skill name
        subcategory: More specific classification
        confidence: Confidence score (0.0 - 1.0)
    """
    text: str
    label: str
    start: int
    end: int
    skill_name: str
    subcategory: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_tuple(self) -> Tuple[int, int, str]:
        """Return (start, end, label) format for spaCy compatibility."""
        return (self.start, self.end, self.label)
    
    def __repr__(self):
        return f"NEREntity('{self.text}', {self.label}, [{self.start}:{self.end}])"


@dataclass
class NERDocument:
    """
    Document with NER annotations.
    
    Attributes:
        doc_id: Document identifier
        text: Original text
        entities: List of extracted entities
    """
    doc_id: str
    text: str
    entities: List[NEREntity] = field(default_factory=list)
    
    @property
    def entity_count(self) -> int:
        return len(self.entities)
    
    @property
    def unique_skills(self) -> List[str]:
        return list(set(e.skill_name for e in self.entities))
    
    @property
    def labels_found(self) -> List[str]:
        return list(set(e.label for e in self.entities))
    
    def get_entities_by_label(self, label: str) -> List[NEREntity]:
        return [e for e in self.entities if e.label == label]
    
    def to_spacy_format(self) -> Dict:
        """Convert to spaCy training data format."""
        return {
            'text': self.text,
            'entities': [e.to_tuple() for e in self.entities]
        }
    
    def to_conll_format(self) -> str:
        """Convert to CoNLL format (token-based)."""
        # Simple word-based tokenization
        tokens = self.text.split()
        labels = ['O'] * len(tokens)
        
        # Mark entity tokens
        char_pos = 0
        for i, token in enumerate(tokens):
            token_start = self.text.find(token, char_pos)
            token_end = token_start + len(token)
            
            for entity in self.entities:
                if token_start >= entity.start and token_end <= entity.end:
                    if token_start == entity.start:
                        labels[i] = f'B-{entity.label}'
                    else:
                        labels[i] = f'I-{entity.label}'
                    break
            
            char_pos = token_end
        
        return '\n'.join(f'{token}\t{label}' for token, label in zip(tokens, labels))


# =============================================================================
# NER RULE DEFINITION
# =============================================================================

class NERRule:
    """
    A rule for Named Entity Recognition.
    
    Each rule defines:
        - Pattern: Regex pattern to match
        - Label: Entity type/category
        - Skill: Canonical skill name
        - Confidence: Match reliability
    """
    
    def __init__(
        self,
        skill: str,
        pattern: str,
        label: str,
        subcategory: str = 'General',
        confidence: float = 0.85,
        aliases: List[str] = None
    ):
        self.skill = skill
        self.pattern = pattern
        self.label = label
        self.subcategory = subcategory
        self.confidence = confidence
        self.aliases = aliases or []
        
        # Compile regex
        try:
            self.compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid pattern for '{skill}': {e}")
    
    def find_entities(self, text: str) -> List[NEREntity]:
        """Find all entity matches in text."""
        entities = []
        for match in self.compiled.finditer(text):
            entity = NEREntity(
                text=match.group(),
                label=self.label,
                start=match.start(),
                end=match.end(),
                skill_name=self.skill,
                subcategory=self.subcategory,
                confidence=self.confidence
            )
            entities.append(entity)
        return entities


# =============================================================================
# RULE-BASED NER ENGINE
# =============================================================================

class RuleBasedNER:
    """
    Rule-Based Named Entity Recognition Engine for Skill Extraction.
    
    This NER system uses predefined regex rules to identify skill entities
    in job descriptions. It provides:
    
    - Entity extraction with character positions
    - Multiple output formats (spaCy, CoNLL, JSON)
    - Overlap handling
    - Confidence scoring
    
    Entity Labels:
        PYTHON, JAVA, DATA_SCIENCE, MACHINE_LEARNING, CLOUD,
        DEVOPS, WEB_DEV, DATABASE, DATA_ENG, MOBILE, SECURITY,
        SOFT_SKILL, TESTING
    """
    
    # Category to NER Label mapping
    LABEL_MAP = {
        'Python': 'PYTHON',
        'Java': 'JAVA',
        'Data Science': 'DATA_SCIENCE',
        'Machine Learning': 'MACHINE_LEARNING',
        'Cloud Platforms': 'CLOUD',
        'DevOps': 'DEVOPS',
        'Web Development': 'WEB_DEV',
        'Database': 'DATABASE',
        'Data Engineering': 'DATA_ENG',
        'Mobile Development': 'MOBILE',
        'Cybersecurity': 'SECURITY',
        'Soft Skills': 'SOFT_SKILL',
        'Testing & QA': 'TESTING'
    }
    
    def __init__(self):
        self.rules: List[NERRule] = []
        self._rules_loaded = False
    
    def add_rule(
        self,
        skill: str,
        pattern: str,
        category: str,
        subcategory: str = 'General',
        confidence: float = 0.85,
        aliases: List[str] = None
    ) -> None:
        """Add a single NER rule."""
        label = self.LABEL_MAP.get(category, category.upper().replace(' ', '_'))
        rule = NERRule(
            skill=skill,
            pattern=pattern,
            label=label,
            subcategory=subcategory,
            confidence=confidence,
            aliases=aliases
        )
        self.rules.append(rule)
    
    def load_rules_from_dict(self, rules_list: List[Dict]) -> int:
        """Load rules from a list of dictionaries."""
        count = 0
        for rule_dict in rules_list:
            try:
                self.add_rule(
                    skill=rule_dict['skill'],
                    pattern=rule_dict['pattern'],
                    category=rule_dict.get('category', 'SKILL'),
                    subcategory=rule_dict.get('subcategory', 'General'),
                    confidence=rule_dict.get('confidence', 0.85),
                    aliases=rule_dict.get('aliases', [])
                )
                count += 1
            except (KeyError, ValueError):
                continue
        self._rules_loaded = True
        return count
    
    def extract_entities(
        self,
        text: str,
        doc_id: str = 'doc_0',
        remove_overlaps: bool = True,
        min_confidence: float = 0.0
    ) -> NERDocument:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to process
            doc_id: Document identifier
            remove_overlaps: Whether to remove overlapping entities
            min_confidence: Minimum confidence threshold
        
        Returns:
            NERDocument with extracted entities
        """
        if not text or not isinstance(text, str):
            return NERDocument(doc_id=doc_id, text='', entities=[])
        
        all_entities = []
        seen_spans = set()  # Track (start, end) to deduplicate
        
        for rule in self.rules:
            if rule.confidence < min_confidence:
                continue
            
            entities = rule.find_entities(text)
            for entity in entities:
                span = (entity.start, entity.end)
                
                # Skip duplicates at same position
                if span in seen_spans:
                    continue
                
                seen_spans.add(span)
                all_entities.append(entity)
        
        # Sort by position
        all_entities.sort(key=lambda e: (e.start, -e.end))
        
        # Remove overlapping entities (keep longer/higher confidence)
        if remove_overlaps:
            all_entities = self._remove_overlaps(all_entities)
        
        return NERDocument(doc_id=doc_id, text=text, entities=all_entities)
    
    def _remove_overlaps(self, entities: List[NEREntity]) -> List[NEREntity]:
        """Remove overlapping entities, keeping the best match."""
        if not entities:
            return []
        
        # Sort by start position, then by length (longer first)
        entities.sort(key=lambda e: (e.start, -(e.end - e.start), -e.confidence))
        
        result = []
        last_end = -1
        
        for entity in entities:
            if entity.start >= last_end:
                result.append(entity)
                last_end = entity.end
        
        return result
    
    def process_batch(
        self,
        documents: List[Tuple[str, str]],
        show_progress: bool = True,
        **kwargs
    ) -> List[NERDocument]:
        """
        Process multiple documents.
        
        Args:
            documents: List of (doc_id, text) tuples
            show_progress: Whether to show progress
            **kwargs: Additional arguments for extract_entities
        
        Returns:
            List of NERDocument objects
        """
        results = []
        total = len(documents)
        
        for i, (doc_id, text) in enumerate(documents):
            if show_progress and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{total} documents...")
            
            doc = self.extract_entities(text, doc_id=doc_id, **kwargs)
            results.append(doc)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get NER engine statistics."""
        label_counts = defaultdict(int)
        for rule in self.rules:
            label_counts[rule.label] += 1
        
        return {
            'total_rules': len(self.rules),
            'labels': list(set(r.label for r in self.rules)),
            'rules_per_label': dict(label_counts)
        }


# =============================================================================
# SKILL RULES DEFINITIONS
# =============================================================================

def get_skill_rules() -> List[Dict]:
    """
    Returns all skill extraction rules.
    
    Each rule contains:
        - skill: Canonical skill name
        - pattern: Regex pattern with word boundaries
        - category: Skill category (maps to NER label)
        - subcategory: More specific classification
        - confidence: Match confidence (0.0 - 1.0)
        - aliases: Alternative names
    """
    
    rules = []
    
    # =========================================================================
    # PYTHON SKILLS
    # =========================================================================
    python_skills = [
        {"skill": "Python", "pattern": r"\bpython\b", "subcategory": "Core Language", "confidence": 0.95, "aliases": ["Python 3"]},
        {"skill": "Django", "pattern": r"\bdjango\b", "subcategory": "Web Framework", "confidence": 0.95},
        {"skill": "Flask", "pattern": r"\bflask\b", "subcategory": "Web Framework", "confidence": 0.90},
        {"skill": "FastAPI", "pattern": r"\bfastapi\b", "subcategory": "Web Framework", "confidence": 0.95},
        {"skill": "Pandas", "pattern": r"\bpandas\b", "subcategory": "Data Analysis", "confidence": 0.95},
        {"skill": "NumPy", "pattern": r"\bnumpy\b", "subcategory": "Scientific Computing", "confidence": 0.95},
        {"skill": "SciPy", "pattern": r"\bscipy\b", "subcategory": "Scientific Computing", "confidence": 0.95},
        {"skill": "PyTorch", "pattern": r"\bpytorch\b", "subcategory": "Deep Learning", "confidence": 0.95},
        {"skill": "TensorFlow", "pattern": r"\btensorflow\b", "subcategory": "Deep Learning", "confidence": 0.95},
        {"skill": "Keras", "pattern": r"\bkeras\b", "subcategory": "Deep Learning", "confidence": 0.95},
        {"skill": "Scikit-learn", "pattern": r"\b(scikit-learn|sklearn)\b", "subcategory": "Machine Learning", "confidence": 0.95},
        {"skill": "Matplotlib", "pattern": r"\bmatplotlib\b", "subcategory": "Visualization", "confidence": 0.95},
        {"skill": "Seaborn", "pattern": r"\bseaborn\b", "subcategory": "Visualization", "confidence": 0.95},
        {"skill": "Plotly", "pattern": r"\bplotly\b", "subcategory": "Visualization", "confidence": 0.90},
        {"skill": "Jupyter", "pattern": r"\bjupyter\b", "subcategory": "Development", "confidence": 0.90},
        {"skill": "PySpark", "pattern": r"\bpyspark\b", "subcategory": "Big Data", "confidence": 0.95},
        {"skill": "Celery", "pattern": r"\bcelery\b", "subcategory": "Task Queue", "confidence": 0.85},
        {"skill": "SQLAlchemy", "pattern": r"\bsqlalchemy\b", "subcategory": "ORM", "confidence": 0.95},
        {"skill": "Pydantic", "pattern": r"\bpydantic\b", "subcategory": "Data Validation", "confidence": 0.95},
        {"skill": "Boto3", "pattern": r"\bboto3\b", "subcategory": "AWS SDK", "confidence": 0.95},
        {"skill": "Airflow", "pattern": r"\bairflow\b", "subcategory": "Orchestration", "confidence": 0.95},
        {"skill": "Streamlit", "pattern": r"\bstreamlit\b", "subcategory": "Dashboard", "confidence": 0.95},
        {"skill": "BeautifulSoup", "pattern": r"\b(beautifulsoup|bs4)\b", "subcategory": "Web Scraping", "confidence": 0.90},
        {"skill": "Scrapy", "pattern": r"\bscrapy\b", "subcategory": "Web Scraping", "confidence": 0.95},
        {"skill": "pytest", "pattern": r"\bpytest\b", "subcategory": "Testing", "confidence": 0.95},
        {"skill": "NLTK", "pattern": r"\bnltk\b", "subcategory": "NLP", "confidence": 0.95},
        {"skill": "spaCy", "pattern": r"\bspacy\b", "subcategory": "NLP", "confidence": 0.95},
        {"skill": "OpenCV", "pattern": r"\b(opencv|cv2)\b", "subcategory": "Computer Vision", "confidence": 0.95},
        {"skill": "Hugging Face", "pattern": r"\b(hugging\s*face|transformers)\b", "subcategory": "NLP/ML", "confidence": 0.90},
        {"skill": "LangChain", "pattern": r"\blangchain\b", "subcategory": "LLM Framework", "confidence": 0.95},
    ]
    for skill in python_skills:
        skill['category'] = 'Python'
        rules.append(skill)
    
    # =========================================================================
    # JAVA SKILLS
    # =========================================================================
    java_skills = [
        {"skill": "Java", "pattern": r"\bjava\b", "subcategory": "Core Language", "confidence": 0.95},
        {"skill": "Spring Framework", "pattern": r"\bspring\b", "subcategory": "Framework", "confidence": 0.90},
        {"skill": "Spring Boot", "pattern": r"\bspring\s*boot\b", "subcategory": "Framework", "confidence": 0.95},
        {"skill": "Spring MVC", "pattern": r"\bspring\s*mvc\b", "subcategory": "Framework", "confidence": 0.90},
        {"skill": "Hibernate", "pattern": r"\bhibernate\b", "subcategory": "ORM", "confidence": 0.95},
        {"skill": "JPA", "pattern": r"\bjpa\b", "subcategory": "ORM", "confidence": 0.90},
        {"skill": "Maven", "pattern": r"\bmaven\b", "subcategory": "Build Tool", "confidence": 0.95},
        {"skill": "Gradle", "pattern": r"\bgradle\b", "subcategory": "Build Tool", "confidence": 0.95},
        {"skill": "JUnit", "pattern": r"\bjunit\b", "subcategory": "Testing", "confidence": 0.95},
        {"skill": "Mockito", "pattern": r"\bmockito\b", "subcategory": "Testing", "confidence": 0.95},
        {"skill": "Tomcat", "pattern": r"\btomcat\b", "subcategory": "Server", "confidence": 0.90},
        {"skill": "Apache Kafka", "pattern": r"\bkafka\b", "subcategory": "Messaging", "confidence": 0.95},
        {"skill": "RabbitMQ", "pattern": r"\brabbitmq\b", "subcategory": "Messaging", "confidence": 0.95},
        {"skill": "Microservices", "pattern": r"\bmicroservices?\b", "subcategory": "Architecture", "confidence": 0.90},
        {"skill": "RESTful API", "pattern": r"\b(restful|rest\s*api)\b", "subcategory": "API", "confidence": 0.90},
        {"skill": "Java EE", "pattern": r"\b(java\s*ee|j2ee|jakarta)\b", "subcategory": "Enterprise", "confidence": 0.90},
    ]
    for skill in java_skills:
        skill['category'] = 'Java'
        rules.append(skill)
    
    # =========================================================================
    # DATA SCIENCE SKILLS
    # =========================================================================
    data_science_skills = [
        {"skill": "Data Science", "pattern": r"\bdata\s*science\b", "subcategory": "Core", "confidence": 0.95},
        {"skill": "Data Analysis", "pattern": r"\bdata\s*analysis\b", "subcategory": "Core", "confidence": 0.95},
        {"skill": "Statistics", "pattern": r"\b(statistics|statistical)\b", "subcategory": "Mathematics", "confidence": 0.90},
        {"skill": "Hypothesis Testing", "pattern": r"\bhypothesis\s*testing\b", "subcategory": "Statistics", "confidence": 0.90},
        {"skill": "Regression", "pattern": r"\bregression\b", "subcategory": "Statistics", "confidence": 0.85},
        {"skill": "Time Series", "pattern": r"\btime\s*series\b", "subcategory": "Statistics", "confidence": 0.90},
        {"skill": "A/B Testing", "pattern": r"\ba/?b\s*testing\b", "subcategory": "Experimentation", "confidence": 0.95},
        {"skill": "Predictive Modeling", "pattern": r"\bpredictive\s*model\b", "subcategory": "Modeling", "confidence": 0.90},
        {"skill": "Data Mining", "pattern": r"\bdata\s*mining\b", "subcategory": "Technique", "confidence": 0.90},
        {"skill": "Feature Engineering", "pattern": r"\bfeature\s*engineering\b", "subcategory": "Technique", "confidence": 0.95},
        {"skill": "ETL", "pattern": r"\betl\b", "subcategory": "Process", "confidence": 0.95},
        {"skill": "Data Visualization", "pattern": r"\bdata\s*visualization\b", "subcategory": "Visualization", "confidence": 0.95},
        {"skill": "Tableau", "pattern": r"\btableau\b", "subcategory": "BI Tool", "confidence": 0.95},
        {"skill": "Power BI", "pattern": r"\bpower\s*bi\b", "subcategory": "BI Tool", "confidence": 0.95},
        {"skill": "Looker", "pattern": r"\blooker\b", "subcategory": "BI Tool", "confidence": 0.90},
        {"skill": "R Programming", "pattern": r"\br\s*(programming|language)?\b", "subcategory": "Programming", "confidence": 0.75},
        {"skill": "SAS", "pattern": r"\bsas\b", "subcategory": "Statistical Software", "confidence": 0.90},
        {"skill": "SPSS", "pattern": r"\bspss\b", "subcategory": "Statistical Software", "confidence": 0.90},
        {"skill": "Excel", "pattern": r"\bexcel\b", "subcategory": "Spreadsheet", "confidence": 0.80},
    ]
    for skill in data_science_skills:
        skill['category'] = 'Data Science'
        rules.append(skill)
    
    # =========================================================================
    # MACHINE LEARNING SKILLS
    # =========================================================================
    ml_skills = [
        {"skill": "Machine Learning", "pattern": r"\bmachine\s*learning\b", "subcategory": "Core", "confidence": 0.95},
        {"skill": "Deep Learning", "pattern": r"\bdeep\s*learning\b", "subcategory": "Core", "confidence": 0.95},
        {"skill": "Artificial Intelligence", "pattern": r"\b(artificial\s*intelligence|\bai\b)\b", "subcategory": "Core", "confidence": 0.95},
        {"skill": "Neural Networks", "pattern": r"\bneural\s*network\b", "subcategory": "Architecture", "confidence": 0.95},
        {"skill": "CNN", "pattern": r"\b(cnn|convolutional\s*neural)\b", "subcategory": "Architecture", "confidence": 0.95},
        {"skill": "RNN", "pattern": r"\b(rnn|recurrent\s*neural)\b", "subcategory": "Architecture", "confidence": 0.95},
        {"skill": "LSTM", "pattern": r"\blstm\b", "subcategory": "Architecture", "confidence": 0.95},
        {"skill": "Transformer", "pattern": r"\btransformer\b", "subcategory": "Architecture", "confidence": 0.90},
        {"skill": "BERT", "pattern": r"\bbert\b", "subcategory": "NLP Model", "confidence": 0.95},
        {"skill": "GPT", "pattern": r"\bgpt\b", "subcategory": "NLP Model", "confidence": 0.95},
        {"skill": "LLM", "pattern": r"\b(llm|large\s*language\s*model)\b", "subcategory": "NLP Model", "confidence": 0.95},
        {"skill": "Generative AI", "pattern": r"\bgenerative\s*ai\b", "subcategory": "Generative", "confidence": 0.95},
        {"skill": "Computer Vision", "pattern": r"\bcomputer\s*vision\b", "subcategory": "Domain", "confidence": 0.95},
        {"skill": "NLP", "pattern": r"\b(nlp|natural\s*language\s*processing)\b", "subcategory": "Domain", "confidence": 0.95},
        {"skill": "Random Forest", "pattern": r"\brandom\s*forest\b", "subcategory": "Algorithm", "confidence": 0.95},
        {"skill": "XGBoost", "pattern": r"\bxgboost\b", "subcategory": "Algorithm", "confidence": 0.95},
        {"skill": "Gradient Boosting", "pattern": r"\bgradient\s*boosting\b", "subcategory": "Algorithm", "confidence": 0.90},
        {"skill": "SVM", "pattern": r"\b(svm|support\s*vector)\b", "subcategory": "Algorithm", "confidence": 0.90},
        {"skill": "Clustering", "pattern": r"\bclustering\b", "subcategory": "Task", "confidence": 0.85},
        {"skill": "Classification", "pattern": r"\bclassification\b", "subcategory": "Task", "confidence": 0.85},
        {"skill": "MLOps", "pattern": r"\bmlops\b", "subcategory": "MLOps", "confidence": 0.95},
        {"skill": "MLflow", "pattern": r"\bmlflow\b", "subcategory": "MLOps", "confidence": 0.95},
        {"skill": "Prompt Engineering", "pattern": r"\bprompt\s*engineering\b", "subcategory": "LLM", "confidence": 0.95},
        {"skill": "RAG", "pattern": r"\b(rag|retrieval\s*augmented)\b", "subcategory": "LLM", "confidence": 0.95},
    ]
    for skill in ml_skills:
        skill['category'] = 'Machine Learning'
        rules.append(skill)
    
    # =========================================================================
    # CLOUD PLATFORMS SKILLS
    # =========================================================================
    cloud_skills = [
        {"skill": "AWS", "pattern": r"\b(aws|amazon\s*web\s*services)\b", "subcategory": "Provider", "confidence": 0.95},
        {"skill": "Azure", "pattern": r"\b(azure|microsoft\s*azure)\b", "subcategory": "Provider", "confidence": 0.95},
        {"skill": "GCP", "pattern": r"\b(gcp|google\s*cloud)\b", "subcategory": "Provider", "confidence": 0.95},
        {"skill": "EC2", "pattern": r"\bec2\b", "subcategory": "AWS Compute", "confidence": 0.95},
        {"skill": "S3", "pattern": r"\bs3\b", "subcategory": "AWS Storage", "confidence": 0.95},
        {"skill": "Lambda", "pattern": r"\blambda\b", "subcategory": "AWS Serverless", "confidence": 0.85},
        {"skill": "ECS", "pattern": r"\becs\b", "subcategory": "AWS Container", "confidence": 0.90},
        {"skill": "EKS", "pattern": r"\beks\b", "subcategory": "AWS Container", "confidence": 0.90},
        {"skill": "RDS", "pattern": r"\brds\b", "subcategory": "AWS Database", "confidence": 0.90},
        {"skill": "DynamoDB", "pattern": r"\bdynamodb\b", "subcategory": "AWS Database", "confidence": 0.95},
        {"skill": "CloudFormation", "pattern": r"\bcloudformation\b", "subcategory": "AWS IaC", "confidence": 0.95},
        {"skill": "SageMaker", "pattern": r"\bsagemaker\b", "subcategory": "AWS ML", "confidence": 0.95},
        {"skill": "Redshift", "pattern": r"\bredshift\b", "subcategory": "AWS Analytics", "confidence": 0.95},
        {"skill": "EMR", "pattern": r"\bemr\b", "subcategory": "AWS Big Data", "confidence": 0.90},
        {"skill": "Kinesis", "pattern": r"\bkinesis\b", "subcategory": "AWS Streaming", "confidence": 0.95},
        {"skill": "BigQuery", "pattern": r"\bbigquery\b", "subcategory": "GCP Analytics", "confidence": 0.95},
        {"skill": "Serverless", "pattern": r"\bserverless\b", "subcategory": "Architecture", "confidence": 0.85},
    ]
    for skill in cloud_skills:
        skill['category'] = 'Cloud Platforms'
        rules.append(skill)
    
    # =========================================================================
    # DEVOPS SKILLS
    # =========================================================================
    devops_skills = [
        {"skill": "DevOps", "pattern": r"\bdevops\b", "subcategory": "Practice", "confidence": 0.95},
        {"skill": "CI/CD", "pattern": r"\b(ci/?cd|continuous\s*integration)\b", "subcategory": "Practice", "confidence": 0.95},
        {"skill": "Jenkins", "pattern": r"\bjenkins\b", "subcategory": "CI/CD Tool", "confidence": 0.95},
        {"skill": "GitLab CI", "pattern": r"\bgitlab\b", "subcategory": "CI/CD Tool", "confidence": 0.90},
        {"skill": "GitHub Actions", "pattern": r"\bgithub\s*actions\b", "subcategory": "CI/CD Tool", "confidence": 0.95},
        {"skill": "Docker", "pattern": r"\bdocker\b", "subcategory": "Containerization", "confidence": 0.95},
        {"skill": "Kubernetes", "pattern": r"\b(kubernetes|k8s)\b", "subcategory": "Orchestration", "confidence": 0.95},
        {"skill": "Helm", "pattern": r"\bhelm\b", "subcategory": "Kubernetes", "confidence": 0.85},
        {"skill": "Terraform", "pattern": r"\bterraform\b", "subcategory": "IaC", "confidence": 0.95},
        {"skill": "Ansible", "pattern": r"\bansible\b", "subcategory": "Configuration", "confidence": 0.95},
        {"skill": "Prometheus", "pattern": r"\bprometheus\b", "subcategory": "Monitoring", "confidence": 0.95},
        {"skill": "Grafana", "pattern": r"\bgrafana\b", "subcategory": "Monitoring", "confidence": 0.95},
        {"skill": "ELK Stack", "pattern": r"\belk\b", "subcategory": "Logging", "confidence": 0.90},
        {"skill": "Elasticsearch", "pattern": r"\belasticsearch\b", "subcategory": "Search", "confidence": 0.95},
        {"skill": "Linux", "pattern": r"\blinux\b", "subcategory": "OS", "confidence": 0.90},
        {"skill": "Bash", "pattern": r"\bbash\b", "subcategory": "Scripting", "confidence": 0.85},
        {"skill": "Git", "pattern": r"\bgit\b", "subcategory": "Version Control", "confidence": 0.90},
        {"skill": "GitHub", "pattern": r"\bgithub\b", "subcategory": "Version Control", "confidence": 0.90},
    ]
    for skill in devops_skills:
        skill['category'] = 'DevOps'
        rules.append(skill)
    
    # =========================================================================
    # WEB DEVELOPMENT SKILLS
    # =========================================================================
    web_skills = [
        {"skill": "JavaScript", "pattern": r"\bjavascript\b", "subcategory": "Language", "confidence": 0.95},
        {"skill": "TypeScript", "pattern": r"\btypescript\b", "subcategory": "Language", "confidence": 0.95},
        {"skill": "HTML", "pattern": r"\bhtml\b", "subcategory": "Markup", "confidence": 0.90},
        {"skill": "CSS", "pattern": r"\bcss\b", "subcategory": "Styling", "confidence": 0.90},
        {"skill": "React", "pattern": r"\breact(js)?\b", "subcategory": "Frontend", "confidence": 0.95},
        {"skill": "Angular", "pattern": r"\bangular\b", "subcategory": "Frontend", "confidence": 0.95},
        {"skill": "Vue.js", "pattern": r"\bvue(\.?js)?\b", "subcategory": "Frontend", "confidence": 0.95},
        {"skill": "Next.js", "pattern": r"\bnext(\.?js)?\b", "subcategory": "Framework", "confidence": 0.95},
        {"skill": "Node.js", "pattern": r"\bnode(\.?js)?\b", "subcategory": "Runtime", "confidence": 0.95},
        {"skill": "Express.js", "pattern": r"\bexpress(\.?js)?\b", "subcategory": "Backend", "confidence": 0.90},
        {"skill": "PHP", "pattern": r"\bphp\b", "subcategory": "Language", "confidence": 0.90},
        {"skill": "Laravel", "pattern": r"\blaravel\b", "subcategory": "PHP Framework", "confidence": 0.95},
        {"skill": "Ruby on Rails", "pattern": r"\b(rails|ruby\s*on\s*rails)\b", "subcategory": "Framework", "confidence": 0.95},
        {"skill": "Redux", "pattern": r"\bredux\b", "subcategory": "State Management", "confidence": 0.90},
        {"skill": "GraphQL", "pattern": r"\bgraphql\b", "subcategory": "API", "confidence": 0.95},
        {"skill": "REST API", "pattern": r"\brest\s*api\b", "subcategory": "API", "confidence": 0.90},
        {"skill": "Tailwind CSS", "pattern": r"\btailwind\b", "subcategory": "CSS Framework", "confidence": 0.95},
        {"skill": "Bootstrap", "pattern": r"\bbootstrap\b", "subcategory": "CSS Framework", "confidence": 0.90},
        {"skill": "Webpack", "pattern": r"\bwebpack\b", "subcategory": "Build Tool", "confidence": 0.90},
        {"skill": "Jest", "pattern": r"\bjest\b", "subcategory": "Testing", "confidence": 0.90},
    ]
    for skill in web_skills:
        skill['category'] = 'Web Development'
        rules.append(skill)
    
    # =========================================================================
    # DATABASE SKILLS
    # =========================================================================
    database_skills = [
        {"skill": "SQL", "pattern": r"\bsql\b", "subcategory": "Language", "confidence": 0.90},
        {"skill": "MySQL", "pattern": r"\bmysql\b", "subcategory": "Relational", "confidence": 0.95},
        {"skill": "PostgreSQL", "pattern": r"\b(postgresql|postgres)\b", "subcategory": "Relational", "confidence": 0.95},
        {"skill": "Oracle", "pattern": r"\boracle\b", "subcategory": "Relational", "confidence": 0.85},
        {"skill": "SQL Server", "pattern": r"\b(sql\s*server|mssql)\b", "subcategory": "Relational", "confidence": 0.90},
        {"skill": "MongoDB", "pattern": r"\bmongodb\b", "subcategory": "NoSQL", "confidence": 0.95},
        {"skill": "Cassandra", "pattern": r"\bcassandra\b", "subcategory": "NoSQL", "confidence": 0.95},
        {"skill": "Redis", "pattern": r"\bredis\b", "subcategory": "Cache", "confidence": 0.95},
        {"skill": "Elasticsearch", "pattern": r"\belasticsearch\b", "subcategory": "Search", "confidence": 0.95},
        {"skill": "Neo4j", "pattern": r"\bneo4j\b", "subcategory": "Graph", "confidence": 0.95},
        {"skill": "Snowflake", "pattern": r"\bsnowflake\b", "subcategory": "Data Warehouse", "confidence": 0.95},
        {"skill": "Redshift", "pattern": r"\bredshift\b", "subcategory": "Data Warehouse", "confidence": 0.95},
        {"skill": "BigQuery", "pattern": r"\bbigquery\b", "subcategory": "Data Warehouse", "confidence": 0.95},
        {"skill": "Pinecone", "pattern": r"\bpinecone\b", "subcategory": "Vector DB", "confidence": 0.95},
        {"skill": "NoSQL", "pattern": r"\bnosql\b", "subcategory": "Type", "confidence": 0.90},
    ]
    for skill in database_skills:
        skill['category'] = 'Database'
        rules.append(skill)
    
    # =========================================================================
    # DATA ENGINEERING SKILLS
    # =========================================================================
    data_eng_skills = [
        {"skill": "Data Engineering", "pattern": r"\bdata\s*engineering\b", "subcategory": "Core", "confidence": 0.95},
        {"skill": "Data Pipeline", "pattern": r"\bdata\s*pipeline\b", "subcategory": "Architecture", "confidence": 0.95},
        {"skill": "Apache Spark", "pattern": r"\bspark\b", "subcategory": "Processing", "confidence": 0.95},
        {"skill": "Apache Kafka", "pattern": r"\bkafka\b", "subcategory": "Streaming", "confidence": 0.95},
        {"skill": "Apache Flink", "pattern": r"\bflink\b", "subcategory": "Streaming", "confidence": 0.90},
        {"skill": "Apache Airflow", "pattern": r"\bairflow\b", "subcategory": "Orchestration", "confidence": 0.95},
        {"skill": "Dagster", "pattern": r"\bdagster\b", "subcategory": "Orchestration", "confidence": 0.90},
        {"skill": "Prefect", "pattern": r"\bprefect\b", "subcategory": "Orchestration", "confidence": 0.90},
        {"skill": "Databricks", "pattern": r"\bdatabricks\b", "subcategory": "Platform", "confidence": 0.95},
        {"skill": "Delta Lake", "pattern": r"\bdelta\s*lake\b", "subcategory": "Storage", "confidence": 0.95},
        {"skill": "Data Lake", "pattern": r"\bdata\s*lake\b", "subcategory": "Architecture", "confidence": 0.95},
        {"skill": "Data Warehouse", "pattern": r"\bdata\s*warehouse\b", "subcategory": "Architecture", "confidence": 0.95},
        {"skill": "dbt", "pattern": r"\bdbt\b", "subcategory": "Transformation", "confidence": 0.95},
        {"skill": "Hadoop", "pattern": r"\bhadoop\b", "subcategory": "Big Data", "confidence": 0.90},
        {"skill": "Data Quality", "pattern": r"\bdata\s*quality\b", "subcategory": "Governance", "confidence": 0.90},
    ]
    for skill in data_eng_skills:
        skill['category'] = 'Data Engineering'
        rules.append(skill)
    
    # =========================================================================
    # MOBILE DEVELOPMENT SKILLS
    # =========================================================================
    mobile_skills = [
        {"skill": "iOS", "pattern": r"\bios\b", "subcategory": "Platform", "confidence": 0.85},
        {"skill": "Android", "pattern": r"\bandroid\b", "subcategory": "Platform", "confidence": 0.90},
        {"skill": "Swift", "pattern": r"\bswift\b", "subcategory": "Language", "confidence": 0.90},
        {"skill": "Kotlin", "pattern": r"\bkotlin\b", "subcategory": "Language", "confidence": 0.95},
        {"skill": "Flutter", "pattern": r"\bflutter\b", "subcategory": "Cross-Platform", "confidence": 0.95},
        {"skill": "React Native", "pattern": r"\breact\s*native\b", "subcategory": "Cross-Platform", "confidence": 0.95},
        {"skill": "SwiftUI", "pattern": r"\bswiftui\b", "subcategory": "iOS Framework", "confidence": 0.95},
        {"skill": "Jetpack Compose", "pattern": r"\bcompose\b", "subcategory": "Android Framework", "confidence": 0.90},
    ]
    for skill in mobile_skills:
        skill['category'] = 'Mobile Development'
        rules.append(skill)
    
    # =========================================================================
    # CYBERSECURITY SKILLS
    # =========================================================================
    security_skills = [
        {"skill": "Cybersecurity", "pattern": r"\bcybersecurity\b", "subcategory": "Core", "confidence": 0.95},
        {"skill": "Information Security", "pattern": r"\b(information\s*security|infosec)\b", "subcategory": "Core", "confidence": 0.90},
        {"skill": "Encryption", "pattern": r"\bencryption\b", "subcategory": "Technique", "confidence": 0.90},
        {"skill": "Authentication", "pattern": r"\bauthentication\b", "subcategory": "Identity", "confidence": 0.85},
        {"skill": "OAuth", "pattern": r"\boauth\b", "subcategory": "Identity", "confidence": 0.90},
        {"skill": "Penetration Testing", "pattern": r"\bpenetration\s*testing\b", "subcategory": "Testing", "confidence": 0.95},
        {"skill": "OWASP", "pattern": r"\bowasp\b", "subcategory": "Framework", "confidence": 0.95},
        {"skill": "GDPR", "pattern": r"\bgdpr\b", "subcategory": "Compliance", "confidence": 0.95},
        {"skill": "HIPAA", "pattern": r"\bhipaa\b", "subcategory": "Compliance", "confidence": 0.95},
        {"skill": "SOC 2", "pattern": r"\bsoc\s*2\b", "subcategory": "Compliance", "confidence": 0.90},
    ]
    for skill in security_skills:
        skill['category'] = 'Cybersecurity'
        rules.append(skill)
    
    # =========================================================================
    # SOFT SKILLS
    # =========================================================================
    soft_skills = [
        {"skill": "Communication", "pattern": r"\bcommunication\b", "subcategory": "Interpersonal", "confidence": 0.85},
        {"skill": "Leadership", "pattern": r"\bleadership\b", "subcategory": "Management", "confidence": 0.85},
        {"skill": "Teamwork", "pattern": r"\bteamwork\b", "subcategory": "Collaboration", "confidence": 0.85},
        {"skill": "Problem Solving", "pattern": r"\bproblem[\s-]?solving\b", "subcategory": "Cognitive", "confidence": 0.90},
        {"skill": "Analytical Skills", "pattern": r"\banalytical\b", "subcategory": "Cognitive", "confidence": 0.85},
        {"skill": "Critical Thinking", "pattern": r"\bcritical\s*thinking\b", "subcategory": "Cognitive", "confidence": 0.90},
        {"skill": "Project Management", "pattern": r"\bproject\s*management\b", "subcategory": "Management", "confidence": 0.90},
        {"skill": "Agile", "pattern": r"\bagile\b", "subcategory": "Methodology", "confidence": 0.85},
        {"skill": "Scrum", "pattern": r"\bscrum\b", "subcategory": "Methodology", "confidence": 0.90},
        {"skill": "Mentoring", "pattern": r"\bmentor(ing)?\b", "subcategory": "Development", "confidence": 0.85},
        {"skill": "Collaboration", "pattern": r"\bcollaboration\b", "subcategory": "Collaboration", "confidence": 0.85},
        {"skill": "Cross-functional", "pattern": r"\bcross[\s-]?functional\b", "subcategory": "Collaboration", "confidence": 0.90},
        {"skill": "Innovation", "pattern": r"\binnovat(ion|ive)\b", "subcategory": "Personal", "confidence": 0.80},
        {"skill": "Requirements Gathering", "pattern": r"\brequirements\b", "subcategory": "Analysis", "confidence": 0.80},
    ]
    for skill in soft_skills:
        skill['category'] = 'Soft Skills'
        rules.append(skill)
    
    # =========================================================================
    # TESTING & QA SKILLS
    # =========================================================================
    testing_skills = [
        {"skill": "Software Testing", "pattern": r"\btesting\b", "subcategory": "Core", "confidence": 0.80},
        {"skill": "Quality Assurance", "pattern": r"\b(quality\s*assurance|qa)\b", "subcategory": "Core", "confidence": 0.90},
        {"skill": "Unit Testing", "pattern": r"\bunit\s*test\b", "subcategory": "Testing Type", "confidence": 0.90},
        {"skill": "Integration Testing", "pattern": r"\bintegration\s*test\b", "subcategory": "Testing Type", "confidence": 0.90},
        {"skill": "Test Automation", "pattern": r"\b(test\s*)?automation\b", "subcategory": "Automation", "confidence": 0.85},
        {"skill": "Selenium", "pattern": r"\bselenium\b", "subcategory": "Automation Tool", "confidence": 0.95},
        {"skill": "Cypress", "pattern": r"\bcypress\b", "subcategory": "Automation Tool", "confidence": 0.95},
        {"skill": "Postman", "pattern": r"\bpostman\b", "subcategory": "API Tool", "confidence": 0.90},
        {"skill": "JMeter", "pattern": r"\bjmeter\b", "subcategory": "Performance", "confidence": 0.90},
        {"skill": "TDD", "pattern": r"\b(tdd|test[\s-]?driven)\b", "subcategory": "Methodology", "confidence": 0.90},
        {"skill": "BDD", "pattern": r"\b(bdd|behavior[\s-]?driven)\b", "subcategory": "Methodology", "confidence": 0.90},
    ]
    for skill in testing_skills:
        skill['category'] = 'Testing & QA'
        rules.append(skill)
    
    return rules


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_skills_ner(
    input_file: str,
    output_file: str = 'ner_extracted_skills.csv',
    text_column: str = 'cleaned_description',
    id_column: str = 'job_id'
) -> Dict:
    """
    Extract skills from job descriptions using Rule-Based NER.
    
    Args:
        input_file: Path to input Excel/CSV file
        output_file: Path to output CSV file
        text_column: Column containing job descriptions
        id_column: Column containing document IDs
    
    Returns:
        Summary statistics dictionary
    """
    print("=" * 70)
    print("RULE-BASED NER SKILL EXTRACTION")
    print("=" * 70)
    
    # Initialize NER engine
    ner = RuleBasedNER()
    
    # Load rules
    rules = get_skill_rules()
    num_rules = ner.load_rules_from_dict(rules)
    
    stats = ner.get_statistics()
    print(f"\nNER Engine Initialized:")
    print(f"  Total Rules: {stats['total_rules']}")
    print(f"  Entity Labels: {len(stats['labels'])}")
    print(f"  Labels: {', '.join(sorted(stats['labels']))}")
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)
    print(f"Total documents: {len(df)}")
    
    # Find text column
    if text_column not in df.columns:
        for alt in ['jd', 'description', 'job_description', 'text', 'cleaned_description']:
            if alt in df.columns:
                text_column = alt
                break
    
    if id_column not in df.columns:
        df['_doc_id'] = range(len(df))
        id_column = '_doc_id'
    
    # Process documents
    print(f"\nExtracting entities...")
    
    results = []
    all_entities = []
    skill_counts = defaultdict(int)
    label_counts = defaultdict(int)
    
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")
        
        doc_id = str(row[id_column])
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        
        # Extract entities using NER
        doc = ner.extract_entities(text, doc_id=doc_id)
        
        # Collect results
        entity_records = []
        for entity in doc.entities:
            entity_records.append({
                'doc_id': doc_id,
                'entity_text': entity.text,
                'entity_label': entity.label,
                'skill_name': entity.skill_name,
                'subcategory': entity.subcategory,
                'start_pos': entity.start,
                'end_pos': entity.end,
                'confidence': entity.confidence
            })
            skill_counts[entity.skill_name] += 1
            label_counts[entity.label] += 1
        
        all_entities.extend(entity_records)
        
        # Summary per document
        results.append({
            'job_id': doc_id,
            'entity_count': doc.entity_count,
            'unique_skills': len(doc.unique_skills),
            'skills': ', '.join(sorted(doc.unique_skills)),
            'labels': ', '.join(sorted(doc.labels_found))
        })
    
    # Save document-level results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\nDocument results saved to: {output_file}")
    
    # Save entity-level results
    entities_file = output_file.replace('.csv', '_entities.csv')
    entities_df = pd.DataFrame(all_entities)
    entities_df.to_csv(entities_file, index=False)
    print(f"Entity details saved to: {entities_file}")
    
    # Generate summary
    summary = {
        'total_documents': len(df),
        'total_entities_extracted': len(all_entities),
        'unique_skills_found': len(skill_counts),
        'total_skill_mentions': sum(skill_counts.values()),
        'avg_entities_per_document': len(all_entities) / len(df),
        'top_20_skills': sorted(skill_counts.items(), key=lambda x: -x[1])[:20],
        'entities_by_label': dict(label_counts)
    }
    
    # Save summary JSON
    summary_file = output_file.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'extraction_date': datetime.now().isoformat(),
            'input_file': input_file,
            'ner_stats': stats,
            'summary': summary
        }, f, indent=2)
    print(f"Summary saved to: {summary_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("NER EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Documents Processed: {summary['total_documents']}")
    print(f"Total Entities Extracted: {summary['total_entities_extracted']}")
    print(f"Unique Skills Found: {summary['unique_skills_found']}")
    print(f"Avg Entities/Document: {summary['avg_entities_per_document']:.2f}")
    
    print(f"\nEntities by Label:")
    for label, count in sorted(summary['entities_by_label'].items()):
        print(f"  {label}: {count}")
    
    print(f"\nTop 20 Skills (by frequency):")
    for i, (skill, count) in enumerate(summary['top_20_skills'], 1):
        pct = count * 100 / len(df)
        print(f"  {i:2}. {skill}: {count} ({pct:.1f}%)")
    
    return summary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Rule-Based NER Skill Extraction from Job Descriptions'
    )
    parser.add_argument(
        '--input', '-i',
        default='cleaned_job_descriptions.xlsx',
        help='Input Excel/CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        default='ner_extracted_skills.csv',
        help='Output CSV file'
    )
    parser.add_argument(
        '--text-column',
        default='cleaned_description',
        help='Column containing job descriptions'
    )
    parser.add_argument(
        '--id-column',
        default='job_id',
        help='Column containing document IDs'
    )
    
    args = parser.parse_args()
    
    extract_skills_ner(
        input_file=args.input,
        output_file=args.output,
        text_column=args.text_column,
        id_column=args.id_column
    )
