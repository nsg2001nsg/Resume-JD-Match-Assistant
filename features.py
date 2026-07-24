import os
import re
import pandas as pd
from data_prep import (
    tfidf_sim,
    keyword_overlap_ratio,
    education_level_score,
    extract_years,
    extract_jd_keywords,
    build_corpus_tfidf,
    load_jd,
    keyword_present,
    get_experience_confidence,
    get_education_confidence,
    extract_education_level,
    detect_experience_type,
)

FEATURE_COLUMNS = [
    'tfidf_similarity',
    'keyword_overlap_ratio', 
    'education_level_score',
    'experience_gap'
]

_corpus_vectorizer = None

NOISY_KEYWORDS = {
    "code", "educational", "party", "unit", "system", "process", "years", "experience",
    "development", "data", "software", "application", "project", "work", "role", "team",
    "rest", "api", "apis", "web", "cloud", "deployment", "databases", "database", "version",
    "control", "pipelines", "pipeline", "framework", "microframework", "programming", "services",
    "like", "tools", "tracking", "related", "field", "oriented", "using", "used", "working", 
    "strong", "good", "excellent", "skills", "skill", "technologies", "technology", "environment", 
    "environments", "various", "multiple", "including", "such", "best", "practices", "practice", 
    "solutions", "solution", "support", "technical", "business", "requirements", "design", "testing", 
    "test", "implementation", "implement", "create", "creating", "ensure", "ensuring", "provide", 
    "providing", "maintain", "maintaining", "quality", "performance", "issues", "issue", "resolve", 
    "resolving", "user", "users", "client", "clients", "customer", "customers", "product", "products", 
    "service", "services", "management", "manage", "managing", "lead", "leading", "team", "teams", 
    "collaborate", "collaborating", "cross", "functional", "agile", "scrum", "methodologies", 
    "methodology", "cycle", "lifecycle", "end", "high", "level", "complex", "simple", "new", 
    "existing", "current", "future", "key", "core", "basic", "advanced", "deep", "understanding", 
    "knowledge", "experience", "hands", "on", "years", "year", "degree", "bachelor", "master", 
    "phd", "science", "computer", "engineering", "related", "field", "preferred", "required", 
    "plus", "bonus", "ideal", "candidate", "role", "position", "job", "description", "responsibilities", 
    "duties", "qualifications", "requirements", "skills", "experience", "education", "background", 
    "company", "we", "our", "us", "you", "your", "they", "their", "them", "it", "its", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", 
    "shall", "should", "can", "could", "may", "might", "must", "and", "or", "but", "if", "then", 
    "else", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", 
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "based", "time", "part", "full", "looking", "join", "help", "make", "impact", "world",
    "fast", "paced", "dynamic", "growing", "innovative", "cutting", "edge", "state", "art",
    "passionate", "driven", "self", "starter", "motivated", "detail", "oriented", "analytical",
    "problem", "solving", "communication", "written", "verbal", "interpersonal", "ability", "able",
    "member", "members", "person", "people", "individual"
}

PHRASE_MAPPINGS = {
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "web api": "Web APIs",
    "web apis": "Web APIs",
    "aws": "Amazon Web Services (AWS)",
    "amazon web services": "Amazon Web Services (AWS)",
    "git": "Git Version Control",
    "github": "GitHub",
    "ci cd": "CI/CD Pipelines",
    "cicd": "CI/CD Pipelines",
    "nosql": "NoSQL Databases",
    "sql": "SQL Databases",
    "cloud": "Cloud Deployment",
    "docker": "Docker Containerization",
    "kubernetes": "Kubernetes Orchestration",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "react": "React Frontend Framework",
    "angular": "Angular Frontend Framework",
    "django": "Django Framework",
    "django framework": "Django Framework",
    "flask": "Flask Microframework",
    "flask microframework": "Flask Microframework",
    "fastapi": "FastAPI Framework",
    "fastapi framework": "FastAPI Framework",
    "python": "Python",
    "docker": "Docker Containerization",
    "docker containerization": "Docker Containerization"
}

GENERIC_ACTION_WORDS = {
    "build", "develop", "maintain", "optimize", "understanding", "responsibilities", 
    "responsibility", "title", "role", "candidate", "position", "ability", "knowledge", "familiarity",
    "write", "written", "writing", "read", "reading", "review", "reviewing", "analyze", "analyzing",
    "troubleshoot", "troubleshooting", "debug", "debugging", "fix", "fixing", "update", "updating",
    "upgrade", "upgrading", "migrate", "migrating", "deploy", "deploying", "release", "releasing",
    "monitor", "monitoring", "improve", "improving", "enhance", "enhancing", "scale", "scaling",
    "architect", "architecting", "design", "designing", "evaluate", "evaluating", "assess", "assessing"
}

EDUCATION_TERMS = {
    "computer science", "science", "related field", "science related", "bachelor", "master", "degree"
}

def clean_and_preserve_keywords(jd_text, raw_keywords):
    """
    Cleans up raw TF-IDF keywords:
    1. Preserves high-level phrases mapped from standard technical terms if they appear in the raw keywords or JD text.
    2. Filters out generic noisy keywords, responsibility terms, and education terms.
    3. Categorizes remaining technical requirements vs responsibility terms.
    """
    if not raw_keywords:
        return {
            "technical_requirements": [],
            "responsibility_terms": []
        }
        
    jd_lower = jd_text.lower()
    preserved_phrases = []
    matched_phrase_keys = set()
    
    # 1. Phrase-preservation check
    for key, mapped_val in PHRASE_MAPPINGS.items():
        pattern = rf'\b{re.escape(key)}\b'
        if re.search(pattern, jd_lower):
            preserved_phrases.append(mapped_val)
            for word in key.split():
                matched_phrase_keys.add(word)
                
    # 2. Token filtering and categorization
    cleaned_keywords = []
    responsibility_terms = []
    
    for kw in raw_keywords:
        kw_clean = kw.lower().strip()
        
        # Skip if generic noisy word
        if kw_clean in NOISY_KEYWORDS:
            continue
            
        # Exclude education terms completely
        if any(term in kw_clean for term in EDUCATION_TERMS):
            continue
            
        # Classify generic action / responsibility terms
        is_resp = any(word in kw_clean.split() or word == kw_clean for word in GENERIC_ACTION_WORDS)
        if is_resp:
            responsibility_terms.append(kw)
            continue
            
        if kw_clean in matched_phrase_keys:
            continue
        if len(kw_clean) <= 2:
            continue
            
        mapped = PHRASE_MAPPINGS.get(kw_clean, kw)
        cleaned_keywords.append(mapped)
        
    seen_tech = set()
    final_tech = []
    for kw in (preserved_phrases + cleaned_keywords):
        kw_lower = kw.lower().strip()
        if kw_lower not in seen_tech:
            seen_tech.add(kw_lower)
            final_tech.append(kw)
            
    seen_resp = set()
    final_resp = []
    for kw in responsibility_terms:
        kw_lower = kw.lower().strip()
        if kw_lower not in seen_resp:
            seen_resp.add(kw_lower)
            final_resp.append(kw)
            
    return {
        "technical_requirements": final_tech,
        "responsibility_terms": final_resp
    }

def get_corpus_vectorizer():
    global _corpus_vectorizer
    if _corpus_vectorizer is None:
        JD_DIR = "data/jds"
        all_jd_texts = []
        if os.path.exists(JD_DIR):
            for root, dirs, files in os.walk(JD_DIR):
                for file in files:
                    if file.endswith(".txt"):
                        t = load_jd(os.path.join(root, file))
                        if t: all_jd_texts.append(t)
        if all_jd_texts:
            _corpus_vectorizer = build_corpus_tfidf(all_jd_texts)
    return _corpus_vectorizer

def compute_features(resume_text, jd_text):
    """
    Computes the 4 core ML features in exact order required by the scaler/model:
    1. tfidf_similarity
    2. keyword_overlap_ratio
    3. education_level_score
    4. experience_gap
    """
    if not resume_text or not jd_text:
        return None
        
    vec = get_corpus_vectorizer()
    raw_jd_kws = extract_jd_keywords(jd_text, vec, top_n=20) if vec else []
    
    # Apply clean and preserve keywords to remove noise and keep technical phrases
    kws_categorized = clean_and_preserve_keywords(jd_text, raw_jd_kws)
    jd_kws = kws_categorized["technical_requirements"]
    
    t_sim = tfidf_sim(resume_text, jd_text)
    kw_ov = keyword_overlap_ratio(resume_text, jd_kws)
    edu_sc = education_level_score(resume_text, jd_text)
    
    res_yrs = extract_years(resume_text)
    jd_yrs = extract_years(jd_text)
    exp_gap = float(res_yrs - jd_yrs)
    
    missing_kws = [kw for kw in jd_kws if not keyword_present(resume_text, kw)]
    
    exp_conf = get_experience_confidence(resume_text, res_yrs)
    res_edu = extract_education_level(resume_text)
    edu_conf = get_education_confidence(resume_text, res_edu)
    
    experience_type = detect_experience_type(resume_text)
    
    jd_lower = jd_text.lower()
    jd_requires_experience = bool(re.search(r'\b(experience|years|fresher|experienced|minimum experience)\b', jd_lower))
    
    features_dict = {
        'tfidf_similarity': t_sim,
        'keyword_overlap_ratio': kw_ov,
        'education_level_score': edu_sc,
        'experience_gap': exp_gap,
        'missing_keywords': missing_kws[:5], # keep top 5 missing
        'experience_confidence': exp_conf,
        'education_confidence': edu_conf,
        'extracted_years': res_yrs,
        'extracted_education_level': res_edu,
        'experience_type': experience_type,
        'jd_requires_experience': jd_requires_experience
    }
    return features_dict

