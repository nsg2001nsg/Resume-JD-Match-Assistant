"""
data_prep.py
Generates silver_dataset.csv from categorized resumes and JDs.
"""
import os
import re
import random
import csv
import pdfplumber
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date

# ── CONFIG ────────────────────────────────────────────────────────────────────
RESUME_DIR      = r"data/resumes/data"
JD_DIR          = r"data/jds"
OUTPUT_CSV      = r"data/silver_dataset.csv"
MAX_CHARS       = 12000
RANDOM_SEED     = 42

SIMILAR_CATEGORIES = {
    "ACCOUNTANT":           ["FINANCE", "BANKING", "CONSULTANT"],
    "FINANCE":              ["ACCOUNTANT", "BANKING", "BUSINESS-DEVELOPMENT"],
    "BANKING":              ["FINANCE", "ACCOUNTANT", "CONSULTANT"],
    "ENGINEERING":          ["AUTOMOBILE", "CONSTRUCTION", "AVIATION"],
    "INFORMATION-TECHNOLOGY": ["ENGINEERING", "DIGITAL-MEDIA", "CONSULTANT"],
    "DIGITAL-MEDIA":        ["PUBLIC-RELATIONS", "ARTS", "DESIGNER"],
    "DESIGNER":             ["ARTS", "DIGITAL-MEDIA", "APPAREL"],
    "ARTS":                 ["DESIGNER", "DIGITAL-MEDIA", "APPAREL"],
    "BUSINESS-DEVELOPMENT": ["SALES", "CONSULTANT", "FINANCE"],
    "SALES":                ["BUSINESS-DEVELOPMENT", "CONSULTANT"],
    "CONSULTANT":           ["BUSINESS-DEVELOPMENT", "FINANCE", "HR"],
    "HR":                   ["CONSULTANT", "BUSINESS-DEVELOPMENT"],
    "HEALTHCARE":           ["FITNESS"],
    "FITNESS":              ["HEALTHCARE"],
    "AUTOMOBILE":           ["ENGINEERING", "CONSTRUCTION"],
    "CONSTRUCTION":         ["ENGINEERING", "AUTOMOBILE"],
    "AVIATION":             ["ENGINEERING", "AUTOMOBILE"],
    "BPO":                  ["SALES", "HR"],
    "APPAREL":              ["DESIGNER", "ARTS"],
    "AGRICULTURE":          [],
    "ADVOCATE":             ["CONSULTANT"],
    "CHEF":                 [],
    "PUBLIC-RELATIONS":     ["DIGITAL-MEDIA", "SALES"],
    "TEACHER":              [],
}

# ── FEATURE HELPER FUNCTIONS ─────────────────────────────────────────────────

def safe_extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
            if sum(len(p) for p in text_parts) > MAX_CHARS:
                break
        raw = " ".join(text_parts)
        raw = re.sub(r'\s+', ' ', raw).strip()
        if raw:
            return raw[:MAX_CHARS]
    except Exception:
        pass

    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=3, y_tolerance=3)
                if t:
                    text_parts.append(t)
                if sum(len(p) for p in text_parts) > MAX_CHARS:
                    break
        raw = " ".join(text_parts)
        raw = re.sub(r'\s+', ' ', raw).strip()
        return raw[:MAX_CHARS]
    except Exception:
        return ""

def load_jd(jd_path):
    try:
        with open(jd_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return re.sub(r'\s+', ' ', text).strip()[:MAX_CHARS]
    except Exception:
        return ""

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

EMPLOYMENT_MARKERS = {
    "worked", "developer", "engineer", "intern", "experience", "employed", "company", "analyst",
    "manager", "lead", "specialist", "role", "position", "consultant", "programmer", "architect",
    "internship", "freelance", "trainee", "associate", "research assistant"
}

def detect_experience_type(text):
    """
    Uses a layered approach to detect what kind of experience the candidate has.
    Returns an object with boolean flags for different types.
    """
    if not text:
        return {"professional": False, "internship": False, "freelance": False, "project": False}
        
    text_lower = text.lower()
    lines = text_lower.split('\n')
    
    prof_headers = {"experience", "work experience", "employment", "professional experience", "work history", "career history"}
    intern_markers = {"intern", "internship"}
    freelance_markers = {"freelance", "freelancer", "contractor", "self-employed"}
    project_headers = {"projects", "academic projects", "personal projects", "portfolio"}
    
    types = {"professional": False, "internship": False, "freelance": False, "project": False}
    
    # 1. Structural heading detection
    for line in lines:
        line_clean = line.strip().rstrip(':')
        words = line_clean.split()
        if len(words) <= 4:
            if line_clean in prof_headers or any(h == line_clean for h in prof_headers):
                types["professional"] = True
            if line_clean in project_headers or any(h == line_clean for h in project_headers):
                types["project"] = True
                
    # 2. Textual marker fallback
    for marker in intern_markers:
        if re.search(rf'\b{re.escape(marker)}\b', text_lower):
            types["internship"] = True
            
    for marker in freelance_markers:
        if re.search(rf'\b{re.escape(marker)}\b', text_lower):
            types["freelance"] = True
            
    if re.search(r'\b(project|projects)\b', text_lower):
        types["project"] = True
        
    # If no explicit headings, look for implicit professional markers like full-time job indicators
    if not types["professional"] and re.search(r'\b(full time|full-time|software engineer|developer|manager|lead)\b', text_lower):
        # weak heuristic if not explicitly found
        pass
        
    return types

def segment_resume_sections(text):
    """
    Splits resume text into education, experience, and general sections to bypass academic years.
    """
    if not text:
        return {"education": "", "experience": "", "general": ""}
    
    lines = text.split('\n')
    sections = {"education": [], "experience": [], "general": []}
    current_sec = "general"
    
    edu_headers = {"education", "academic", "academics", "qualifications", "credentials", "studies", "schooling", "university", "college"}
    exp_headers = {"experience", "employment", "professional", "work history", "career history", "projects", "internships", "employment history", "roles"}
    
    for line in lines:
        line_clean = line.strip().lower().rstrip(':')
        words = line_clean.split()
        if len(words) <= 3 and any(w in edu_headers for w in words):
            current_sec = "education"
            continue
        elif len(words) <= 4 and any(w in exp_headers for w in words):
            current_sec = "experience"
            continue
            
        sections[current_sec].append(line)
        
    return {
        "education": "\n".join(sections["education"]),
        "experience": "\n".join(sections["experience"]),
        "general": "\n".join(sections["general"])
    }

def parse_year_value(value):
    value = value.lower().strip()
    if value.isdigit():
        return int(value)
    return NUMBER_WORDS.get(value)

def parse_date_marker(month_text, year_text):
    year = int(year_text)
    if year < 1970 or year > date.today().year:
        return None
    month = MONTHS.get((month_text or "").lower(), 1)
    return year + ((month - 1) / 12)

def extract_date_range_years(text):
    snippet = text[:3000]
    month_pattern = r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    range_pattern = re.compile(
        rf'(?:{month_pattern}\s+)?((?:19|20)\d{{2}})\s*(?:-|to|–|—)\s*(?:(?:{month_pattern}\s+)?((?:19|20)\d{{2}})|present|current|now)',
        re.IGNORECASE
    )

    intervals = []
    for match in range_pattern.finditer(snippet):
        # Surrounding context check: require job context markers to avoid education matches
        start_idx = max(0, match.start() - 100)
        end_idx = min(len(snippet), match.end() + 100)
        context = snippet[start_idx:end_idx].lower()
        
        has_context = any(marker in context for marker in EMPLOYMENT_MARKERS)
        if not has_context:
            continue
            
        start_month = match.group(1)
        start_year = match.group(2)
        end_month = match.group(3)
        end_year = match.group(4)

        start = parse_date_marker(start_month, start_year)
        if end_year:
            end = parse_date_marker(end_month, end_year)
        else:
            today = date.today()
            end = today.year + ((today.month - 1) / 12)

        if start is None or end is None:
            continue

        if 0 < (end - start) < 40:
            intervals.append((start, end))

    if not intervals:
        return 0

    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    total = sum(end - start for start, end in merged)
    return min(int(round(total)), 39)

def extract_years(text):
    # Segment resume into sections to bypass education dates completely
    sections = segment_resume_sections(text)
    
    # Strictly ignore sections['education']!
    target_text = sections["experience"]
    if not target_text.strip():
        target_text = sections["general"]
        
    snippet = target_text[:3000]
    number_token = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)'
    patterns = [
        rf'{number_token}\+?\s*years?\s*of\s*experience',
        rf'{number_token}\+?\s*years?\s*experience',
        rf'experience\s*(?:of\s*)?{number_token}\+?\s*years?',
        rf'{number_token}\+?\s*yrs?\s*(?:of\s*)?(?:exp|experience)',
    ]
    for pat in patterns:
        m = re.search(pat, snippet, re.IGNORECASE)
        if m:
            val = parse_year_value(m.group(1))
            if val and 0 < val < 40:
                return val
    return extract_date_range_years(snippet)

def extract_jd_required_years(jd_text):
    """
    Robust JD required experience parser handling ranges, minimums, and standard abbreviations.
    """
    if not jd_text:
        return 0
        
    # Case 1: "2-5 years", "3-5 years", "2 to 4 yrs"
    m_range = re.search(r'(\d+)\s*(?:-|to)\s*\d+\s*(?:years|yrs|year|yr)', jd_text, re.IGNORECASE)
    if m_range:
        return int(m_range.group(1))
        
    # Case 2: "2+ years", "3+ yrs", "5+ yrs"
    m_plus = re.search(r'(\d+)\+\s*(?:years|yrs|year|yr)', jd_text, re.IGNORECASE)
    if m_plus:
        return int(m_plus.group(1))
        
    # Case 3: "minimum 3 years", "min 2 years", "at least 3 years", "requires 2 years"
    m_min = re.search(r'(?:minimum|min|at least|require|requires|required|needs?|asking for)\s*(\d+)\s*(?:years|yrs|year|yr)', jd_text, re.IGNORECASE)
    if m_min:
        return int(m_min.group(1))
        
    # Case 4: "3 yrs of experience", "5 years experience"
    m_exp = re.search(r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience', jd_text, re.IGNORECASE)
    if m_exp:
        return int(m_exp.group(1))
        
    # Fallback to standard extract_years
    return extract_years(jd_text)


def tfidf_sim(text_a, text_b):
    try:
        vec = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf = vec.fit_transform([text_a, text_b])
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return round(float(score), 4)
    except Exception:
        return 0.0

def build_corpus_tfidf(all_jd_texts):
    vec = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.7,
        max_features=5000
    )
    vec.fit(all_jd_texts)
    return vec

def extract_jd_keywords(jd_text, fitted_vectorizer, top_n=20):
    try:
        tfidf_matrix = fitted_vectorizer.transform([jd_text])
        feature_names = fitted_vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        return keywords
    except Exception:
        return []

def normalize_keyword_text(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()

NORMALIZED_TECH_MAP = {
    "amazon web services (aws)": ["aws", "amazon web services", "s3", "ec2", "lambda"],
    "git version control": ["git", "github", "version control"],
    "github": ["github", "git"],
    "django framework": ["django"],
    "flask microframework": ["flask"],
    "sql databases": ["sql", "mysql", "postgresql", "postgres", "database", "databases"],
    "nosql databases": ["nosql", "mongodb", "database", "databases"],
    "rest apis": ["rest api", "rest apis", "rest", "api"],
    "web apis": ["web api", "web apis", "api"],
    "ci/cd pipelines": ["ci/cd", "cicd", "pipelines", "pipeline"],
    "cloud deployment": ["cloud", "deployment", "aws", "gcp", "azure"],
    "docker containerization": ["docker", "container", "containerization"],
    "kubernetes orchestration": ["kubernetes", "k8s", "orchestration"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "react frontend framework": ["react", "reactjs"],
    "angular frontend framework": ["angular"],
    "fastapi framework": ["fastapi"],
    "python programming": ["python"]
}

def keyword_present(text, keyword):
    normalized_text = f" {normalize_keyword_text(text)} "
    kw_lower = keyword.lower().strip()
    
    variations = [kw_lower]
    if kw_lower in NORMALIZED_TECH_MAP:
        variations.extend(NORMALIZED_TECH_MAP[kw_lower])
        
    for var in variations:
        normalized_var = normalize_keyword_text(var)
        if not normalized_var:
            continue
            
        tokens = normalized_var.split()
        if len(tokens) > 1:
            singular = normalized_var.rstrip('s')
            plural = normalized_var + 's'
            if (
                f" {normalized_var} " in normalized_text
                or f" {singular} " in normalized_text
                or f" {plural} " in normalized_text
            ):
                return True
        else:
            pattern = rf'\b{re.escape(normalized_var)}\b'
            if re.search(pattern, normalized_text):
                return True
                
    return False

def keyword_overlap_ratio(resume_text, jd_keywords):
    if not jd_keywords:
        return 0.0
    matches = sum(1 for kw in jd_keywords if keyword_present(resume_text, kw))
    return round(matches / len(jd_keywords), 4)

EDUCATION_TIERS = {
    1: ["10th", "ssc", "matric", "high school", "secondary school", "12th", "hsc", "higher secondary", "intermediate", "pu college"],
    2: ["diploma", "certificate course", "itc", "iti", "polytechnic", "associate degree"],
    3: ["b.sc", "b.com", "b.a", "b.e", "b.tech", "bca", "bba", "b.ed", "bachelor", "undergraduate", "ug", "b.arch", "b.des", "b.pharm", "bsc", "bcom", "btech", "llb"],
    4: ["m.sc", "m.com", "m.a", "m.e", "m.tech", "mca", "mba", "m.ed", "master", "postgraduate", "pg diploma", "pgdm", "m.arch", "m.des", "msc", "mcom", "mtech", "llm", "chartered accountant", "icwa", "cma", "company secretary", "mbbs"],
    5: ["ph.d", "phd", "doctorate", "d.sc", "d.litt"],
}

def extract_education_level(text):
    text_lower = text.lower()
    for tier in [5, 4, 3, 2, 1]:
        for keyword in EDUCATION_TIERS[tier]:
            pattern = r'\b' + re.escape(keyword.strip()) + r'\b'
            if re.search(pattern, text_lower):
                return tier
    return 3

def education_level_score(resume_text, jd_text):
    resume_edu = extract_education_level(resume_text)
    jd_edu     = extract_education_level(jd_text)
    return resume_edu - jd_edu

def get_education_status(resume_text, jd_text):
    """
    Returns a unified status object containing:
    {
        "tier_delta": delta,
        "label": "EXCEEDS" | "MEETS" | "BELOW",
        "summary": str
    }
    reused centrally across scoring, recruiter notes, explainability, and UI badges.
    """
    resume_edu = extract_education_level(resume_text)
    jd_edu = extract_education_level(jd_text)
    delta = resume_edu - jd_edu
    
    if delta > 0:
        label = "EXCEEDS"
        summary = "Educational qualifications exceed the level specified in the job description."
    elif delta == 0:
        label = "MEETS"
        summary = "Academic qualifications meet the job description expectations at baseline."
    else:
        label = "BELOW"
        summary = "Extracted academic qualifications are below the requested level."
        
    return {
        "tier_delta": int(delta),
        "label": label,
        "summary": summary
    }

def get_experience_confidence(text, extracted_years):
    """
    Determines extraction confidence for candidate tenure.
    """
    if not text:
        return "WEAK"
    snippet = text[:3000]
    number_token = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)'
    patterns = [
        rf'{number_token}\+?\s*years?\s*of\s*experience',
        rf'{number_token}\+?\s*years?\s*experience',
        rf'experience\s*(?:of\s*)?{number_token}\+?\s*years?',
    ]
    for pat in patterns:
        if re.search(pat, snippet, re.IGNORECASE):
            return "STRONG"
            
    if extracted_years > 0:
        return "STRONG" if extracted_years < 20 else "MODERATE"
        
    return "WEAK"

def get_education_confidence(text, extracted_tier):
    """
    Determines extraction confidence for candidate academic qualifications.
    """
    if not text:
        return "WEAK"
    text_lower = text.lower()
    specific_degrees = ["mca", "mba", "mtech", "m.tech", "btech", "b.tech", "phd", "ph.d", "bca", "bsc", "b.sc", "bcom", "b.com", "chartered accountant", "llb", "llm", "mbbs"]
    for degree in specific_degrees:
        if re.search(rf'\b{re.escape(degree)}\b', text_lower):
            return "STRONG"
            
    for tier in [5, 4, 3, 2, 1]:
        for keyword in EDUCATION_TIERS[tier]:
            if re.search(rf'\b{re.escape(keyword.strip())}\b', text_lower):
                return "MODERATE"
                
    return "WEAK"


def sentence_level_match(sent_embeddings, jd_embedding):
    if sent_embeddings is None:
        return 0.0
    sims = cosine_similarity(sent_embeddings, jd_embedding.reshape(1,-1))
    return float(sims.max())

# ── MAIN ──────────────────────────────────────────────────────────────────────

def build_dataset():
    from sentence_transformers import SentenceTransformer

    random.seed(RANDOM_SEED)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    all_categories = [
        d for d in os.listdir(RESUME_DIR)
        if os.path.isdir(os.path.join(RESUME_DIR, d))
    ]
    print(f"Found {len(all_categories)} categories: {all_categories}\n", flush=True)

    jd_pool = {}
    all_jd_texts = []
    
    for cat in all_categories:
        jd_folder = os.path.join(JD_DIR, cat)
        jd_files  = []
        if os.path.isdir(jd_folder):
            for fname in os.listdir(jd_folder):
                if fname.endswith('.txt'):
                    path = os.path.join(jd_folder, fname)
                    text = load_jd(path)
                    if text:
                        jd_files.append((path, text))
                        all_jd_texts.append(text)
        jd_pool[cat] = jd_files
        print(f"  {cat}: {len(jd_files)} JDs loaded", flush=True)

    # Fit corpus TF-IDF once on all 72 JDs
    print(f"\nFitting Corpus TF-IDF on {len(all_jd_texts)} documents...", flush=True)
    corpus_vectorizer = build_corpus_tfidf(all_jd_texts)
    
    # Pre-compute JD keywords cache
    print("Pre-computing JD keywords cache...", flush=True)
    jd_keywords_cache = {}
    jd_embeddings_cache = {}
    for cat in all_categories:
        for jd_path, jd_text in jd_pool[cat]:
            jd_keywords_cache[jd_path] = extract_jd_keywords(jd_text, corpus_vectorizer, top_n=20)
            jd_embeddings_cache[jd_path] = sbert.encode(jd_text, convert_to_numpy=True)

    rows = []
    skipped = 0
    total_processed = 0

    for cat in all_categories:
        resume_folder = os.path.join(RESUME_DIR, cat)
        if not os.path.isdir(resume_folder):
            continue

        pos_jds = jd_pool.get(cat, [])
        if not pos_jds:
            continue

        excluded = set(SIMILAR_CATEGORIES.get(cat, [])) | {cat}
        neg_candidates = [c for c in all_categories if c not in excluded]

        pdf_files = [f for f in os.listdir(resume_folder) if f.lower().endswith('.pdf')]
        print(f"\nProcessing {cat}: {len(pdf_files)} resumes", flush=True)

        for i, fname in enumerate(pdf_files):
            pdf_path = os.path.join(resume_folder, fname)

            if os.path.getsize(pdf_path) > 3 * 1024 * 1024:
                skipped += 1
                continue

            resume_text = safe_extract_text(pdf_path)
            if len(resume_text.strip()) < 100:
                skipped += 1
                continue

            resume_years = extract_years(resume_text)
            
            try:
                resume_emb = sbert.encode(resume_text, convert_to_numpy=True)
                
                # Pre-calculate sentence embeddings ONCE per resume
                resume_sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 20]
                resume_sentences = resume_sentences[3:33] if len(resume_sentences) > 3 else resume_sentences
                
                if resume_sentences:
                    sent_embeddings = sbert.encode(
                        resume_sentences, 
                        convert_to_numpy=True,
                        batch_size=32,
                        show_progress_bar=False
                    )
                else:
                    sent_embeddings = None
                    
            except Exception:
                skipped += 1
                continue

            # ── POSITIVE pairs ────────────────────────────
            for jd_path, jd_text in pos_jds:
                jd_emb = jd_embeddings_cache[jd_path]
                sbert_score = float(cosine_similarity(resume_emb.reshape(1, -1), jd_emb.reshape(1, -1))[0][0])

                jd_years = extract_years(jd_text)
                exp_gap  = resume_years - jd_years
                tfidf_s  = tfidf_sim(resume_text, jd_text)
                
                tfidf_normalized = min(tfidf_s / 0.5, 1.0)
                
                composite_score = (
                    0.5 * sbert_score +
                    0.4 * tfidf_normalized +
                    0.1 * (1 if exp_gap >= 0 else 0)
                )

                if composite_score > 0.55:
                    kw_overlap = keyword_overlap_ratio(resume_text, jd_keywords_cache[jd_path])
                    edu_score = education_level_score(resume_text, jd_text)
                    sent_match = sentence_level_match(sent_embeddings, jd_emb)
                    
                    rows.append({
                        "resume_path": pdf_path,
                        "jd_path": jd_path,
                        "category": cat,
                        "tfidf_similarity": tfidf_s,
                        "keyword_overlap_ratio": kw_overlap,
                        "education_level_score": edu_score,
                        "experience_gap": exp_gap,
                        "sentence_level_match": sent_match,
                        "sbert_score": round(sbert_score, 4),
                        "label": 1
                    })

            # ── NEGATIVE pairs ────────────────────────
            if not neg_candidates:
                continue

            neg_cat = random.choice(neg_candidates)
            neg_jds = jd_pool.get(neg_cat, [])
            if not neg_jds:
                continue

            for jd_path, jd_text in random.sample(neg_jds, min(3, len(neg_jds))):
                jd_emb = jd_embeddings_cache[jd_path]
                sbert_score = float(cosine_similarity(resume_emb.reshape(1, -1), jd_emb.reshape(1, -1))[0][0])

                jd_years = extract_years(jd_text)
                exp_gap  = resume_years - jd_years
                tfidf_s  = tfidf_sim(resume_text, jd_text)
                
                tfidf_normalized = min(tfidf_s / 0.5, 1.0)
                
                composite_score = (
                    0.5 * sbert_score +
                    0.4 * tfidf_normalized +
                    0.1 * (1 if exp_gap >= 0 else 0)
                )

                if composite_score < 0.35:
                    kw_overlap = keyword_overlap_ratio(resume_text, jd_keywords_cache[jd_path])
                    edu_score = education_level_score(resume_text, jd_text)
                    sent_match = sentence_level_match(sent_embeddings, jd_emb)
                    
                    rows.append({
                        "resume_path": pdf_path,
                        "jd_path": jd_path,
                        "category": cat,
                        "tfidf_similarity": tfidf_s,
                        "keyword_overlap_ratio": kw_overlap,
                        "education_level_score": edu_score,
                        "experience_gap": exp_gap,
                        "sentence_level_match": sent_match,
                        "sbert_score": round(sbert_score, 4),
                        "label": 0
                    })

            total_processed += 1
            if (i + 1) % 50 == 0:
                print(f"    [{cat}] {i+1}/{len(pdf_files)} done | rows so far: {len(rows)}", flush=True)

    if not rows:
        print("\nNo rows generated.", flush=True)
        return

    fieldnames = [
        "resume_path", "jd_path", "category", "tfidf_similarity", 
        "keyword_overlap_ratio", "education_level_score", 
        "experience_gap", "sentence_level_match", "sbert_score", "label"
    ]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*50}", flush=True)
    print(f"DONE. Total rows: {len(rows)}", flush=True)
    print(f"Skipped files:   {skipped}", flush=True)
    print(f"CSV saved to:    {OUTPUT_CSV}", flush=True)

if __name__ == "__main__":
    build_dataset()
