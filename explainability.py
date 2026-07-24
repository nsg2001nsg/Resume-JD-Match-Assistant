import re
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from features import FEATURE_COLUMNS

# Centralized Constants for Syntactic Context Analysis
ACTION_VERBS = {
    "built", "implemented", "deployed", "created", "designed", "engineered",
    "programmed", "coded", "led", "developed", "architected", "optimized",
    "wrote", "authored", "scaled", "automated", "refactored", "delivered",
    "launched", "configured", "established", "headed"
}

DIRECT_SUBJECTS = {"i", "my", "we", "our"}

WORKFLOW_TERMS = {
    "using", "with", "via", "through", "alongside", "workflow", "workflows",
    "environment", "stack", "assisted", "helped", "supported", "collaborated",
    "worked with", "worked alongside", "team using"
}

DEPRIORITIZE_TERMS = {
    "familiarity", "fundamentals", "interested", "basic", "fundamental", "elementary",
    "concept", "concepts", "overview", "intro", "introduction", "beginner", "novice"
}

HIGH_VALUE_TECH = {
    "api", "apis", "microservices", "backend", "pipeline", "pipelines", "automation",
    "deployment", "scalability", "architecture", "integrations", "integration", "databases", "database"
}

# Soft, Confident, and Statistically Responsible Explainability Phrasing
FEATURE_EXPLANATIONS = {
    "tfidf_similarity": {
        "positive": "Vocabulary similarity shows strong alignment with core job description terms.",
        "negative": "Term choices show limited similarity to the phrasing used in the job description.",
        "neutral": "Vocabulary similarity shows standard baseline alignment."
    },
    "keyword_overlap_ratio": {
        "positive": "Strong textual evidence found for key domain and technical requirements.",
        "negative": "Domain keyword overlap shows limited coverage, suggesting key technical terms are absent.",
        "neutral": "Keyword coverage shows standard baseline overlap."
    },
    "education_level_score": {
        "positive": "Educational qualifications exceed the level specified in the job description.",
        "negative": "Extracted academic qualifications are below the requested level.",
        "neutral": "Academic qualifications meet the job description expectations at baseline."
    },
    "experience_gap": {
        "positive": "Extracted tenure meets or exceeds target requirements.",
        "negative": "Extracted professional tenure falls below target requirements.",
        "neutral": "Extracted tenure is aligned with the job description expectations."
    }
}

def get_soft_explainability_insights(shap_dict, education_status=None):
    """
    Translates SHAP log-odds contributions into statistically responsible, soft English summaries.
    """
    insights = {}
    for feature in FEATURE_COLUMNS:
        val = shap_dict.get(feature, 0.0)
        
        if feature == "education_level_score" and education_status is not None:
            text = education_status["summary"]
            impact = "positive" if education_status["tier_delta"] > 0 else ("neutral" if education_status["tier_delta"] == 0 else "negative")
        else:
            if val > 0.15:
                text = FEATURE_EXPLANATIONS[feature]["positive"]
            elif val < -0.15:
                text = FEATURE_EXPLANATIONS[feature]["negative"]
            else:
                text = FEATURE_EXPLANATIONS[feature]["neutral"]
            impact = "positive" if val > 0.15 else ("negative" if val < -0.15 else "neutral")
            
        insights[feature] = {
            "value": val,
            "impact": impact,
            "summary": text
        }
    return insights

def split_into_sentences(text):
    """
    Splits text into cleaned sentences while preserving readability.
    """
    if not text:
        return []
    raw_sentences = re.split(r'[.!?\n\r]+', text)
    cleaned = []
    for s in raw_sentences:
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if len(s_clean) > 10:  # Ignore trivial snippets
            cleaned.append(s_clean)
    return cleaned

def qualify_snippet_context(sentence, keyword):
    """
    Qualifies the evidence strength of a sentence for a keyword using deterministic rules:
    - Deprioritized terms (familiarity, basic, etc.) are forced to BRIEF_MENTION (priority 1).
    - STRONG_IMPLEMENTATION: requires BOTH an action verb AND direct subject ownership/context
      (or start-verb window, or high-value architectural nouns).
    - WORKFLOW_CONTEXT: workflow indicator terms are present.
    - BRIEF_MENTION: default context.
    """
    sentence_lower = sentence.lower()
    
    # 0. Deprioritization Check: Force BRIEF_MENTION if sentence contains any deprioritization term
    has_deprioritized = any(term in sentence_lower for term in DEPRIORITIZE_TERMS)
    if has_deprioritized:
        return {
            "code": "BRIEF_MENTION",
            "label": "Brief contextual mention",
            "priority": 1
        }
        
    words = re.findall(r'\b[a-z]+\b', sentence_lower)
    
    # 1. Action Verb Check
    has_action = any(v in words for v in ACTION_VERBS)
    
    # 2. Subject Ownership / Extended Context check
    has_subject = any(s in words for s in DIRECT_SUBJECTS)
    starts_with_action = False
    if words:
        # Resume bullets might start after brief modifiers, so we expand the window to 4 words
        starts_with_action = any(words[i] in ACTION_VERBS for i in range(min(4, len(words))))
        
    has_high_value_tech = any(t in words for t in HIGH_VALUE_TECH)
    
    is_strong = has_action and (has_subject or starts_with_action or has_high_value_tech)
    
    if is_strong:
        return {
            "code": "STRONG_IMPLEMENTATION",
            "label": "Strong implementation evidence",
            "priority": 3
        }
        
    # 3. Workflow Context Check
    has_workflow = any(w in sentence_lower for w in WORKFLOW_TERMS)
    if has_workflow:
        return {
            "code": "WORKFLOW_CONTEXT",
            "label": "Workflow context",
            "priority": 2
        }
        
    return {
        "code": "BRIEF_MENTION",
        "label": "Brief contextual mention",
        "priority": 1
    }

def extract_evidence_snippets(resume_text, jd_text, jd_keywords):
    """
    Scans the resume for JD keywords, extracts the sentence context, qualifies the strength,
    ranks them deterministically by priority, and slices to the Top 5 unique snippets.
    """
    if not resume_text or not jd_keywords:
        return {"matched": [], "missing": []}
        
    sentences = split_into_sentences(resume_text)
    matched_map = {}  # maps sentence -> {terms: [], priority: X, label: Y}
    matched_set = set()
    
    from data_prep import keyword_present
    
    for kw in jd_keywords:
        if not kw.strip():
            continue
        for sentence in sentences:
            if keyword_present(sentence, kw):
                matched_set.add(kw)
                
                # Determine quality/priority of this snippet for the keyword
                qualification = qualify_snippet_context(sentence, kw)
                
                if sentence not in matched_map:
                    matched_map[sentence] = {
                        "terms": [],
                        "priority": qualification["priority"],
                        "label": qualification["label"]
                    }
                matched_map[sentence]["terms"].append(kw)
                # Keep the highest priority if multiple keywords hit the same sentence
                matched_map[sentence]["priority"] = max(matched_map[sentence]["priority"], qualification["priority"])
                # Update label to reflect the maximum priority
                if matched_map[sentence]["priority"] == qualification["priority"]:
                    matched_map[sentence]["label"] = qualification["label"]
                break
                
    # Format and rank matched snippets deterministically
    matched_list = []
    for sentence, info in matched_map.items():
        matched_list.append({
            "snippet": sentence,
            "terms": sorted(list(set(info["terms"]))),
            "priority": info["priority"],
            "label": info["label"]
        })
        
    # Deterministic Sort: priority descending (3 -> 2 -> 1), then term count descending
    matched_list.sort(key=lambda x: (-x["priority"], -len(x["terms"])))
    
    # Slice to Top 5 evidence snippets to reduce UI clutter
    triaged_matched = matched_list[:5]
    
    # Clean priority key before sending to API to keep payload clean
    for item in triaged_matched:
        item.pop("priority", None)
        
    missing_keywords = [kw for kw in jd_keywords if kw not in matched_set]
    
    return {
        "matched": triaged_matched,
        "missing": missing_keywords
    }

def global_shap():
    print("Loading data and model...")
    df = pd.read_csv("data/silver_dataset.csv")
    df.fillna(0, inplace=True)
    
    X = df[FEATURE_COLUMNS]
    y = df['label']
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = joblib.load('models/lr_model.pkl')
    scaler = joblib.load("models/scaler.pkl")
    X_test_scaled = scaler.transform(X_test)
    
    print("Computing global SHAP values...")
    explainer = shap.LinearExplainer(model, X_test_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    
    print("Generating global summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLUMNS, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_global_importance.png")
    plt.close()
    print("Saved shap_global_importance.png")

if __name__ == "__main__":
    global_shap()
