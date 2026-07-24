from flask import Flask, request, jsonify, render_template
import os
import joblib
import pandas as pd
import numpy as np
import traceback
import shap
import re
import uuid
import tempfile
import logging
from werkzeug.utils import secure_filename
from data_prep import safe_extract_text, extract_jd_keywords, extract_jd_required_years, get_education_status
from features import compute_features, FEATURE_COLUMNS, get_corpus_vectorizer, clean_and_preserve_keywords
from fairness import run_counterfactual_probe
from explainability import get_soft_explainability_insights, extract_evidence_snippets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', tempfile.gettempdir())
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.pdf'}

scaler = None
lr_model = None
explainer = None

try:
    lr_model = joblib.load('models/lr_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    logger.info("Successfully loaded scaler and lr_model.pkl")
    
    # Initialize SHAP explainer
    logger.info("Initializing SHAP explainer...")
    X_test_scaled = np.load("models/shap_background.npy")
    explainer = shap.LinearExplainer(lr_model, X_test_scaled)
    logger.info("SHAP explainer initialized.")
except Exception as e:
    logger.warning(f"Model/Scaler/Explainer not fully initialized yet. {e}")


def allowed_resume_file(filename):
    _, ext = os.path.splitext(filename or '')
    return ext.lower() in ALLOWED_EXTENSIONS

def build_upload_path(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        safe_name = "resume.pdf"
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    return os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/score', methods=['POST'])
def score():
    file_path = None
    try:
        if 'resume' not in request.files:
            return jsonify({"error": "No resume file provided."}), 400
            
        file = request.files['resume']
        jd_text = request.form.get('jd_text', '').strip()
        
        if file.filename == '':
            return jsonify({"error": "Empty filename."}), 400

        if not allowed_resume_file(file.filename):
            return jsonify({"error": "Only PDF resumes are supported."}), 400

        if not jd_text:
            return jsonify({"error": "Job description text is required."}), 400
            
        if scaler is None or lr_model is None or explainer is None:
            return jsonify({"error": "Machine learning model not loaded."}), 500
            
        file_path = build_upload_path(file.filename)
        file.save(file_path)
        
        resume_text = safe_extract_text(file_path)
        if len(resume_text.strip()) < 50:
            logger.warning("Failed to extract sufficient text from PDF.")
            return jsonify({
                "error": "Could not extract enough readable text from this PDF. Ensure it is not a scanned image, password-protected, or corrupted."
            }), 400
            
        # Input Quality & Reliability Checks
        input_quality = {
            "status": "OK",
            "warnings": []
        }
        
        word_counts = {}
        for word in resume_text.lower().split():
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
                
        total_words = len(resume_text.split())
        max_freq = max(word_counts.values()) if word_counts else 0
        
        if total_words > 0 and (max_freq / total_words) > 0.05 and max_freq > 20:
            input_quality["status"] = "WARNING"
            input_quality["warnings"].append("Unusually high repetition of specific words detected (potential keyword stuffing).")
            
        if len(resume_text) < 500:
            input_quality["status"] = "WARNING"
            input_quality["warnings"].append("Extracted text is very short. Ensure the PDF is fully readable and not an image.")
        
        # 1. Feature Extraction
        features_dict = compute_features(resume_text, jd_text)
        if not features_dict:
            return jsonify({"error": "Failed to extract text or features from the provided resume/JD."}), 400
            
        missing_keywords = features_dict.pop('missing_keywords', [])
        features_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
        
        # Canonical Education Status Unification
        education_status = get_education_status(resume_text, jd_text)
        
        # Scoring Calibration (Fresher Mismatch Penalty and Clamp)
        extracted_years = features_dict.get('extracted_years', 0)
        jd_required_years = extract_jd_required_years(jd_text)
        is_fresher_mismatch = (extracted_years == 0 and jd_required_years >= 2)
        
        # 2. Prediction
        X_scaled = scaler.transform(features_df)
        lr_prob = float(lr_model.predict_proba(X_scaled)[0][1])
        
        if is_fresher_mismatch:
            # Apply penalty multiplier (0.70) and clamp final score to continuous range [0.35, 0.58]
            lr_prob = lr_prob * 0.70
            lr_prob = max(0.35, min(lr_prob, 0.58))
        else:
            # Score Ceiling Calibration (Soft ceiling to prevent literal 100% match)
            if lr_prob > 0.80:
                lr_prob = 0.80 + (min(lr_prob, 1.0) - 0.80) * 0.60
            
        match_label = "Strong JD Fit" if lr_prob >= 0.5 else "Needs Review"
        
        # 3. SHAP Values
        shap_vals = explainer.shap_values(X_scaled)[0]
        base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) else explainer.expected_value
        final_log_odds = float(base_val + sum(shap_vals))
        
        shap_dict = {
            "base_value": round(float(base_val), 2),
            "final_log_odds": round(final_log_odds, 2)
        }
        for feature, val in zip(FEATURE_COLUMNS, shap_vals):
            shap_dict[feature] = round(float(val), 2)
            
        # Conversational plain-English insights (reusing unified education Fit status)
        shap_insights = get_soft_explainability_insights(shap_dict, education_status=education_status)

        # Extract Evidence Snippets & Missing Requirements (deduplicated and cleaned with phrase-preservation)
        vec = get_corpus_vectorizer()
        raw_jd_kws = extract_jd_keywords(jd_text, vec, top_n=20) if vec else []
        kws_categorized = clean_and_preserve_keywords(jd_text, raw_jd_kws)
        jd_kws = kws_categorized["technical_requirements"]
        evidence = extract_evidence_snippets(resume_text, jd_text, jd_kws)
            
        # 4. Fairness (Multidimensional Counterfactual Sensitivity Probe)
        fairness_dict = run_counterfactual_probe(resume_text, jd_text, lr_model, scaler)
        if fairness_dict:
            # Inject flat legacy key 'variance' for backwards compatibility in UI
            fairness_dict["variance"] = fairness_dict["max_variance"]
        else:
            fairness_dict = {
                "original_score": round(lr_prob, 2),
                "variance": 0.0,
                "max_variance": 0.0,
                "result": "ERROR",
                "probe": "multidimensional_counterfactual_probe",
                "note": "Failed to run fairness probe."
            }
        
        # 5. Divergence Warning (Using user's heuristic threshold)
        divergence_warning = None
        if features_dict['tfidf_similarity'] > 0.4 and features_dict['experience_gap'] < 0:
            divergence_warning = "High semantic similarity but negative experience gap detected. Model alignment may overestimate fit."
            
        # 6. Structured Recommendation Engine (Severity Tiers, Recruiter vs Candidate split, deduplication, and ethical guardrails)
        recruiter_notes = []
        candidate_advice = []
        
        # Recruiter Risk Notes (Severity Tiers: CRITICAL, WARNING, INFO)
        if is_fresher_mismatch:
            recruiter_notes.append({
                "tier": "CRITICAL",
                "message": f"Experience requirement mismatch: Candidate is a fresher (0 years extracted experience) applying for an experienced role requiring at least {jd_required_years} years."
            })
            
        if features_dict['experience_gap'] < -3:
            recruiter_notes.append({
                "tier": "CRITICAL",
                "message": f"Significant experience gap detected: Extracted candidate tenure is {abs(features_dict['experience_gap']):.1f} years below the target requirements."
            })
        elif features_dict['experience_gap'] < 0:
            # Only add general gap warning if we haven't already added a fresher mismatch warning
            if not is_fresher_mismatch:
                recruiter_notes.append({
                    "tier": "WARNING",
                    "message": f"Experience gap: Extracted candidate tenure is {abs(features_dict['experience_gap']):.1f} years below requirements."
                })
            
        if features_dict['keyword_overlap_ratio'] < 0.20:
            recruiter_notes.append({
                "tier": "CRITICAL",
                "message": f"Critical core-skill deficit: Vocabulary coverage is extremely sparse ({features_dict['keyword_overlap_ratio'] * 100:.0f}% overlap)."
            })
        elif features_dict['keyword_overlap_ratio'] < 0.35:
            recruiter_notes.append({
                "tier": "WARNING",
                "message": f"Moderate core-skill deficit: Domain vocabulary coverage is relatively low ({features_dict['keyword_overlap_ratio'] * 100:.0f}% overlap)."
            })
            
        # Check for missing cloud/deployment keywords
        cloud_deployment_terms = {
            "aws", "amazon web services", "cloud", "deployment", "docker", "kubernetes", 
            "cicd", "ci/cd", "ci cd", "pipelines", "pipeline", "gcp", "azure",
            "amazon web services (aws)", "cloud deployment", "docker containerization",
            "kubernetes orchestration", "ci/cd pipelines"
        }
        missing_cloud_dev = [kw for kw in evidence.get("missing", []) if kw.lower() in cloud_deployment_terms]
        if missing_cloud_dev:
            recruiter_notes.append({
                "tier": "WARNING",
                "message": f"Missing cloud/deployment requirement: Candidate has no documented experience with {', '.join(missing_cloud_dev[:2])}."
            })
            
        if education_status['label'] == "BELOW":
            recruiter_notes.append({
                "tier": "INFO",
                "message": f"Education tier variance: Candidate degree is below the tier requested in the JD (Variance of {abs(education_status['tier_delta'])} tier(s))."
            })
            
        # Candidate Actionable Advice (Strict Ethical Guidelines, Soft Advice, Deduplicated)
        missing_terms = evidence.get("missing", [])
        
        if len(missing_terms) >= 2:
            limit_missing = missing_terms[:4]
            candidate_advice.append(
                f"To better demonstrate match visibility, consider elaborating on your actual, honest experience with: {', '.join(limit_missing)} if applicable."
            )
        else:
            candidate_advice.append(
                "No significant missing technical skills detected."
            )
            
        if features_dict['experience_gap'] < 0:
            candidate_advice.append(
                "Ensure that all internships, prior employment tenures, or overlapping projects are explicitly detailed with dates to accurately represent your total experience."
            )
            
        if education_status['label'] == "BELOW":
            candidate_advice.append(
                "If you hold equivalent professional certifications, intensive bootcamps, or relevant coursework matching the requested academic tier, consider listing them prominently in your education section."
            )
            
        # Fallbacks:
        if not recruiter_notes:
            recruiter_notes.append({
                "tier": "INFO",
                "message": "Candidate features show solid general alignment across all extracted metrics."
            })
            
        if not candidate_advice:
            candidate_advice.append(
                "Your resume demonstrates strong vocabulary alignment. Keep prior role responsibilities explicitly detailed with metrics where possible."
            )
            
        # Deterministic Prioritization and Triage slicing (Top 3 for signal strength)
        PRIORITY_ORDER = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
        
        def get_note_priority(note):
            msg = note["message"].lower()
            if "experience requirement mismatch" in msg:
                return 4.5
            if "experience gap" in msg:
                return 4.0
            
            base_priority = PRIORITY_ORDER.get(note["tier"], 1)
            # Cloud/deployment missing has secondary warning priority within its tier
            if "cloud" in msg or "deployment" in msg:
                return base_priority + 0.5
            return base_priority
            
        recruiter_notes.sort(key=lambda x: -get_note_priority(x))
        triaged_recruiter_notes = recruiter_notes[:3]
        
        triaged_candidate_advice = candidate_advice[:3]
        
        response = {
            "match_score": round(lr_prob, 2),
            "match_probability": round(lr_prob, 2),
            "match_label": match_label,
            "extracted_features": {
                k: (round(v, 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                for k, v in features_dict.items()
            },
            "education_status": education_status,
            "shap_values": shap_dict,
            "shap_insights": shap_insights,
            "fairness": fairness_dict,
            "evidence": evidence,
            "input_quality": input_quality,
            "recruiter_notes": triaged_recruiter_notes,
            "candidate_advice": triaged_candidate_advice,
            "recommendation_list": triaged_candidate_advice  # Backwards compatibility
        }
        
        if divergence_warning:
            response["divergence_warning"] = divergence_warning
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing resume: {traceback.format_exc()}")
        return jsonify({
            "error": "An unexpected error occurred while processing the resume.",
            "details": str(e)
        }), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                logger.error(f"Failed to clean up file: {file_path}")

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug_mode, port=5000)
