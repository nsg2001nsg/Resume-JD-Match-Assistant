import warnings
import pandas as pd
import joblib
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from data_prep import safe_extract_text, load_jd
from features import compute_features, FEATURE_COLUMNS
from fairness import apply_counterfactuals

# Load models once
print("Loading models...")
scaler = joblib.load('models/scaler.pkl')
lr_model = joblib.load('models/lr_model.pkl')

scenarios = [
    ("Suraj", "uploads/Suraj Gupta Resume (Yeh wala toh definitely).pdf", 
     "jds/python_backend.txt", "Should be STRONG ~90%+"),
    ("Suraj", "uploads/Suraj Gupta Resume (Yeh wala toh definitely).pdf", 
     "jds/data_scientist.txt", "Should be PARTIAL ~30-40%"),
    ("Nandini", "uploads/Update_Nandini_Resume.pdf",
     "jds/python_backend.txt", "Should be PARTIAL ~40-55%"),
    ("Nandini", "uploads/Update_Nandini_Resume.pdf",
     "jds/ACCOUNTANT/ACCOUNTANT_entry.txt", "Should be MISMATCH <25%"),
]

print("\nRunning final validation...\n")
for name, resume_path, jd_path, expectation in scenarios:
    resume_text = safe_extract_text(resume_path)[:12000]
    jd_text = load_jd(jd_path)
    
    # ORIGINAL
    features_dict = compute_features(resume_text, jd_text)
    features_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
    X_scaled = scaler.transform(features_df)
    lr_prob = float(lr_model.predict_proba(X_scaled)[0][1])
    
    # SWAPPED
    swapped_text = apply_counterfactuals(resume_text)
    swapped_features = compute_features(swapped_text, jd_text)
    swap_df = pd.DataFrame([swapped_features], columns=FEATURE_COLUMNS)
    swap_scaled = scaler.transform(swap_df)
    swap_prob = float(lr_model.predict_proba(swap_scaled)[0][1])
    
    variance = abs(lr_prob - swap_prob)
    
    print(f"--- {name} vs {jd_path.split('/')[-1]} ---")
    print(f"  Expected: {expectation}")
    print(f"  lr_prob: {lr_prob:.4f}")
    print(f"  swap_prob: {swap_prob:.4f}")
    print(f"  variance: {variance:.4f}")
    print("")
