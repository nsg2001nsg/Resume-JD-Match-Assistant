import re
import joblib
import numpy as np
import pandas as pd
from data_prep import load_jd, safe_extract_text
from features import FEATURE_COLUMNS, compute_features

# Multi-dimensional Swap Maps for Counterfactual Probing

GENDER_SWAPS = {
    # Female to Male names/pronouns
    "Priya": "Rahul", "Neha": "Amit", "Pooja": "Vikram",
    "Anjali": "Arjun", "Nandini": "Nikhil", "Shreya": "Shrey",
    "Divya": "Dev", "Riya": "Raj", "Ananya": "Ananth",
    "Sunita": "Sunil", "Kavita": "Karan", "Meena": "Mohan",
    "priya": "rahul", "neha": "amit", "pooja": "vikram",
    "anjali": "arjun", "nandini": "nikhil", "shreya": "shrey",
    "divya": "dev", "riya": "raj", "ananya": "ananth",
    "sunita": "sunil", "kavita": "karan", "meena": "mohan",
    "she": "he", "her": "his", "hers": "his",
    "She": "He", "Her": "His", "Hers": "His",
    "Ms.": "Mr.", "Mrs.": "Mr.", "Madam": "Sir",
    "ms.": "mr.", "mrs.": "mr.", "madam": "sir",
    "Miss": "Mr.", "miss": "mr.",
    # Male to Female names/pronouns (reverse probe)
    "Rahul": "Priya", "Amit": "Neha", "Vikram": "Pooja",
    "Arjun": "Anjali", "Nikhil": "Nandini", "Shrey": "Shreya",
    "Dev": "Divya", "Raj": "Riya", "Ananth": "Ananya",
    "Sunil": "Sunita", "Karan": "Kavita", "Mohan": "Meena",
    "rahul": "priya", "amit": "neha", "vikram": "pooja",
    "arjun": "anjali", "nikhil": "nandini", "shrey": "shreya",
    "dev": "divya", "raj": "riya", "ananth": "ananya",
    "sunil": "sunita", "karan": "kavita", "mohan": "meena",
    "he": "she", "his": "her", "him": "her",
    "He": "She", "His": "Her", "Him": "Her",
    "Mr.": "Ms.", "Sir": "Madam", "mr.": "ms.", "sir": "madam"
}

AGE_SWAPS = {
    # Young dynamic to experienced
    "recent graduate": "seasoned professional",
    "recent grad": "industry veteran",
    "entry-level grad": "seasoned specialist",
    "digital native": "industry veteran",
    "junior engineer": "senior engineer",
    "fresh graduate": "senior lead",
    "Recent Graduate": "Experienced Professional",
    "Recent Grad": "Seasoned Veteran",
    "Junior Engineer": "Senior Engineer",
    # Experienced to entry-level
    "seasoned professional": "recent graduate",
    "industry veteran": "recent grad",
    "senior director": "junior analyst",
    "decades of experience": "recent training",
    "retired senior": "recent grad",
    "Seasoned Professional": "Recent Graduate",
    "Industry Veteran": "Recent Grad",
    "Senior Director": "Junior Analyst"
}

REGION_SWAPS = {
    "Bangalore": "Mumbai",
    "Delhi": "Chennai",
    "Kolkata": "Pune",
    "Hyderabad": "Ahmedabad",
    "bangalore": "mumbai",
    "delhi": "chennai",
    "kolkata": "pune",
    "hyderabad": "ahmedabad",
    "Mumbai": "Bangalore",
    "Chennai": "Delhi",
    "Pune": "Kolkata",
    "Ahmedabad": "Hyderabad",
    "mumbai": "bangalore",
    "chennai": "delhi",
    "pune": "kolkata",
    "ahmedabad": "hyderabad"
}

PRESTIGE_SWAPS = {
    "IIT": "State University",
    "IIT Bombay": "Local Technical College",
    "IIT Madras": "Local Engineering College",
    "BITS Pilani": "State College",
    "Stanford University": "Community College",
    "Stanford": "State University",
    "MIT": "Technical College",
    "Harvard": "Generic College",
    "iit": "state university",
    "stanford": "state university",
    "mit": "technical college",
    "IIT Kharagpur": "State University",
    "IIT Delhi": "State University"
}

def swap_terms(text, swap_map):
    """
    Applies text substitutions based on a swap map with exact word boundaries.
    """
    if not text:
        return ""
    swapped = text
    for k, v in swap_map.items():
        swapped = re.sub(rf'\b{re.escape(k)}\b', v, swapped)
    return swapped

def apply_counterfactuals(text):
    """
    Legacy helper mapping only gender for backwards compatibility.
    """
    return swap_terms(text, GENDER_SWAPS)

def get_counterfactual_texts(text):
    """
    Generates perturbed text variants for each independent axis.
    """
    return {
        "gender": swap_terms(text, GENDER_SWAPS),
        "age": swap_terms(text, AGE_SWAPS),
        "region": swap_terms(text, REGION_SWAPS),
        "prestige": swap_terms(text, PRESTIGE_SWAPS)
    }

def run_counterfactual_probe(resume_text, jd_text, model, scaler):
    """
    Evaluates score sensitivity of a single resume/JD pair across all axes.
    """
    if not resume_text or not jd_text:
        return None
        
    features_orig = compute_features(resume_text, jd_text)
    if not features_orig:
        return None
        
    features_orig.pop("missing_keywords", None)
    df_orig = pd.DataFrame([features_orig], columns=FEATURE_COLUMNS)
    prob_orig = float(model.predict_proba(scaler.transform(df_orig))[0][1])
    
    perturbed = get_counterfactual_texts(resume_text)
    probe_results = {
        "original_score": round(prob_orig, 2),
        "variances": {}
    }
    
    max_var = 0.0
    
    for axis, swapped_text in perturbed.items():
        features_swap = compute_features(swapped_text, jd_text)
        if features_swap:
            features_swap.pop("missing_keywords", None)
            df_swap = pd.DataFrame([features_swap], columns=FEATURE_COLUMNS)
            prob_swap = float(model.predict_proba(scaler.transform(df_swap))[0][1])
            variance = abs(prob_orig - prob_swap)
        else:
            prob_swap = prob_orig
            variance = 0.0
            
        probe_results["variances"][axis] = {
            "score": round(prob_swap, 2),
            "variance": round(variance, 4)
        }
        max_var = max(max_var, variance)
        
    probe_results["max_variance"] = round(max_var, 4)
    # 0.02 is a strict threshold representing about 2% difference in JD match score
    probe_results["result"] = "LOW_SENSITIVITY" if max_var < 0.02 else "REVIEW"
    probe_results["note"] = "Probes measure score sensitivity to specific text substitutions (gender, age terms, region, prestige). High sensitivity requires human review."
    
    return probe_results

def run_aggregate_fairness():
    print("Loading silver dataset...")
    df = pd.read_csv('data/silver_dataset.csv')
    pairs = df.sample(min(50, len(df)), random_state=42)
    
    print("Loading model and scaler...")
    model = joblib.load('models/lr_model.pkl')
    scaler = joblib.load("models/scaler.pkl")
    
    print(f"Testing multidimensional fairness on {len(pairs)} pairs...")
    
    metrics = {
        "gender": [],
        "age": [],
        "region": [],
        "prestige": []
    }
    
    for i, (idx, row) in enumerate(pairs.iterrows(), start=1):
        resume_text = safe_extract_text(row['resume_path'])
        jd_text = load_jd(row['jd_path'])
        
        if not resume_text or not jd_text:
            continue
            
        probe = run_counterfactual_probe(resume_text, jd_text, model, scaler)
        if not probe:
            continue
            
        for axis in metrics.keys():
            metrics[axis].append(probe["variances"][axis]["variance"])
            
        if i % 10 == 0:
            print(f"  Processed {i}/{len(pairs)}")
            
    print("\n--- MULTIDIMENSIONAL AGGREGATE FAIRNESS TEST ---")
    for axis, vars_list in metrics.items():
        vars_arr = np.array(vars_list)
        mean_var = vars_arr.mean()
        max_var = vars_arr.max()
        pass_rate = (vars_arr < 0.02).mean() * 100
        print(f"\nAxis: {axis.upper()}")
        print(f"  Mean Variance: {mean_var:.6f}")
        print(f"  Max Variance:  {max_var:.6f}")
        print(f"  Pass Rate:     {pass_rate:.1f}% (<0.02 variance threshold)")

if __name__ == "__main__":
    run_aggregate_fairness()
