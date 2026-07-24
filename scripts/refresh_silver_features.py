import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_prep import load_jd, safe_extract_text
from features import FEATURE_COLUMNS, compute_features


def extract_resume_texts(paths, max_workers=8):
    cache = {}
    paths = list(paths)
    print(f"Extracting {len(paths)} unique resumes with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(safe_extract_text, path): path for path in paths}
        for i, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                cache[path] = future.result()
            except Exception:
                cache[path] = ""
            if i % 250 == 0:
                print(f"  Extracted {i}/{len(paths)} resumes", flush=True)
    return cache


def refresh_silver_features():
    print("Loading existing silver pair labels...")
    df = pd.read_csv("data/silver_dataset.csv")

    resume_cache = extract_resume_texts(df["resume_path"].dropna().unique())
    jd_cache = {path: load_jd(path) for path in df["jd_path"].dropna().unique()}

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        resume_path = row["resume_path"]
        jd_path = row["jd_path"]

        features = compute_features(resume_cache[resume_path], jd_cache[jd_path])
        if not features:
            continue

        features.pop("missing_keywords", None)
        updated = row.to_dict()
        for feature in FEATURE_COLUMNS:
            updated[feature] = features[feature]
        rows.append(updated)

        if (i + 1) % 500 == 0:
            print(f"  Refreshed {i + 1}/{total}", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv("data/silver_dataset.csv", index=False)
    print(f"Saved refreshed silver_dataset.csv with {len(out_df)} rows.")
    print("Label counts:")
    print(out_df["label"].value_counts().to_string())


if __name__ == "__main__":
    refresh_silver_features()
