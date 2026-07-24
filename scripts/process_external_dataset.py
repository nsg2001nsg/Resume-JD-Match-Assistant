import pandas as pd

from features import FEATURE_COLUMNS, compute_features


LABEL_MAP = {
    "No Fit": 0,
    "Good Fit": 1,
}


def process_external():
    print("Loading labeled resume/JD pairs...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df["split"] = "train"
    test_df["split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True)

    print("Keeping binary labels only: No Fit and Good Fit.")
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df["binary_label"] = df["label"].map(LABEL_MAP)

    print(f"Rows after dropping Potential Fit: {len(df)}")
    print(df["label"].value_counts().to_string())

    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        resume_text = str(row["resume_text"])
        jd_text = str(row["job_description_text"])
        features = compute_features(resume_text, jd_text)
        if not features:
            continue

        features.pop("missing_keywords", None)
        rows.append(
            {
                "split": row["split"],
                "source_label": row["label"],
                "label": int(row["binary_label"]),
                **{feature: features[feature] for feature in FEATURE_COLUMNS},
            }
        )

        if i % 500 == 0:
            print(f"  Processed {i}/{total}", flush=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv("data/external_features.csv", index=False)
    print(f"Saved {len(out_df)} rows to external_features.csv")


if __name__ == "__main__":
    process_external()
