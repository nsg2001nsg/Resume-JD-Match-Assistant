import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from features import FEATURE_COLUMNS


REPORT_DIR = Path("reports")


def evaluate():
    print("Loading external features...")
    df = pd.read_csv("data/external_features.csv")
    df = df.dropna(subset=["label"])

    X = df[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    print("Loading scaler and model...")
    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load("models/lr_model.pkl")

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    report = {
        "dataset": {
            "source": "train.csv + test.csv labeled resume/JD pairs",
            "rows": int(len(df)),
            "label_counts": {str(k): int(v) for k, v in y.value_counts().items()},
            "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().items()} if "split" in df.columns else None,
            "dropped_label": "Potential Fit",
        },
        "metrics": {
            "roc_auc": float(roc_auc_score(y, y_prob)),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
        },
        "limitations": [
            "This is external validation against labeled resume/JD pairs, not training data.",
            "Potential Fit rows are excluded for binary evaluation.",
            "Labels may reflect the external dataset's labeling policy, which may differ from the silver-label generator.",
        ],
    }

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "external_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# External Validation Report",
        "",
        "This report evaluates the trained silver-label model against the labeled resume/JD pair dataset.",
        "`Potential Fit` rows are excluded so the binary model is evaluated on `No Fit` vs `Good Fit`.",
        "",
        "## Dataset",
        "",
        f"- Rows: {report['dataset']['rows']}",
        f"- Label counts: {report['dataset']['label_counts']}",
        f"- Split counts: {report['dataset']['split_counts']}",
        "",
        "## Metrics",
        "",
        f"- ROC-AUC: {report['metrics']['roc_auc']:.4f}",
        f"- Confusion matrix: {report['metrics']['confusion_matrix']}",
        "",
        "## Limitations",
        "",
        "- External labels may not use the same definition of fit as the silver-label generator.",
        "- This should be read as transfer validation, not proof of hiring validity.",
        "",
    ]
    (REPORT_DIR / "external_validation.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n--- EXTERNAL VALIDATION RESULTS ---")
    print("\nConfusion Matrix:")
    print(report["metrics"]["confusion_matrix"])
    print(f"\nROC-AUC Score: {report['metrics']['roc_auc']:.4f}")
    print("\nSaved reports/external_validation.*")


if __name__ == "__main__":
    evaluate()
