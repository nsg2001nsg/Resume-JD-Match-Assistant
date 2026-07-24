import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from features import FEATURE_COLUMNS

# Load BOTH models and compare on external test set
silver_model = joblib.load('models/lr_model.pkl')
finetuned_model = joblib.load('models/finetuned_lr_model.pkl')
scaler = joblib.load('models/scaler.pkl')

ext = pd.read_csv('data/external_features.csv')
ext = ext[ext['label'].isin([0, 1])]

X = scaler.transform(ext[FEATURE_COLUMNS])
y = ext['label']

for name, model in [('Silver', silver_model), ('Finetuned', finetuned_model)]:
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    print(f"\n--- {name} model ---")
    print(classification_report(y, preds))
    print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}")
    
    coef_dict = {col: round(val, 4) for col, val in zip(FEATURE_COLUMNS, model.coef_[0])}
    print(f"Coefficients: {coef_dict}")
