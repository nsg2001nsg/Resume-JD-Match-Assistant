import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def finetune():
    print("Loading external dataset...")
    external_df = pd.read_csv('data/external_features.csv')
    external_df = external_df[external_df['label'].isin([0, 1])]
    
    features = ['tfidf_similarity', 'keyword_overlap_ratio', 'education_level_score', 'experience_gap']
    X_ext = external_df[features]
    y_ext = external_df['label']
    
    print("Splitting...")
    X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
        X_ext, y_ext, test_size=0.2, stratify=y_ext, random_state=42
    )
    
    print("Loading pre-trained model and scaler...")
    scaler = joblib.load('models/scaler.pkl')
    lr_model = joblib.load('models/lr_model.pkl')
    
    print("Scaling...")
    X_train_ext_scaled = scaler.transform(X_train_ext)
    X_test_ext_scaled = scaler.transform(X_test_ext)
    
    print("Fine-tuning model (Transfer Learning)...")
    # warm_start=True keeps the existing coefficients and updates them
    lr_model.set_params(warm_start=True, max_iter=500)
    lr_model.fit(X_train_ext_scaled, y_train_ext)
    
    print("Evaluating...")
    y_pred = lr_model.predict(X_test_ext_scaled)
    y_prob = lr_model.predict_proba(X_test_ext_scaled)[:, 1]
    
    print("\n--- FINE-TUNED MODEL VALIDATION ---")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_ext, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test_ext, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test_ext, y_prob):.4f}")
    
    joblib.dump(lr_model, 'models/finetuned_lr_model.pkl')
    print("\nSaved finetuned_lr_model.pkl")

if __name__ == "__main__":
    finetune()
