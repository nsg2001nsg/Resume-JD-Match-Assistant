import pandas as pd

def analyze():
    print("Loading external dataset...")
    external_df = pd.read_csv('data/external_features.csv')
    external_df = external_df[external_df['label'].isin([0, 1])]
    
    # False Positives of pure similarity-based screening
    disagreements = external_df[
        (external_df['sbert_score'] > 0.5) & (external_df['label'] == 0)
    ]
    
    print(f"\nCases where semantic similarity misled (SBERT > 0.5 but Human = 0): {len(disagreements)}")
    
    features = ['tfidf_similarity', 'keyword_overlap_ratio', 'education_level_score', 'experience_gap', 'sbert_score']
    
    print("\nDescriptive statistics of the misleading cases:")
    print(disagreements[features].describe().round(4))
    
    # Let's also look at the False Negatives (SBERT < 0.35 but Human = 1)
    disagreements_fn = external_df[
        (external_df['sbert_score'] < 0.35) & (external_df['label'] == 1)
    ]
    print(f"\nCases where semantic similarity missed (SBERT < 0.35 but Human = 1): {len(disagreements_fn)}")
    if len(disagreements_fn) > 0:
        print("\nDescriptive statistics of the missed cases:")
        print(disagreements_fn[features].describe().round(4))

if __name__ == "__main__":
    analyze()
