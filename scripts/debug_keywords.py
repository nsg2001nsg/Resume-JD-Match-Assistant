import warnings
warnings.filterwarnings('ignore')

from data_prep import safe_extract_text, load_jd, extract_jd_keywords
from features import get_corpus_vectorizer

vec = get_corpus_vectorizer()
jd_text = load_jd("jds/python_backend.txt")
jd_kws = extract_jd_keywords(jd_text, vec, top_n=20)
print("JD Keywords:", jd_kws)

resume_text = safe_extract_text("uploads/Update_Nandini_Resume.pdf")
resume_lower = resume_text.lower()

# Old way
old_matches = [kw for kw in jd_kws if kw.lower() in resume_lower]
print("Old Matched:", old_matches)
print("Old Ratio:", len(old_matches)/len(jd_kws))

# New way
new_matches = []
for kw in jd_kws:
    words = kw.lower().split()
    if all(word in resume_lower for word in words):
        new_matches.append(kw)
print("New Matched:", new_matches)
print("New Ratio:", len(new_matches)/len(jd_kws))
