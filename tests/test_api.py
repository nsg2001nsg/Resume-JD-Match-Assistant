import requests
import json
import os

def test():
    resume_dir = os.path.join("resumes", "data", "ACCOUNTANT")
    resumes = [f for f in os.listdir(resume_dir) if f.lower().endswith(".pdf")]
    
    jds_dir = os.path.join("jds", "ACCOUNTANT")
    jds = [f for f in os.listdir(jds_dir) if f.lower().endswith(".txt")]
    
    with open(os.path.join(jds_dir, jds[0]), "r", encoding="utf-8") as f:
        jd_text = f.read()
        
    url = "http://127.0.0.1:5000/api/score"
    
    with open(os.path.join(resume_dir, resumes[0]), "rb") as f:
        files = {'resume': f}
        data = {'jd_text': jd_text}
        
        print("Sending request...")
        response = requests.post(url, files=files, data=data)
        
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test()
