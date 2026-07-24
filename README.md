# Resume-JD Match Assistant (Backend Portfolio Project)

![Resume Scorer Demo](roc_curve.png) <!-- Update with actual app screenshot if available -->

## Overview
The **Resume-JD Match Assistant** is a web application designed to evaluate the alignment between a candidate's resume and a specific job description. 

**Project Scope & Authorship:** 
This project was developed primarily to demonstrate **backend engineering, API design, and deployment robustness**. While the underlying machine learning pipeline (Logistic Regression, TF-IDF, SHAP explainability) is fully functional, it was built by prompting AI. My core contribution and focus in this project is the backend architecture—specifically, how to safely, reliably, and efficiently serve an ML model in a production-ready environment without blocking operations or leaking memory.

## Backend Architecture & Technologies
- **Backend & API**: Python, Flask, Werkzeug, Gunicorn
- **Observability**: Python native `logging`
- **Data Processing**: Pandas, NumPy (Optimized for minimal footprint)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **AI-Assisted ML Pipeline**: Scikit-Learn (Logistic Regression), SHAP, TF-IDF Vectorization, PyPDF2, pdfplumber

## Backend Engineering Highlights
- **Zero-Overhead Memory Optimization**: Instead of running heavy Pandas data-wrangling on server boot, the SHAP background dataset is pre-computed offline and loaded as a lightweight NumPy array (`shap_background.npy`). This prevents severe memory spikes and ensures fast boot times, making the application stable for constrained PaaS free-tiers.
- **Hardened Error & File Handling**: Uploaded PDFs are strictly validated and parsed safely. The app gracefully catches file corruption (e.g., `PyPDF2.errors.PdfReadError`) and returns clean 400/500 JSON responses without hanging the main thread. All uploaded files are securely written to temporary directories and guaranteed to be wiped in `finally` blocks, preventing disk-space exhaustion.
- **Adversarial Input Detection (Input Quality)**: Actively scans for unnatural term frequencies to flag keyword stuffing and other attempts to manipulate the text extraction, surfacing these as Input Quality warnings to the API consumer.
- **Standardized Observability**: Utilizes Python's native `logging` module throughout the request lifecycle for proper production-grade tracing instead of standard prints.
- **PaaS Deployment Ready**: Shipped with a configured `Procfile` and Gunicorn WSGI server.

## Features (AI-Assisted ML Pipeline)
- **PDF Resume Parsing**: Handles standard text-based PDFs with heuristic fallback parsing.
- **Explainable ML Scoring**: Uses a Logistic Regression model trained via Weak Supervision on simulated silver data.
- **SHAP Explainability**: Visualizes exactly which features positively or negatively impacted the candidate's score.
- **Fairness Probes**: Implements counterfactual probes to ensure the model isn't overly biased by simple keyword matches.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ResumeAutomation.git
   cd ResumeAutomation
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application (Development)**
   ```bash
   python app.py
   ```

5. **Run the application (Production)**
   ```bash
   gunicorn app:app
   ```

## Usage
1. Open the application in your browser (`http://localhost:5000` or your deployed URL).
2. Upload a candidate's resume (PDF format).
3. Paste the target Job Description (JD) text into the provided text area.
4. Click **Analyze Resume**.
5. Review the resulting JD Fit Score, Input Quality warnings, extracted features, SHAP explainability insights, and triaged recruiter notes.

## Limitations
- **Synchronous ML Inference**: Currently, ML inference runs synchronously on the main thread. In a high-traffic production scenario, this should be offloaded to an asynchronous task queue (e.g., Celery + Redis).
- **Data Extraction**: Scanned image-based PDFs require OCR, which is currently outside the scope of this implementation.
- **AI-Generated ML Code**: The ML models and heuristics are not intended to represent state-of-the-art research; they exist to provide a realistic workload for the backend API.
