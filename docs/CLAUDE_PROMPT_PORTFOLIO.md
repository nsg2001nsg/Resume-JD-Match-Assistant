# Technical Portfolio & Prompt for Academic Research Paper Drafting

Copy and paste this document directly into Claude (or another LLM) to generate a high-quality, publication-grade academic research paper. This document contains the full context of the project, including technical architectures, mathematical formulations, code logic, and empirical evaluation metrics.

---

## SECTION 1: SYSTEM CONTEXT & MOTIVATION

### 1.1 Baseline System Overview
* **Baseline Reference**: The system builds upon the multi-agent recruitment framework proposed by **Gan et al. (2024)** (*"Application of LLM Agents in Recruitment: A Novel Framework for Resume Screening"*). 
* **Baseline Architecture**: The upstream baseline uses LLM-agents to ingest unstructured resumes, classify sentences, summarize profiles, and output candidate suitability scores.
* **The "Black-Box" Problem**: The baseline LLM scoring layer operates as a black box. It lacks transparency, does not provide feature-level mathematical justifications, and introduces potential demographic/cognitive biases based on sensitive identifiers (e.g., gendered names, regional city proxies, age-coded terms, and educational prestige).

### 1.2 Our Architectural Extension
We have designed and fully implemented a **Fairness-Aware and Explainable LLM-Agent Extension** inside a Python/Flask dashboard framework. Instead of relying on an LLM for direct scoring, we feed candidate features into an interpretable **Logistic Regression Classifier**. Transparency is established via **SHAP (Shapley Additive exPlanations)**, while bias is programmatically audited through a **Multi-Dimensional Counterfactual Fairness Probe**.

---

## SECTION 2: SYSTEM ARCHITECTURE & FEATURE ENGINEERING

The system extracts and normalizes a hybrid set of features. Each candidate-job description (JD) pair is converted into a feature vector $\mathbf{x} = [x_1, x_2, x_3, x_4]^T$:

1. **Semantic Similarity ($x_1 = \text{tfidf\_similarity}$)**:
   * Computed via TF-IDF cosine similarity between the TF-IDF representation of the resume and the job description.
   * Captured using a trained vectorizer (`tfidf_vectorizer.pkl`) representing the background vocabulary.
2. **Technical Keyword Overlap Ratio ($x_2 = \text{keyword\_overlap\_ratio}$)**:
   * Ratio of JD technical keywords present in the candidate resume:
     $$x_2 = \frac{|K_{\text{Resume}} \cap K_{\text{JD}}|}{|K_{\text{JD}}|}$$
   * Supported by an alias-aware matching pipeline mapping standard variants (e.g., `"AWS"`, `"Amazon Web Services"`, `"S3"`, `"EC2"` map to the canonical `"Amazon Web Services (AWS)"`).
3. **Education Level Score ($x_3 = \text{education\_level\_score}$)**:
   * Calculated by mapping degrees to standardized academic tiers:
     * **Tier 4 (Postgraduate/Doctorate)**: PhD, Master of Computer Applications (MCA), Master of Technology (M.Tech), MS, MBA.
     * **Tier 3 (Professional Bachelor's)**: B.E., B.Tech, B.Sc. in Computer Science.
     * **Tier 2 (General Bachelor's/Diploma)**: BCA, General B.Sc., Polytechnic Diploma.
     * **Tier 1 (Secondary/Foundational)**: High School, HSC, SSC.
   * Features represent the delta between candidate tier and JD required tier:
     $$x_3 = \text{Tier}_{\text{Candidate}} - \text{Tier}_{\text{JD}}$$
4. **Experience Gap ($x_4 = \text{experience\_gap}$)**:
   * Professional work tenure in years minus the JD requested experience:
     $$x_4 = Y_{\text{Candidate}} - Y_{\text{JD}}$$
   * Calculated using section-aware parsing and date regex matching.

---

## SECTION 3: THE SCORING ENGINE & MATH FORMULATION

### 3.1 Logistic Regression Classifier
The scoring engine computes the probability of candidate match $P(y = 1 | \mathbf{x})$ using the logistic sigmoid function:
$$P(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Where:
* $\mathbf{w}$ is the weight vector representing feature coefficients.
* $b$ is the model bias.
* The calibrated coefficients trained on our silver-label dataset are:
  * $w_1$ (TF-IDF Similarity): **6.5771**
  * $w_2$ (Keyword Overlap): **1.1188**
  * $w_3$ (Education Tier Delta): **-0.1350**
  * $w_4$ (Experience Gap): **0.4310**
  * Model Intercept (Bias $b$): **-4.68**

### 3.2 Score Calibration, Penalties, and Clamping
To prevent false-positive matching on fresher resumes and avoid over-fitting, we implemented three deterministic calibration equations:

1. **Fresher Mismatch Penalty**: If the candidate has 0 years of professional experience ($Y_{\text{Candidate}} = 0$) but the JD requires an experienced hire ($Y_{\text{JD}} \ge 2$ years), the raw probability is penalized:
   $$P_{\text{penalized}} = P(\mathbf{x}) \times 0.70$$
2. **Fresher Match Clamping**: Under the same fresher-mismatch condition, the score is strictly clamped to a continuous range to prevent high keyword matches from inflating the final screening score:
   $$P_{\text{final}} = \max\left(0.35, \min\left(P_{\text{penalized}}, 0.58\right)\right)$$
3. **Upper-Range Headroom Compression**: To prevent the system from outputting a literal "100% match" (which degrades recruiter trust), a piecewise ceiling compression is applied to all scores above 80%:
   $$\text{For } P(\mathbf{x}) > 0.80: \quad P_{\text{final}} = 0.80 + \left(\min\left(P(\mathbf{x}), 1.0\right) - 0.80\right) \times 0.60$$
   *This maps a raw 1.0 probability to a maximum calibrated ceiling of exactly 92%.*

---

## SECTION 4: COUNTERFACTUAL FAIRNESS & EXPLAINABILITY

### 4.1 Multi-Dimensional Counterfactual Fairness Probe
We evaluate demographic neutrality by generating four distinct counterfactual profiles for each resume:
* **Gender Attribute Swap**: Swaps gendered names (e.g., *John* $\leftrightarrow$ *Jane*, *Nandini* $\leftrightarrow$ *Rahul*) and gendered pronouns (*he/him* $\leftrightarrow$ *she/her*).
* **Age-Coded Language Swap**: Swaps age-coded vocabulary (e.g., *"recent graduate"*, *"digital native"* $\leftrightarrow$ *"seasoned veteran"*, *"years of experience"*).
* **Regional & Geographic Proxy Swap**: Substitutes regional addresses or city names with neutral counterparts.
* **Educational Prestige Proxy Swap**: Swaps Elite/Tier-1 institutions (e.g., *IIT*, *VJTI*, *Stanford*) with standard/general colleges.

For each counterfactual profile $\mathbf{x}'_j$, we compute the score delta:
$$\Delta_j = |P(\mathbf{x}) - P(\mathbf{x}'_j)|$$

The system audits the model in real-time, calculating the maximum variance:
$$\text{Max Variance} = \max\left(\Delta_{\text{gender}}, \Delta_{\text{age}}, \Delta_{\text{region}}, \Delta_{\text{prestige}}\right)$$

If $\text{Max Variance} < 0.02$ (2% score variation), the system outputs a `LOW_SENSITIVITY` certification badge. If it exceeds 2%, it triggers a `"Review Needed"` alert for recruiters.

### 4.2 SHAP Explainability & Waterfall Overrides
We use SHAP values to decompose the model's prediction into individual feature contributions:
$$g(\mathbf{x}') = \phi_0 + \sum_{i=1}^M \phi_i x'_i$$
Where:
* $\phi_0$ is the base expected value of the model (**-4.68**).
* $\phi_i$ is the Shapley value representing the impact of feature $i$.
* **The Contradiction Fix (Crucial Upgrade)**: To prevent trust-breaking contradictions where the UI badge says `Education: Exceeds` but the SHAP explanation says `Education level has negative impact` (caused by negative coefficients during linear modeling), we implemented a canonical override. The text output of $\phi_3$ is mathematically linked to the candidate's actual education status delta:
  * If $\text{Tier}_{\text{Candidate}} \ge \text{Tier}_{\text{JD}}$, the verbal explanation is forced to display as a **positive alignment** regardless of the raw sign of $\phi_3$.

### 4.3 Matched Evidence Snippets & Textual Evidence Strength
To provide inspectable proof to recruiters, we extract exact sentence snippets showing skills in context. 
* **Subject-Verb Ownership Checks**: To filter out passive mentions (e.g., *"worked in a team that used Docker"*), we classify snippets based on context:
  * **Strong Implementation** (Priority 3): Sentences containing active verbs (e.g., *built, implemented, engineered, deployed*) combined with direct subject ownership (*I, my, we*, or sentence-starting verbs).
  * **Workflow Context** (Priority 2): Generic mention of tools in a team environment.
  * **Brief Mention** (Priority 1): Passive mentions or sentences containing deprioritized qualifiers (e.g., *"basic familiarity with"*, *"fundamentals of"*).
* **Textual Evidence Strength Badges**: We replace the false precision of numeric confidence scores with categorical badges: `Strong evidence` (Priority 3), `Moderate evidence` (Priority 2), and `Weak evidence` (Priority 1).

---

## SECTION 5: REAL-WORLD CALIBRATION FIXES (OUR CORE CODE UPGRADES)

### 5.1 Section-Aware Date Parser
* **The Problem**: A standard date range regex (e.g., `2024 - 2026`) extracted from a fresher's resume was previously interpreted as 2 years of professional experience, inflating their match scores.
* **The Solution**: We implemented `segment_resume_sections(text)` to segment resumes into `education`, `experience`, and `general` blocks. The years extractor completely ignores date ranges found inside the `education` block.
* **Role Context Requirement**: To count as professional experience, a date range must reside in the `experience` block and be neighboring an employment marker/role (e.g., *internship, freelance, consultant, trainee, associate, research assistant*).

### 5.2 Skill-Focused Technical Requirement Filtering
* **The Problem**: TF-IDF n-grams extracted generic structure headings (e.g. `"title junior"`, `"responsibilities develop"`) and academic terms (e.g. `"computer science"`) as missing keywords, polluting the UI dashboard.
* **The Solution**: We introduced two centralized filtering lists inside `features.py`:
  * `GENERIC_ACTION_WORDS = {"build", "develop", "maintain", "optimize", "understanding", "responsibilities", "role", "candidate", "ability", "knowledge", "familiarity", ...}`
  * `EDUCATION_TERMS = {"computer science", "science", "related field", "bachelor", "master", "degree", ...}`
* Keywords matching these patterns are excluded from missing skill displays and recommendations, ensuring the extracted requirements are strictly technical skills (e.g., `"REST APIs"`, `"Docker Containerization"`).

---

## SECTION 6: EMPIRICAL PERFORMANCE & VALIDATION METRICS

We evaluated our model through two separate validation pipelines:

### 6.1 Internal Silver-Label Validation
* **Dataset Size**: 7,480 rows (6,317 negative labels, 1,163 positive labels).
* **Unique Resumes**: 2,399.
* **Unique JDs**: 72.
* **Baseline Accuracy**: 0.8443.
* **Row-Split Model Accuracy**: **99.53%** (ROC-AUC: **0.9998**).
* **Resume-Group Holdout Accuracy**: **99.26%** (ROC-AUC: **0.9999**).

### 6.2 External Transfer Validation (The Generalization Gap)
* **Dataset**: Labeled human resume-JD pair dataset (from Hugging Face).
* **Rows Evaluated**: 6,000 (excluding `Potential Fit` rows; 4,000 `No Fit`, 2,000 `Good Fit`).
* **Generalization Accuracy**: **65.91% ROC-AUC**.
* **Confusion Matrix**:
  $$\begin{pmatrix} 3207 & 793 \\ 1246 & 754 \end{pmatrix}$$
  *True Negatives: 3,207 | False Positives: 793 | False Negatives: 1,246 | True Positives: 754*
* **Academic Significance**: This exposes a significant **Generalization Gap** between internal silver labels (99.5% accuracy) and real-world human labels (65.9% ROC-AUC), highlighting the limitations of static matching and the importance of human-in-the-loop screening systems.

---

## SECTION 7: PROMPT FOR CLAUDE TO DRAFT THE PAPER

> **Claude, please write a highly rigorous, publication-grade academic research paper matching the style of IEEE Transactions or ACM conference proceedings.**
>
> **Paper Title**: AI-Based Resume Screening for Smart Recruitment: A Fairness-Aware and Explainable LLM-Agent Extension
>
> **Author**: Nandini Gupta (Roll No. 242011019)
> **Advisor**: Dr. Swati Chopade
> **Affiliation**: Veermata Jijabai Technological Institute (VJTI)
>
> Use the technical data, mathematical formulas, algorithms, calibration parameters, and evaluation metrics detailed in the prompt portfolio above.
>
> Ensure the paper is written in clean, professional LaTeX markup (using sections, standard equations, and tabular environments) and structured as:
> 1. **Abstract**
> 2. **Introduction** (discuss the shift from ATS keyword matchers to LLMs, and introduce the "black-box" and bias problems)
> 3. **Literature Review** (contrast LLM recruitment agents with traditional TF-IDF systems and detail ethical/fairness concerns)
> 4. **Methodology** (document feature engineering, Logistic Regression scoring engine, fresher penalties, clamps, upper ceilings, SHAP explanations, matched snippets, and multi-dimensional counterfactual probes)
> 5. **Experimental Evaluation** (present internal silver-label validation metrics and the external transfer validation results, including the ROC-AUC score and Confusion Matrix, highlighting the Generalization Gap)
> 6. **Discussion** (explore the implications of the generalization gap and ethical design limits)
> 7. **Conclusion & Future Work**
> 8. **References**
