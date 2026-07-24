# RESEARCH PAPER DRAFT

**Title**: AI-Based Resume Screening for Smart Recruitment: A Fairness-Aware and Explainable LLM-Agent Extension

**Author**: Nandini Gupta (Roll No. 242011019)  
**Advisor/Guide**: Dr. Swati Chopade  
**Affiliation**: Department of Computer Applications, Veermata Jijabai Technological Institute (VJTI), Mumbai, India  

---

## Abstract
Modern recruitment pipelines are overwhelmed by the volume of candidate applications, necessitating automated screening solutions. While traditional Applicant Tracking Systems (ATS) rely on rigid keyword matching, recent Large Language Model (LLM) multi-agent architectures offer deep semantic parsing but function as opaque "black boxes" that risk propagating demographic bias. This paper presents a fairness-aware and explainable architectural extension to LLM-agent recruitment pipelines. By replacing LLM-based scoring with an interpretable Logistic Regression classifier and applying Shapley Additive exPlanations (SHAP) alongside a multi-dimensional counterfactual fairness probe, we achieve high-fidelity alignment screening with audited demographic neutrality. We implement real-world calibration mechanisms—including section-aware parsing to eliminate fresher experience inflation, continuous mismatch score clamping, and skill-focused keyword filtering. Empirical evaluation on a 7,480-row internal silver-label dataset demonstrates a row-split ROC-AUC of 0.9998, while transfer validation on a 6,000-row external human-labeled dataset reveals a generalization gap with a ROC-AUC of 0.6591, demonstrating the crucial importance of human-in-the-loop screening systems.

---

## 1. Introduction
The efficiency and fairness of candidate selection are critical to organizational success. In modern high-volume hiring environments, manual review of thousands of resumes per job role is logistically prohibitive, leading to recruiter fatigue and inconsistent assessments. The recruitment industry has widely adopted Applicant Tracking Systems (ATS) to automate initial filtering. However, legacy ATS implementations are predominantly keyword-based, failing to capture semantic synonyms, context, or project depth, and are easily bypassed by candidate keyword-stuffing.

The advent of transformer-based Large Language Models (LLMs) has enabled intelligent semantic parsing. As proposed in foundational frameworks such as *Gan et al. (2024)*, multi-agent LLM systems can collaboratively segment, extract, summarize, and score candidate resumes. Nonetheless, directly utilizing LLMs for scoring presents significant ethical and operational risks:
1. **Explainability Deficit**: Deep learning scoring layers do not provide mathematically traceable justifications. Recruiter trust is degraded when decisions are represented by a single numeric score without feature-level attribution.
2. **Demographic Bias Propagation**: LLMs can perpetuate biases present in their training data, discriminating against candidates based on names, regional pronouns, age-coded language, or university prestige.
3. **Calibrational Mismatch**: LLMs struggle with precise mathematical boundary rules, often misinterpreting educational dates (e.g. academic range `2024-2026`) as professional tenure, thereby inflating fresher suitability scores.

To resolve these challenges, this paper introduces a fairness-aware and explainable architectural extension to LLM-agent recruitment systems. We replace upstream LLM scoring with an interpretable Logistic Regression classifier built on engineered features (TF-IDF semantic similarity, keyword overlap ratio, education tier delta, and professional experience gap). We introduce real-time SHAP explainability, evidence snippet subject-verb verification, multi-dimensional counterfactual fairness auditing, and robust date/experience calibration.

---

## 2. Literature Review
The automation of resume screening has transitioned through three major technological generations:

1. **Keyword-Based Matchers**: Early ATS platforms computed basic term frequencies or utilized Boolean search queries. These systems were highly susceptible to synonym neglect (e.g., matching "Python Developer" but ignoring "Backend Engineer with Django experience") and keyword manipulation.
2. **Shallow Machine Learning**: Classification models (e.g., Random Forests, Support Vector Machines) trained on TF-IDF or Word2Vec representations improved classification bounds. However, they lacked structural parsing capabilities, treating candidate resumes as unstructured bags of words.
3. **LLM-Agent Frameworks**: Modern approaches, such as the multi-agent system designed by *Gan et al. (2024)*, partition resume processing across sequential LLM agents (Extraction, Summarization, and Grading). By utilizing the contextual understanding of LLMs, these systems extract structured candidate profiles and generate human-like summaries.

Despite their semantic richness, LLM-agent frameworks fail to satisfy the transparency requirements of ethical AI deployment. Studies on algorithmic bias (e.g., *Fairness and Bias in AI Recruitment, 2023-2024*) show that machine learning models easily pick up on proxies for protected classes, such as geographic addresses or gendered names. Our work addresses this gap by establishing a theoretical and empirical framework for counterfactual auditing and feature attribution.

---

## 3. Proposed Methodology

### 3.1 System Overview
The proposed system integrates an explainability and counterfactual evaluation layer between the upstream LLM-agent extraction pipeline and the final recruiter interface. 

```
[Unstructured Resume] -> [LLM Sentence Classifier] -> [Structured JSON Profile]
                                                             |
                                                             v
[Job Description]     -> [Keyword & Tier Extractor]    -> [Feature Engineering]
                                                             |
                                                             v
[SHAP Attribution]    <- [Scoring & Calibration Clamps] <- [Logistic Regression]
        |                            |
        v                            v
[Evidence Snippets]    [Counterfactual Fairness Probe] -> [Recruiter Dashboard]
```

### 3.2 Feature Representation
For each candidate-job pair, the system constructs a 4-dimensional feature vector $\mathbf{x} = [x_1, x_2, x_3, x_4]^T$:

1. **TF-IDF Cosine Similarity ($x_1$)**: Measures the overall vocabulary alignment between the resume and the job description.
2. **Keyword Overlap Ratio ($x_2$)**: Calculates the fraction of job-specific technical keywords documented in the candidate's profile:
   $$x_2 = \frac{|K_{\text{Resume}} \cap K_{\text{JD}}|}{|K_{\text{JD}}|}$$
3. **Education Level Delta ($x_3$)**: Standardizes academic degrees into four tiers (Tier 4: Postgraduate/Doctorate; Tier 3: Professional Bachelor's; Tier 2: General Bachelor's/Diploma; Tier 1: Foundational). The feature represents:
   $$x_3 = \text{Tier}_{\text{Candidate}} - \text{Tier}_{\text{JD}}$$
4. **Experience Gap ($x_4$)**: Quantifies the difference in years between candidate professional experience and the JD requirements:
   $$x_4 = Y_{\text{Candidate}} - Y_{\text{JD}}$$

### 3.3 Scoring Engine & Calibration Heuristics
The classification score is computed via the logistic sigmoid function:
$$P(y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

To ensure real-world calibration and prevent score inflation, we apply three post-prediction heuristics:
* **Fresher Mismatch Penalty**: If $Y_{\text{Candidate}} = 0$ and $Y_{\text{JD}} \ge 2$, a 30% penalty is applied:
  $$P_{\text{penalized}} = P(y = 1 | \mathbf{x}) \times 0.70$$
* **Continuous Clamping**: For fresher-mismatch cases, the score is clamped to prevent high keyword matches from producing false matches:
  $$P_{\text{calibrated}} = \max\left(0.35, \min\left(P_{\text{penalized}}, 0.58\right)\right)$$
* **Upper Bound Ceiling**: To maintain user trust and avoid implying absolute fit, scores above 80% are scaled:
  $$P_{\text{calibrated}} = 0.80 + \left(\min\left(P(\mathbf{x}), 1.0\right) - 0.80\right) \times 0.60$$
  *This compresses a raw 100% prediction to a maximum ceiling of 92%.*

### 3.4 Multi-Dimensional Counterfactual Fairness Probe
To verify demographic neutrality, we dynamically alter the candidate's profile text across four sensitivity axes to generate counterfactual feature vectors $\mathbf{x}'_j$:
1. **Gender Swap**: Modifying names (*Jane* $\leftrightarrow$ *John*) and pronouns (*she/her* $\leftrightarrow$ *he/him*).
2. **Age Swap**: Replacing age-coded vocabulary (*recent graduate* $\leftrightarrow$ *industry veteran*).
3. **Region Swap**: Replacing addresses and city proxies.
4. **Prestige Swap**: Substituting Tier-1 institutions (*VJTI*, *IIT*) with standard general colleges.

We compute the maximum score variance:
$$\text{Max Variance} = \max\left(|P(\mathbf{x}) - P(\mathbf{x}'_{\text{gender}})|, |P(\mathbf{x}) - P(\mathbf{x}'_{\text{age}})|, |P(\mathbf{x}) - P(\mathbf{x}'_{\text{region}})|, |P(\mathbf{x}) - P(\mathbf{x}'_{\text{prestige}})|\right)$$

A threshold of $0.02$ is enforced; variations below 2% trigger a `LOW_SENSITIVITY` pass badge.

### 3.5 Explainability Override & Evidence Snippets
Feature contributions are mapped via SHAP values:
$$g(\mathbf{x}') = \phi_0 + \sum_{i=1}^M \phi_i x'_i$$
To prevent cognitive contradictions where a candidate exceeding educational requirements yields a negative SHAP coefficient (due to model weight distributions), we force the verbal explanation of $\phi_3$ to register positive alignment when $x_3 \ge 0$.

Furthermore, matched resume sentences are extracted and prioritized using active verbs and direct subject ownership checks, filtering out passive or brief mentions.

---

## 4. Empirical Evaluation & Results

### 4.1 Internal Silver-Label Performance
The Logistic Regression scoring engine was trained on a silver-label dataset generated via high-fidelity matching heuristics. The dataset consists of 7,480 rows representing 2,399 unique resumes and 72 job descriptions.

* **Row-Split Model Accuracy**: **99.53%** (ROC-AUC: **0.9998**)
* **Resume-Group Holdout Accuracy**: **99.26%** (ROC-AUC: **0.9999**)
* **Trained Coefficients**:
  * $w_1$ (TF-IDF Similarity): **6.5771**
  * $w_2$ (Keyword Overlap): **1.1188**
  * $w_3$ (Education Tier Delta): **-0.1350**
  * $w_4$ (Experience Gap): **0.4310**
  * Intercept ($b$): **-4.68**

### 4.2 External Validation & Generalization Gap
To test the transfer viability of the model, we evaluated it against an external Hugging Face dataset consisting of 6,000 human-labeled resume-JD pairs (4,000 "No Fit", 2,000 "Good Fit").

* **External ROC-AUC**: **0.6591**
* **Confusion Matrix**:
  $$\begin{pmatrix} 3207 & 793 \\ 1246 & 754 \end{pmatrix}$$

This significant drop in accuracy (from **99.5%** to **65.9%**) highlights the **Generalization Gap**. The silver-label heuristics represent an idealized rule set, whereas real-world human recruitment decisions are subject to complex, noisy, and non-linear patterns, confirming the limits of autonomous screening.

### 4.3 Fairness Sensitivity Evaluation
The counterfactual fairness probe was executed across the evaluation profiles. The model demonstrated extremely high demographic neutrality:
* **Gender Sensitivity Variance**: **0.0000**
* **Age-Coded Language Variance**: **0.0000**
* **Regional address Proxy Variance**: **0.0000**
* **College Prestige Variance**: **0.0000**
* **Result**: **PASS** (Zero variance under demographic swapping, certifying robustness).

---

## 5. Conclusion & Future Work
This paper presented a fairness-aware and explainable architectural extension for LLM-agent resume screening. By replacing opaque LLM scoring with an interpretable Logistic Regression classifier and applying post-prediction calibration clamps, we successfully mitigated fresher score inflation and eliminated keyword-stuffing exploits. Shapley Additive exPlanations and active evidence snippets provide recruiters with clear mathematical justifications for candidate scoring, while the counterfactual fairness probe certifies demographic neutrality. Future work will investigate transition to local, lightweight semantic encoders (e.g., SentenceTransformers) to replace TF-IDF cosine similarity, and multilingual counterfactual mapping.
