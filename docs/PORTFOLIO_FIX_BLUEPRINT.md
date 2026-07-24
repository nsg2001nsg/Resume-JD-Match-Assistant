# Portfolio Fix Blueprint

This sandbox is the working copy for turning the MCA prototype into a portfolio-grade resume/JD screening assistant. The original project root should stay untouched while fixes are developed here.

## Product Repositioning

Current risky claim:

- Automated resume screening assistant that scores candidates for hiring.

Safer portfolio claim:

- Explainable resume-to-job-description matching assistant for human reviewers.
- Provides text-match evidence, uncertainty, missing-requirement analysis, counterfactual checks, and improvement suggestions.
- Does not make autonomous hiring decisions.

## Track 1: Project Hygiene And Reproducibility

Problems:

- `requirements.txt` is a full environment dump with unrelated packages.
- No clean install path, no README workflow, no test command.
- Generated artifacts and source files are mixed together.

Fixes:

- Create minimal `requirements.in` / clean `requirements.txt`.
- Add `.env.example` and document local run commands.
- Move model artifacts and generated plots into `models/` and `reports/`.
- Add a `README.md` with scope, limitations, architecture, setup, and demo script.

Acceptance checks:

- Fresh virtual environment installs successfully.
- `python app.py` starts without missing dependency errors.
- README explains that this is decision support, not automated hiring.

## Track 2: API And File Safety

Problems:

- Uploaded filename is used directly.
- Uploaded files are stored under a shared folder.
- Flask debug mode is hardcoded.
- PDF extraction silently fails.

Fixes:

- Use `werkzeug.utils.secure_filename`.
- Generate unique upload names.
- Restrict extension and file size.
- Return clear extraction diagnostics.
- Move `debug=True` behind an environment variable.
- Add cleanup for temporary uploads.

Acceptance checks:

- Path traversal filenames cannot escape the upload directory.
- Non-PDF files are rejected.
- Scanned or unreadable PDFs return a useful error message.
- App does not run debug mode by default.

## Track 3: Feature Extraction Robustness

Problems:

- Experience extraction only catches simple numeric patterns.
- Education extraction is keyword-tier based and brittle.
- Keyword overlap can reward keyword stuffing.
- Missing keyword detection uses substring matching.

Fixes:

- Add section-aware parsing for skills, education, and experience.
- Support date-range experience estimation as a fallback.
- Normalize skills using a curated skill alias map.
- Use token/phrase matching instead of raw substring checks.
- Report extraction confidence for each feature.

Acceptance checks:

- Unit tests cover numeric years, date ranges, word-number years, missing years, degree variants, and false positives.
- Feature output includes confidence and source snippets where possible.

## Track 4: Model And Dataset Validity

Problems:

- Labels are silver labels generated from similarity heuristics.
- Model learns to reproduce the label generator.
- Very high ROC-AUC may be an artifact of the labeling process.
- No calibration or uncertainty reporting.

Fixes:

- Rename current score to `match_score`, not `hire_probability`.
- Add evaluation reports that explicitly separate silver-label performance from real-world validity.
- Add calibration curve and confidence bands.
- Add stress tests for keyword stuffing, irrelevant experience, and short resumes.
- Consider a hybrid scoring model: hard requirement checks plus semantic similarity plus transparent weighting.

Acceptance checks:

- Report says "validated against silver labels" instead of implying real hiring accuracy.
- Test suite includes adversarial resumes.
- API returns uncertainty or confidence notes.

## Track 5: Fairness Evaluation

Problems:

- Current fairness test only swaps a small set of gendered names and pronouns.
- A pass result can be misleading because the model features ignore most identity terms.
- No group-level adverse-impact metrics.

Fixes:

- Rename the current test to `gender_counterfactual_probe`.
- Add probes for name, pronoun, age-coded terms, college prestige, location, employment gap, and disability-related wording.
- Report score deltas across a counterfactual test suite.
- Add documentation explaining what the fairness check can and cannot prove.

Acceptance checks:

- Fairness output says "probe passed" rather than "algorithmically fair".
- Report includes max delta, mean delta, and examples.
- UI includes limitation text in an appropriate place.

## Track 6: Explainability

Problems:

- SHAP values are shown as log-odds, which may confuse users.
- Explanations are feature-level only; they do not show resume/JD evidence.
- SHAP initialization at app startup can slow or break demos.

Fixes:

- Convert explanations into human-readable contribution summaries.
- Add evidence snippets for matched/missing requirements.
- Lazy-load or precompute explainability resources.
- Label charts clearly as model-feature impact, not causal reasons.

Acceptance checks:

- Output explains which requirements were matched, missing, or weak.
- UI does not imply SHAP proves causality.
- App startup is fast enough for demo use.

## Track 7: Recommendations

Problems:

- Recommendations are threshold rules.
- They can sound too authoritative.
- They may encourage keyword stuffing.

Fixes:

- Tie every recommendation to a missing JD requirement or weak evidence.
- Use wording like "To better demonstrate fit..." instead of "candidate lacks...".
- Separate candidate-facing feedback from recruiter-facing risk notes.
- Add guardrails against suggesting fabricated experience.

Acceptance checks:

- Recommendations include evidence and ethical wording.
- No recommendation asks the user to fake credentials or years of experience.

## Track 8: Frontend Polish

Problems:

- Broken mojibake characters are visible.
- External CDN assets can break offline demos.
- `innerHTML` is used for dynamic text.
- UI claims "Algorithmic Fairness: PASS", which overstates the check.

Fixes:

- Fix encoding issues.
- Vendor or locally bundle Chart.js, or provide a graceful fallback.
- Replace `innerHTML` with safe DOM construction.
- Rename fairness badge to "Counterfactual Probe".
- Add a compact limitations panel.

Acceptance checks:

- No broken characters in the UI.
- Demo works without internet.
- Dynamic recommendations are rendered safely.

## Track 9: Portfolio Story

Problems:

- The current project can be challenged as ethically overclaimed.
- Evaluation metrics may be misunderstood.

Fixes:

- Add an architecture diagram.
- Add a model card.
- Add a fairness card.
- Add a "Known Limitations" section.
- Add a short demo script that shows both strengths and failure handling.

Acceptance checks:

- Interviewer can see mature judgment: limitations are named, mitigated, and tested.
- The project reads as responsible ML engineering, not just a classifier demo.

## Suggested Order

1. Project hygiene and safe app behavior.
2. Rename/reposition risky claims in API and UI.
3. File upload hardening.
4. Frontend encoding and safe rendering.
5. Feature extraction tests and fixes.
6. Better evaluation report with silver-label caveats.
7. Fairness probe redesign.
8. Evidence-based recommendations.
9. Model card, fairness card, and README portfolio polish.

## First Target

Start with Tracks 1, 2, and 8 because they reduce demo risk quickly and do not require retraining the model.
