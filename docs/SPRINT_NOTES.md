# Sprint Notes

## Sprint 1: Safer First Pass

Goal:

- Keep the existing resume-JD scoring pipeline intact.
- Make the app safer, more honest, and easier to demo.
- Work only in the sandbox copy.

Changed files:

- `app.py`
- `data_prep.py`
- `templates/index.html`
- `static/script.js`
- `static/style.css`

What changed:

- Renamed the product surface from recruitment screening to resume-JD match assistance.
- Changed the displayed score from "Match Probability" to "JD Fit Score".
- Kept `match_probability` in the API for compatibility, but added `match_score` as the safer preferred field.
- Changed fairness output from "Algorithmic Fairness: PASS/FAIL" to "Counterfactual Probe: Low sensitivity/Review needed".
- Added a fairness note saying the check is a limited sensitivity probe, not a full fairness audit.
- Secured upload filenames with `secure_filename`.
- Added unique temporary filenames for uploaded resumes.
- Rejected non-PDF uploads early.
- Rejected empty JD text early.
- Added a 5 MB upload size limit.
- Added a clearer error for scanned/unreadable PDFs.
- Deleted temporary resume uploads after processing.
- Removed hardcoded Flask debug mode; it now depends on `FLASK_DEBUG`.
- Replaced dynamic `innerHTML` rendering with safe DOM construction.
- Fixed visible broken text in the dashboard.
- Added a limitation note explaining that the tool supports review and does not make hiring decisions.
- Moved the heavy `SentenceTransformer` import into `build_dataset()` so app startup is much faster.

Verification:

- Python syntax check passed.
- JavaScript syntax check passed.
- Missing resume API validation returns `400`.
- Non-PDF API validation returns `400`.
- Filename sanitization converts unsafe names into safe temp paths.
- App import check improved from roughly 97 seconds to roughly 8 seconds.

Expected outcome:

- The original ML behavior still works.
- The project now makes more responsible claims.
- Demo startup is faster.
- Basic upload abuse and bad-input cases are handled cleanly.
- The code is better prepared for portfolio review.

## Sprint 2: Project Hygiene And Portfolio Structure

Goal:

- Make the sandbox easier to install, explain, and review as a portfolio project.
- Keep the Sprint 1 safety changes intact.

Changed files:

- `requirements.txt`
- `requirements-training.txt`
- `.env.example`
- `README.md`
- `DEPENDENCY_NOTES.md`
- `FIX_TRACKER.md`
- `SPRINT_NOTES.md`

What changed:

- Replaced the bloated environment-dump `requirements.txt` with a clean runtime dependency list.
- Added `requirements-training.txt` for heavier optional dataset/training scripts.
- Added `.env.example` for runtime configuration.
- Added a portfolio-ready README with purpose, setup, API contract, demo flow, architecture, and limitations.
- Added dependency notes explaining why the old full environment freeze should not be the portfolio dependency file.
- Added `FIX_TRACKER.md` as the living checklist for all future fixes.

Verification:

- Python syntax check passed.
- JavaScript syntax check passed.
- Missing resume API validation returns `400`.
- Non-PDF API validation returns `400`.
- Full sample resume/JD scoring returns `200`.
- Full sample response includes `match_score`, `shap_values`, and `fairness`.
- Active code/UI no longer contains the old overclaiming labels.

Expected outcome:

- A reviewer can understand and run the project more easily.
- Dependencies look intentional rather than copied from a whole machine environment.
- The project story is now aligned with responsible AI decision support.

## Sprint 3: Feature Extraction Robustness

Goal:

- Make the model inputs less brittle without replacing the model.
- Add tests for the feature behavior we now depend on.
- Keep Sprint 1 and Sprint 2 behavior intact.

Changed files:

- `data_prep.py`
- `features.py`
- `tests/test_feature_extraction.py`
- `FIX_TRACKER.md`
- `SPRINT_NOTES.md`

What changed:

- Added word-number experience parsing, such as "five years of experience".
- Added date-range experience parsing, such as "2019 - 2024" and "Jan 2021 - Present".
- Added keyword normalization and phrase-boundary matching.
- Prevented obvious false positives such as `java` matching `javascript`.
- Changed missing-keyword detection to use the same safer matcher as keyword overlap.
- Added unit tests for experience parsing, keyword matching, and education parsing.

Verification:

- Feature extraction unit tests passed: 9 tests.
- Python syntax check passed for changed feature files.
- Full sample resume/JD API scoring still returns `200`.
- Sample response still includes `match_score`, extracted features, and fairness result.

Expected outcome:

- The score inputs are less fragile.
- Recommendations based on missing keywords are less likely to be caused by substring errors.
- Future parser changes now have a test suite to protect previous behavior.

## Sprint 4: Model Validity & Fairness Redesign

Goal:

- Finalize F4 by evaluating and reporting on external datasets to expose the generalization gap.
- Redesign F5 to implement a broader, multi-dimensional counterfactual sensitivity suite (gender, age-coded terms, regional proxies, prestige proxies).

Changed files:

- `FIX_TRACKER.md`
- `SPRINT_NOTES.md`
- `fairness.py`
- `app.py`
- `static/script.js`

What changed:

- Finalized F4: Rebuilt the features on the full silver dataset, retrained the Logistic Regression model, and evaluated generalization on the Hugging Face 6,000-row external dataset.
- Exposed the model's performance mismatch: ~99.5% accuracy internally on silver-labeled data vs ~65.9% ROC-AUC externally on real-labeled pairs. This highlights the risk of relying only on heuristics and silver labels in automated hiring.
- Logged these findings in `reports/model_report.md` and `reports/external_validation.md` for full portfolio transparency.
- Redesigned `fairness.py` to evaluate four independent counterfactual sensitivity axes:
  - Gender: Names and pronoun swaps.
  - Age: Age-coded phrases (e.g., "seasoned veteran", "recent grad").
  - Geography/Region: Regional proxies and local institute mentions.
  - Prestige: Tier-1 university swaps.

Verification:

- Generated external validation and model training reports in `reports/`.
- Regression checks still passing.

