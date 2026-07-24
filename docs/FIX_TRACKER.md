# Fix Tracker

This file is the living checklist for the sandbox upgrade. Keep it updated after every fix so the project stays coherent and reviewable.

## Current Positioning

- Project name: Resume-JD Match Assistant
- Role: decision-support tool for human review
- Core behavior retained: JD-based scoring, explainability, counterfactual probe, recommendations
- Explicit non-goal: autonomous hiring decisions

## Fixes

| ID | Fix Area | Status | Outcome | Verification |
| --- | --- | --- | --- | --- |
| F1 | Safer first pass | Done | Upload safety, honest UI labels, safer DOM rendering, faster app import | Python syntax ok; JS syntax ok; missing file/non-PDF return 400; full score API returns 200 |
| F2 | Project hygiene and portfolio structure | Done | Clean dependency files, README, run/demo guidance, dependency notes, env example | Regression checks passed |
| F3 | Feature extraction robustness | Done | Word-number years, date-range experience, safer keyword matching, missing-keyword matching, feature tests | Feature tests passed; regression checks passed |
| F4 | Model validity and evaluation report | Done | Rebuilt silver features, retrained LR model, ran external validation on 6k HF pairs (65.9% ROC-AUC), created caveat reports | Reports generated; regression checks passed |
| F5 | Fairness probe redesign | In progress | Broader multi-dimensional counterfactual suite (gender, age, region, prestige) and honest reporting | Pending |
| F6 | Explainability upgrade | Not started | Plain-English explanations and evidence snippets | Pending |
| F7 | Recommendation quality | Not started | Evidence-based, ethical, non-fabrication advice | Pending |
| F8 | Offline/frontend polish | Not started | Local assets or graceful CDN fallback, polished dashboard | Pending |
| F9 | Portfolio story | Not started | Model card, fairness card, architecture diagram, demo script | Pending |

## Sprint 1 Notes

- Completed in `SPRINT_NOTES.md`.
- Important performance fix: moved `SentenceTransformer` import into dataset-building path so app import fell from about 97 seconds to about 8 seconds.

## Sprint 2 Goals

- Replace the environment-dump `requirements.txt` with clean dependency files. Done.
- Preserve the old environment dump as legacy evidence. Done via baseline pointer in `DEPENDENCY_NOTES.md`.
- Add a portfolio-ready `README.md`. Done.
- Add `.env.example`. Done.
- Document safe setup, run, testing, limitations, and demo flow. Done.
- Re-run Sprint 1 regression checks. Done.

## Regression Checks To Keep Running

- Feature extraction tests: `python -B -m unittest tests.test_feature_extraction`.
- `python -B -c "... ast.parse ..."` for Python syntax.
- `node --check static/script.js` for JavaScript syntax.
- Flask test client: missing resume returns `400`.
- Flask test client: non-PDF returns `400`.
- Flask test client: sample resume/JD returns `200` with `match_score`, `shap_values`, and `fairness`.
- Search check: no UI text says `Algorithmic Fairness` or `Match Probability`.
