# Dependency Notes

Sprint 2 replaces the original environment-dump `requirements.txt` with a clean dependency list.

The original full freeze is still available in the preserved baseline project at:

- `D:\ResumeAutomation\requirements.txt`

Why this matters:

- The old file included unrelated packages such as Django, TensorFlow, Stripe, Streamlit, PySpark, notebooks, and developer tooling.
- A portfolio reviewer should see the dependencies the project actually needs, not a machine-wide environment snapshot.
- Training and dataset-building dependencies are separated from runtime dependencies so the Flask app remains easier to install and run.
