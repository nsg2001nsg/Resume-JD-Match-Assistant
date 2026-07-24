# External Validation Report

This report evaluates the trained silver-label model against the labeled resume/JD pair dataset.
`Potential Fit` rows are excluded so the binary model is evaluated on `No Fit` vs `Good Fit`.

## Dataset

- Rows: 6000
- Label counts: {'0': 4000, '1': 2000}
- Split counts: {'train': 4685, 'test': 1315}

## Metrics

- ROC-AUC: 0.6591
- Confusion matrix: [[3207, 793], [1246, 754]]

## Limitations

- External labels may not use the same definition of fit as the silver-label generator.
- This should be read as transfer validation, not proof of hiring validity.
