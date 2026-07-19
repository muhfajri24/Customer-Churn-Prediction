# Threshold Selection

Decision objective: identify a high proportion of actual churners while keeping retention outreach operationally manageable.

Selection data: a stratified validation subset drawn only from the training set. The held-out test set was not used for model or threshold selection.

Candidate thresholds: 0.25 to 0.70 in increments of 0.05. Candidates must reach precision >= 0.45 and flag <= 50% of validation customers.

Selection score: 30% churn recall + 30% churn F1 + 25% PR-AUC + 15% churn precision.

Selected model: random_forest

Selected threshold: 0.30

Validation precision: 0.4629

Validation recall: 0.8663

Validation churn F1: 0.6034

Validation customers flagged: 700 (49.68%)
