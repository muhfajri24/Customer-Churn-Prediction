# Risk Segmentation Method

## Approaches compared

- Business-linked: Low < 0.15; Medium 0.15 to < 0.30; High >= 0.30.
- Probability quantiles: three approximately equal-sized score groups.

## Observed holdout quality

Business-linked actual churn rates: Low 3.14%, Medium 15.25%, High 45.90%.

Quantile actual churn rates: Low 3.19%, Medium 20.26%, High 56.17%.

## Recommendation

Use the business-linked method. Both approaches produce monotonic risk separation, but the business-linked boundaries keep High Risk identical to the validated intervention population. Quantiles remain a useful portfolio comparison, not the operational rule.
