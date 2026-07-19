# Power BI Dashboard Guide

Power BI is the business-facing decision layer for this project. The repository provides clean
datasets and a page specification; it does not claim that a `.pbix` file already exists.

## 1. Executive Retention Overview

Decision question: how large is the actionable churn-risk portfolio and how reliably does the
selected operating point detect churners?

- KPIs: scored customers, High Risk customers, Priority 1 customers, selected threshold,
  churn recall, churn precision, false negatives.
- Charts: risk-segment customer count; priority-level customer count.
- Filters: risk segment, priority, contract, internet service.

## 2. Customer Risk Portfolio

Decision question: which parts of the customer portfolio contain the most concentrated risk?

- KPIs: average churn probability, observed churn rate, average tenure, average monthly charges.
- Charts: segment distribution; contract mix by risk segment; probability distribution.
- Filters: risk segment, contract, tenure group, payment method, internet service.
- Detail table: customer ID, probability, risk, priority, value proxy, and observed indicators.

## 3. Churn Drivers

Decision question: which observed customer characteristics should inform investigation and
intervention design?

- KPIs: overall churn rate and number of validated observed indicators.
- Charts: higher-churn indicators; retention indicators; churn rate by contract/payment/service.
- Filters: indicator type and original feature.
- Caveat: use “associated with,” never causal wording.

## 4. Retention Action Queue

Decision question: who should be reviewed first and what intervention hypothesis is appropriate?

- KPIs: Priority 1 customers, customers assigned to each action, average priority score.
- Chart: customer count by suggested action and priority.
- Filters: priority, risk, suggested action, primary indicator, contract.
- Detail table: customer ID, probability, priority score, value proxy, action, primary reason,
  secondary reason, and suggested success metric.

## Recommended relationships

- Use `customer_retention_queue` as the customer-level decision table.
- Keep `risk_segment_summary` and `model_performance_summary` disconnected summary tables unless
  a dedicated dimension is created.
- Do not relate full-population historical summaries to the held-out queue by customer count.

## Refresh

Run `python run_pipeline.py`, then refresh the CSV sources in Power BI. Preserve the generated
column names so measures and visuals remain stable.
