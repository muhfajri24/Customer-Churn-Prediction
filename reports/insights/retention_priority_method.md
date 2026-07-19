# Retention Priority Method

The priority score is a transparent decision heuristic, not another trained model.

## Formula

- 60% churn probability.
- 25% customer-value proxy.
- 15% intervention urgency.

The value proxy combines 60% MonthlyCharges and 40% TotalCharges after each is scaled to its training-data 95th percentile and capped at 100%. Reference values: MonthlyCharges=107.3675; TotalCharges=6979.8250. It is not Customer Lifetime Value.

Urgency combines 60% inverse tenure (capped at 72 months) and 40% month-to-month contract status.

## Priority mapping

- Priority 1: High Risk and priority score >= 70.
- Priority 2: other High Risk customers.
- Priority 3: Medium Risk customers.
- Monitor: Low Risk customers.

The score-70 boundary is an explainable portfolio rule, not a proven economic optimum. It must be revisited when real capacity, customer value, and intervention outcomes exist.
