# Power BI Dashboard Guide

This folder contains datasets ready to be imported into Power BI:

- `customer_churn_scoring.csv`
- `summary_by_contract.csv`
- `summary_by_payment_method.csv`
- `summary_by_internet_service.csv`

## Suggested Dashboard Pages

1. Overview
   - KPI cards: Total Customers, Churn Rate, Average Monthly Charges, Average Tenure
   - Donut chart: Churn vs Non-Churn
   - Bar chart: Churn rate by contract

2. Customer Risk
   - Table: `customerID`, `Contract`, `PaymentMethod`, `tenure`, `MonthlyCharges`, `churn_probability`, `risk_segment`
   - Slicers: `risk_segment`, `Contract`, `InternetService`

3. Business Drivers
   - Bar chart: Churn rate by payment method
   - Bar chart: Churn rate by internet service
   - Scatter plot: `tenure` vs `MonthlyCharges`, colored by churn outcome

## Suggested DAX Measures

```DAX
Total Customers = COUNTROWS(customer_churn_scoring)
Churn Customers = CALCULATE(COUNTROWS(customer_churn_scoring), customer_churn_scoring[actual_churn_flag] = 1)
Churn Rate = DIVIDE([Churn Customers], [Total Customers], 0)
Average Monthly Charges = AVERAGE(customer_churn_scoring[MonthlyCharges])
Average Tenure = AVERAGE(customer_churn_scoring[tenure])
```
