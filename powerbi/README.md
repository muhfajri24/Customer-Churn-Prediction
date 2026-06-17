# Power BI Dashboard Guide

Folder ini berisi dataset siap import ke Power BI:

- `customer_churn_scoring.csv`
- `summary_by_contract.csv`
- `summary_by_payment_method.csv`
- `summary_by_internet_service.csv`

Saran halaman dashboard:

1. Overview
   - KPI: Total Customer, Churn Rate, Avg Monthly Charges, Avg Tenure
   - Donut chart: Churn vs Non-Churn
   - Bar chart: Churn rate by Contract

2. Customer Risk
   - Table: customerID, Contract, PaymentMethod, tenure, MonthlyCharges, churn_probability, risk_segment
   - Slicer: risk_segment, Contract, InternetService

3. Business Driver
   - Bar chart: Churn rate by PaymentMethod
   - Bar chart: Churn rate by InternetService
   - Scatter plot: tenure vs MonthlyCharges dengan warna berdasarkan churn

Measure DAX dasar yang bisa dipakai:

```DAX
Total Customers = COUNTROWS(customer_churn_scoring)
Churn Customers = CALCULATE(COUNTROWS(customer_churn_scoring), customer_churn_scoring[actual_churn_flag] = 1)
Churn Rate = DIVIDE([Churn Customers], [Total Customers], 0)
Average Monthly Charges = AVERAGE(customer_churn_scoring[MonthlyCharges])
Average Tenure = AVERAGE(customer_churn_scoring[tenure])
```
