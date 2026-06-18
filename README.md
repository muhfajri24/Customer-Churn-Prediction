# Customer Churn Prediction

This project predicts telecom customers who are likely to churn and turns the model output into retention-focused business insight.

## What This Project Does

- Cleans and prepares the Telco Customer Churn dataset
- Builds and compares `Logistic Regression`, `Random Forest`, and `XGBoost`
- Evaluates model performance with classification metrics
- Exports model artifacts, figures, and business-ready outputs for reporting

## Why It Matters

This project shows how Python can be used to move from raw customer data to a practical churn prediction workflow that supports retention strategy, risk prioritization, and stakeholder communication.

## Primary Workflow

Python is the main way to run this project.

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Main entrypoints:

- `run_pipeline.py` for the full end-to-end run
- `src/churn_pipeline.py` for the reusable pipeline logic

## Optional Notebook

Notebook files are included only as a secondary option for walkthroughs, recruiter demos, or quick testing.

- `notebooks/customer_churn_walkthrough.py`
- `notebooks/customer_churn_walkthrough.ipynb`

## Dataset

Source dataset: Kaggle `blastchar/telco-customer-churn`

The pipeline downloads the dataset automatically and stages it in `data/raw/`.

## Tools

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- Power BI

## Project Structure

```text
Customer Churn Prediction/
|-- data/
|-- models/
|-- notebooks/
|-- powerbi/
|-- reports/
|-- src/
|   `-- churn_pipeline.py
|-- run_pipeline.py
|-- requirements.txt
`-- README.md
```
