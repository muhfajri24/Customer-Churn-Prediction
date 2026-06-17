"""
Entry point project Customer Churn Prediction.

This file is intentionally simple so the execution flow is easy to understand:
1. Download or copy the dataset into the project folder.
2. Run data cleaning and feature engineering.
3. Train multiple classification models.
4. Evaluate model performance.
5. Save artifacts for follow-up analysis and Power BI.
"""

from src.churn_pipeline import run_project_pipeline


if __name__ == "__main__":
    run_project_pipeline()
