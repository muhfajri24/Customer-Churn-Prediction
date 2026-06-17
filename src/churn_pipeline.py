"""
Main pipeline for the Customer Churn Prediction project.

The functions in this file are intentionally separated by analysis stage so the
workflow is easier to read, debug, and explain during demos or interviews.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import kagglehub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Resolve the project root relative to this file so the pipeline still works
# even when it is executed from a different working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
POWERBI_DIR = PROJECT_ROOT / "powerbi"


@dataclass
class TrainedModelResult:
    """
    Store the training result for one model in a compact object.

    This avoids scattering model outputs across many separate variables, which
    tends to make analytical code harder to maintain.
    """

    name: str
    pipeline: Pipeline
    y_pred: pd.Series
    y_proba: pd.Series
    metrics: Dict[str, float]


def ensure_directories() -> None:
    """Create the project output folders if they do not already exist."""
    for folder in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, POWERBI_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    """
    Download the dataset with KaggleHub and copy it into the project folder.

    Keeping the CSV inside the repository structure makes the project easier to
    test and keeps all artifacts in one place.

    The lookup order is:
    1. use the local project file if it already exists,
    2. use the local Kaggle cache if available,
    3. fall back to an online download.

    This makes reruns more reliable even when internet access is not available.
    """
    target_csv = RAW_DIR / "telco_customer_churn.csv"

    # Reuse the local project copy when it already exists to avoid repeated downloads.
    if target_csv.exists():
        return target_csv

    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "blastchar" / "telco-customer-churn"
    cache_candidates = sorted(cache_root.glob("versions/*/*.csv"))

    # Reuse the local Kaggle cache when it is available.
    if cache_candidates:
        shutil.copy2(cache_candidates[-1], target_csv)
        return target_csv

    kaggle_path = Path(kagglehub.dataset_download("blastchar/telco-customer-churn"))
    source_csv = next(kaggle_path.glob("*.csv"))
    shutil.copy2(source_csv, target_csv)
    return target_csv


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset so it is ready for modeling.

    Cleaning steps:
    - remove unnecessary whitespace,
    - convert data types,
    - drop duplicates,
    - handle missing values.
    """
    cleaned = df.copy()

    # `TotalCharges` is often read as an object column because blank strings
    # appear in the source file.
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    # Trim whitespace from categorical columns when present.
    for column in cleaned.select_dtypes(include="object").columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()

    # Convert empty strings into NA values for simpler downstream handling.
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA})

    # Remove duplicates if they exist.
    cleaned = cleaned.drop_duplicates()

    # Keep customer IDs in the main table for traceability. They are excluded
    # from model features later in the pipeline.
    return cleaned


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add business-relevant derived features.

    The feature engineering stays intentionally simple so the project is easy
    to explain during presentations.
    """
    featured = df.copy()

    # Subscription-length segmentation makes retention patterns easier to read.
    featured["TenureGroup"] = pd.cut(
        featured["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"],
    )

    # The ratio of total charges to tenure approximates average spend across the
    # customer relationship.
    featured["AvgMonthlySpend"] = featured["TotalCharges"] / featured["tenure"].replace(0, 1)

    # A simple proxy for customers with relatively low support-service adoption.
    protection_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]
    featured["SupportServiceCount"] = featured[protection_columns].eq("Yes").sum(axis=1)

    # Convert the target to 0/1 so the classifiers can be trained directly.
    featured["ChurnFlag"] = featured["Churn"].map({"No": 0, "Yes": 1})

    return featured


def save_cleaned_dataset(df: pd.DataFrame) -> None:
    """Save the cleaned and feature-engineered dataset."""
    df.to_csv(PROCESSED_DIR / "customer_churn_cleaned.csv", index=False)


def prepare_train_test_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str], pd.DataFrame]:
    """
    Separate features, target, and customer ID.

    `customerID` is kept separately so final predictions can be linked back to
    real customers for dashboards and business recommendations.
    """
    feature_df = df.copy()
    customer_reference = feature_df[["customerID", "Churn"]].copy()

    y = feature_df["ChurnFlag"]
    X = feature_df.drop(columns=["Churn", "ChurnFlag"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features, customer_reference


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build a unified preprocessing pipeline.

    Numeric features:
    - median imputation
    - scaling

    Categorical features:
    - most-frequent imputation
    - one-hot encoding
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """
    Build the three models used in this project.

    All models are wrapped in the same pipeline so preprocessing stays
    consistent across experiments.
    """
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> TrainedModelResult:
    """
    Generate predictions and evaluation metrics for one model.

    Main metrics:
    - accuracy
    - precision
    - recall
    - F1-score
    - ROC-AUC
    """
    y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index, name="prediction")
    y_proba = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=y_test.index, name="churn_probability")

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    return TrainedModelResult(
        name=name,
        pipeline=pipeline,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
    )


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> List[TrainedModelResult]:
    """Train every model and return their evaluation results."""
    models = build_models(preprocessor)
    results: List[TrainedModelResult] = []

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(name, pipeline, X_test, y_test))

    return results


def save_metrics(results: List[TrainedModelResult]) -> pd.DataFrame:
    """Save the cross-model metrics summary to CSV."""
    metrics_df = pd.DataFrame(
        [{"model": result.name, **result.metrics} for result in results]
    ).sort_values(by="roc_auc", ascending=False)

    metrics_df.to_csv(REPORT_DIR / "metrics_summary.csv", index=False)
    return metrics_df


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Save the churn class distribution chart."""
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=200)
    plt.close()


def plot_roc_curves(results: List[TrainedModelResult], y_test: pd.Series) -> None:
    """Create a ROC-curve comparison for all models."""
    plt.figure(figsize=(8, 6))

    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result.y_proba)
        plt.plot(fpr, tpr, label=f"{result.name} (AUC={result.metrics['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curve_comparison.png", dpi=200)
    plt.close()


def plot_confusion_matrices(results: List[TrainedModelResult], y_test: pd.Series) -> None:
    """Save the confusion matrix for each model."""
    for result in results:
        matrix = confusion_matrix(y_test, result.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
        disp.plot(cmap="Blues", colorbar=False)
        plt.title(f"Confusion Matrix - {result.name}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"confusion_matrix_{result.name}.png", dpi=200)
        plt.close()


def extract_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract feature names from the preprocessing step.

    This matters because one-hot encoding expands the original categorical
    columns into many derived dummy features.
    """
    return preprocessor.get_feature_names_out().tolist()


def save_feature_importance(
    best_result: TrainedModelResult,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Calculate feature importance with permutation importance.

    This is a model-agnostic approach, so it works across different pipeline
    architectures.
    """
    importance = permutation_importance(
        best_result.pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)

    importance_df.to_csv(REPORT_DIR / "feature_importance_best_model.csv", index=False)
    return importance_df


def save_model_artifacts(results: List[TrainedModelResult]) -> None:
    """Save the trained models for reuse."""
    for result in results:
        joblib.dump(result.pipeline, MODEL_DIR / f"{result.name}.joblib")


def save_prediction_outputs(
    best_result: TrainedModelResult,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Save predictions from the best-performing model.

    This file is useful for:
    - manual validation
    - high-risk customer analysis
    - Power BI import
    """
    prediction_df = X_test.copy()
    prediction_df["actual_churn_flag"] = y_test.values
    prediction_df["predicted_churn_flag"] = best_result.y_pred.values
    prediction_df["churn_probability"] = best_result.y_proba.round(4).values
    prediction_df["risk_segment"] = pd.cut(
        prediction_df["churn_probability"],
        bins=[-0.01, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    prediction_df.to_csv(PROCESSED_DIR / "test_predictions_best_model.csv", index=False)
    prediction_df.to_csv(POWERBI_DIR / "customer_churn_scoring.csv", index=False)
    return prediction_df


def build_powerbi_summary(df: pd.DataFrame) -> None:
    """
    Build summary tables for the Power BI dashboard.

    These pre-aggregated tables make dashboard creation faster.
    """
    summary_contract = (
        df.groupby("Contract", observed=False)
        .agg(
            total_customers=("customerID", "count"),
            churn_rate=("ChurnFlag", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_tenure=("tenure", "mean"),
        )
        .reset_index()
    )

    summary_payment = (
        df.groupby("PaymentMethod", observed=False)
        .agg(
            total_customers=("customerID", "count"),
            churn_rate=("ChurnFlag", "mean"),
            avg_total_charges=("TotalCharges", "mean"),
        )
        .reset_index()
    )

    summary_internet = (
        df.groupby("InternetService", observed=False)
        .agg(
            total_customers=("customerID", "count"),
            churn_rate=("ChurnFlag", "mean"),
        )
        .reset_index()
    )

    summary_contract["churn_rate"] = summary_contract["churn_rate"].round(4)
    summary_payment["churn_rate"] = summary_payment["churn_rate"].round(4)
    summary_internet["churn_rate"] = summary_internet["churn_rate"].round(4)

    summary_contract.to_csv(POWERBI_DIR / "summary_by_contract.csv", index=False)
    summary_payment.to_csv(POWERBI_DIR / "summary_by_payment_method.csv", index=False)
    summary_internet.to_csv(POWERBI_DIR / "summary_by_internet_service.csv", index=False)


def save_classification_reports(results: List[TrainedModelResult], y_test: pd.Series) -> None:
    """Save each model's classification report as reusable JSON."""
    for result in results:
        report = classification_report(y_test, result.y_pred, output_dict=True)
        with open(REPORT_DIR / f"classification_report_{result.name}.json", "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)


def save_business_recommendations(
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> None:
    """
    Save business recommendations based on model output and data patterns.

    The goal is to extend the project beyond modeling into business-ready
    insights.
    """
    top_contract = cleaned_df.groupby("Contract", observed=False)["ChurnFlag"].mean().sort_values(ascending=False)
    top_payment = cleaned_df.groupby("PaymentMethod", observed=False)["ChurnFlag"].mean().sort_values(ascending=False)
    top_internet = cleaned_df.groupby("InternetService", observed=False)["ChurnFlag"].mean().sort_values(ascending=False)

    best_model_name = metrics_df.iloc[0]["model"]
    top_features = importance_df.head(5)["feature"].tolist()

    recommendations = [
        f"The best model by ROC-AUC is {best_model_name}.",
        f"The contract type with the highest churn rate is {top_contract.index[0]} ({top_contract.iloc[0]:.2%}).",
        f"The payment method with the highest churn rate is {top_payment.index[0]} ({top_payment.iloc[0]:.2%}).",
        f"The internet service with the highest churn rate is {top_internet.index[0]} ({top_internet.iloc[0]:.2%}).",
        f"The most influential churn drivers in the best model are: {', '.join(top_features)}.",
        "Prioritize retention programs for month-to-month customers with high predicted churn probability.",
        "Create bundling offers or loyalty benefits for customers with relatively low support-service adoption.",
        "Review potential friction in electronic check payments because this segment shows consistently higher churn.",
    ]

    with open(REPORT_DIR / "business_recommendations.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(recommendations))


def run_project_pipeline() -> None:
    """
    Execute the full project workflow from start to finish.

    The sequence is intentionally linear so new readers can follow it easily.
    """
    ensure_directories()

    csv_path = download_dataset()
    raw_df = load_dataset(csv_path)
    cleaned_df = clean_data(raw_df)
    featured_df = add_features(cleaned_df)
    save_cleaned_dataset(featured_df)
    plot_class_distribution(featured_df)

    X_train, X_test, y_train, y_test, numeric_features, categorical_features, _ = prepare_train_test_data(featured_df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    metrics_df = save_metrics(results)
    save_model_artifacts(results)
    save_classification_reports(results, y_test)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)

    best_result = sorted(results, key=lambda item: item.metrics["roc_auc"], reverse=True)[0]
    importance_df = save_feature_importance(best_result, X_test, y_test)
    save_prediction_outputs(best_result, X_test, y_test)
    build_powerbi_summary(featured_df)
    save_business_recommendations(metrics_df, importance_df, featured_df)

    print("Project pipeline completed successfully.")
    print(f"Best model: {best_result.name}")
    print("Metrics summary:")
    print(metrics_df.to_string(index=False))
