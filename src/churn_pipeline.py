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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
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

from src.retention_intelligence import (
    RISK_METHOD,
    assign_risk_segments,
    build_error_analysis,
    build_global_churn_drivers,
    build_retention_queue,
    compare_segmentation_methods,
    save_driver_figures,
    save_model_metadata,
    save_risk_figure,
    simulate_retention_strategies,
    summarize_risk_segments,
    validate_observed_indicators,
    write_error_summary,
    write_retention_recommendations,
    write_simulation_assumptions,
)


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
METRICS_DIR = REPORT_DIR / "metrics"
INSIGHTS_DIR = REPORT_DIR / "insights"

RANDOM_SEED = 42
TARGET_COLUMN = "ChurnFlag"
RAW_TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
EXCLUDED_MODEL_COLUMNS = [ID_COLUMN, RAW_TARGET_COLUMN, TARGET_COLUMN]
REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
MIN_ACCEPTABLE_PRECISION = 0.45


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
    threshold: float = 0.5


def ensure_directories() -> None:
    """Create the project output folders if they do not already exist."""
    for folder in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, METRICS_DIR, INSIGHTS_DIR, FIGURE_DIR, POWERBI_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def validate_required_columns(df: pd.DataFrame) -> None:
    """Fail early when the input cannot support the churn workflow."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    if df[RAW_TARGET_COLUMN].isna().any():
        raise ValueError("Churn contains missing values.")

    target_values = set(df[RAW_TARGET_COLUMN].astype(str).str.strip().unique())
    unsupported_targets = sorted(target_values - {"Yes", "No"})
    if unsupported_targets:
        raise ValueError(f"Unsupported Churn values: {unsupported_targets}")

    customer_ids = df[ID_COLUMN].astype("string").str.strip()
    if customer_ids.isna().any() or customer_ids.eq("").any():
        raise ValueError("customerID contains missing or empty values.")


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
    validate_required_columns(df)
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

    if cleaned[ID_COLUMN].duplicated().any():
        raise ValueError("customerID must be unique after duplicate-row removal.")

    if cleaned["TotalCharges"].notna().sum() == 0:
        raise ValueError("TotalCharges contains no valid numeric values.")

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
    customer_reference = feature_df[[ID_COLUMN, RAW_TARGET_COLUMN]].copy()

    y = feature_df[TARGET_COLUMN]
    X = feature_df.drop(columns=EXCLUDED_MODEL_COLUMNS)

    if X.empty or X.shape[1] == 0:
        raise ValueError("Model feature matrix is empty.")
    leaked_columns = set(EXCLUDED_MODEL_COLUMNS).intersection(X.columns)
    if leaked_columns:
        raise AssertionError(f"Excluded columns reached model features: {sorted(leaked_columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
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
        "dummy_classifier": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", DummyClassifier(strategy="prior", random_state=RANDOM_SEED)),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "balanced_logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=2,
                        random_state=RANDOM_SEED,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
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
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def metrics_at_threshold(y_true: pd.Series, y_proba: pd.Series, threshold: float) -> Tuple[pd.Series, Dict[str, float]]:
    """Calculate decision metrics using an explicit churn threshold."""
    y_pred = pd.Series((y_proba >= threshold).astype(int), index=y_true.index, name="prediction")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return y_pred, metrics


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> TrainedModelResult:
    """
    Generate predictions and evaluation metrics for one model.

    Main metrics:
    - accuracy
    - precision
    - recall
    - F1-score
    - ROC-AUC
    """
    y_proba = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=y_test.index, name="churn_probability")
    y_pred, raw_metrics = metrics_at_threshold(y_test, y_proba, threshold)
    metrics = {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in raw_metrics.items()
    }

    return TrainedModelResult(
        name=name,
        pipeline=pipeline,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
        threshold=threshold,
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


def build_threshold_analysis(
    model_name: str,
    y_true: pd.Series,
    y_proba: pd.Series,
) -> pd.DataFrame:
    """Evaluate candidate operating thresholds on validation predictions."""
    rows = []
    for threshold in THRESHOLDS:
        y_pred, metrics = metrics_at_threshold(y_true, y_proba, threshold)
        customers_flagged = int(y_pred.sum())
        flagged_percentage = customers_flagged / len(y_true)
        selection_score = (
            0.30 * metrics["recall"]
            + 0.30 * metrics["f1_score"]
            + 0.25 * metrics["pr_auc"]
            + 0.15 * metrics["precision"]
        )
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "pr_auc": metrics["pr_auc"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "customers_flagged": customers_flagged,
                "flagged_percentage": flagged_percentage,
                "selection_score": selection_score,
                "selection_eligible": (
                    metrics["precision"] >= MIN_ACCEPTABLE_PRECISION
                    and flagged_percentage <= 0.50
                ),
            }
        )
    return pd.DataFrame(rows)


def select_model_and_threshold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[str, float, pd.DataFrame]:
    """Select model and threshold using a validation subset of training data.

    The documented score prioritizes churn recall and churn F1, followed by
    PR-AUC and precision. Eligible candidates must reach 45% precision and
    flag at most 50% of validation customers.
    """
    X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )

    analyses = []
    for model_name, pipeline in build_models(preprocessor).items():
        pipeline.fit(X_subtrain, y_subtrain)
        probabilities = pd.Series(
            pipeline.predict_proba(X_validation)[:, 1],
            index=y_validation.index,
        )
        analyses.append(build_threshold_analysis(model_name, y_validation, probabilities))

    threshold_df = pd.concat(analyses, ignore_index=True)
    eligible = threshold_df[threshold_df["selection_eligible"]]
    candidates = eligible if not eligible.empty else threshold_df
    selected_row = candidates.sort_values(
        ["selection_score", "recall", "false_negatives", "precision"],
        ascending=[False, False, True, False],
    ).iloc[0]
    threshold_df["selected"] = (
        threshold_df["model"].eq(selected_row["model"])
        & threshold_df["threshold"].eq(selected_row["threshold"])
    )
    threshold_df = threshold_df.round(4)
    threshold_df.to_csv(METRICS_DIR / "threshold_analysis.csv", index=False)
    return str(selected_row["model"]), float(selected_row["threshold"]), threshold_df


def train_final_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    threshold_df: pd.DataFrame,
) -> List[TrainedModelResult]:
    """Refit every model on all training rows and evaluate on test once."""
    results = []
    for model_name, pipeline in build_models(preprocessor).items():
        model_rows = threshold_df[
            (threshold_df["model"] == model_name)
            & threshold_df["selection_eligible"]
        ]
        if model_rows.empty:
            model_rows = threshold_df[threshold_df["model"] == model_name]
        threshold = float(
            model_rows.sort_values("selection_score", ascending=False).iloc[0]["threshold"]
        )
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(model_name, pipeline, X_test, y_test, threshold))
    return results


def save_metrics(
    results: List[TrainedModelResult],
    selected_model: str | None = None,
) -> pd.DataFrame:
    """Save the cross-model metrics summary to CSV."""
    metrics_df = pd.DataFrame(
        [
            {
                "model": result.name,
                "decision_threshold": result.threshold,
                "selected_model": result.name == selected_model,
                **result.metrics,
            }
            for result in results
        ]
    ).sort_values(by=["selected_model", "pr_auc"], ascending=[False, False])

    metrics_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    metrics_df.to_csv(REPORT_DIR / "metrics_summary.csv", index=False)
    return metrics_df


def plot_threshold_analysis(selected_model: str, selected_threshold: float, threshold_df: pd.DataFrame) -> None:
    """Plot validation operating points for the selected model."""
    selected_rows = threshold_df[threshold_df["model"] == selected_model]

    plt.figure(figsize=(8, 6))
    for metric in ["precision", "recall", "f1_score"]:
        plt.plot(selected_rows["threshold"], selected_rows[metric], marker="o", label=metric)
    plt.axvline(selected_threshold, color="black", linestyle="--", label="selected threshold")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Trade-off - {selected_model}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "threshold_tradeoff.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(selected_rows["recall"], selected_rows["precision"], marker="o")
    chosen = selected_rows[selected_rows["selected"]]
    if not chosen.empty:
        plt.scatter(chosen["recall"], chosen["precision"], color="red", s=80, label="selected threshold")
        plt.legend()
    plt.xlabel("Churn Recall")
    plt.ylabel("Churn Precision")
    plt.title(f"Precision-Recall Operating Points - {selected_model}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "precision_recall_curve.png", dpi=200)
    plt.close()


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
    customer_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Save predictions from the best-performing model.

    This file is useful for:
    - manual validation
    - high-risk customer analysis
    - Power BI import
    """
    prediction_df = X_test.copy()
    if customer_reference is not None:
        prediction_df.insert(0, ID_COLUMN, customer_reference.loc[X_test.index, ID_COLUMN])
    prediction_df["actual_churn_flag"] = y_test.values
    prediction_df["predicted_churn_flag"] = (
        best_result.y_proba.values >= best_result.threshold
    ).astype(int)
    prediction_df["churn_probability"] = best_result.y_proba.values
    prediction_df["risk_segment"] = assign_risk_segments(
        prediction_df["churn_probability"], best_result.threshold
    ).values

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
        report = classification_report(y_test, result.y_pred, output_dict=True, zero_division=0)
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
        f"The selected model for the retention decision objective is {best_model_name}.",
        f"The contract type with the highest churn rate is {top_contract.index[0]} ({top_contract.iloc[0]:.2%}).",
        f"The payment method with the highest churn rate is {top_payment.index[0]} ({top_payment.iloc[0]:.2%}).",
        f"The internet service with the highest churn rate is {top_internet.index[0]} ({top_internet.iloc[0]:.2%}).",
        f"The most influential churn drivers in the best model are: {', '.join(top_features)}.",
        "Rule-matched intervention hypotheses and their evidence are documented in reports/insights/retention_recommendations.md.",
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

    X_train, X_test, y_train, y_test, numeric_features, categorical_features, customer_reference = prepare_train_test_data(featured_df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    selected_model, selected_threshold, threshold_df = select_model_and_threshold(
        X_train, y_train, preprocessor
    )
    selected_validation = threshold_df[threshold_df["selected"]].iloc[0]
    threshold_note = (
        "# Threshold Selection\n\n"
        "Decision objective: identify a high proportion of actual churners while "
        "keeping retention outreach operationally manageable.\n\n"
        "Selection data: a stratified validation subset drawn only from the training set. "
        "The held-out test set was not used for model or threshold selection.\n\n"
        "Candidate thresholds: 0.25 to 0.70 in increments of 0.05. Candidates must "
        f"reach precision >= {MIN_ACCEPTABLE_PRECISION:.2f} and flag <= 50% of validation customers.\n\n"
        "Selection score: 30% churn recall + 30% churn F1 + 25% PR-AUC + 15% churn precision.\n\n"
        f"Selected model: {selected_model}\n\n"
        f"Selected threshold: {selected_threshold:.2f}\n\n"
        f"Validation precision: {selected_validation['precision']:.4f}\n\n"
        f"Validation recall: {selected_validation['recall']:.4f}\n\n"
        f"Validation churn F1: {selected_validation['f1_score']:.4f}\n\n"
        f"Validation customers flagged: {int(selected_validation['customers_flagged'])} "
        f"({selected_validation['flagged_percentage']:.2%})\n"
    )
    (METRICS_DIR / "threshold_selection.md").write_text(threshold_note, encoding="utf-8")
    results = train_final_models(
        X_train, X_test, y_train, y_test, preprocessor, threshold_df
    )
    metrics_df = save_metrics(results, selected_model)
    save_model_artifacts(results)
    save_classification_reports(results, y_test)
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)

    best_result = next(result for result in results if result.name == selected_model)
    if best_result.threshold != selected_threshold:
        raise AssertionError("Selected threshold is inconsistent with final evaluation.")
    plot_threshold_analysis(selected_model, selected_threshold, threshold_df)
    importance_df = save_feature_importance(best_result, X_test, y_test)
    prediction_df = save_prediction_outputs(best_result, X_test, y_test, customer_reference)

    segmentation_comparison = compare_segmentation_methods(prediction_df, selected_threshold)
    segmentation_comparison.to_csv(INSIGHTS_DIR / "risk_segmentation_comparison.csv", index=False)
    linked_summary = segmentation_comparison[
        segmentation_comparison["method"].eq(RISK_METHOD)
    ].set_index("risk_segment")
    quantile_summary = segmentation_comparison[
        segmentation_comparison["method"].eq("probability_quantiles")
    ].set_index("risk_segment")
    segmentation_note = (
        "# Risk Segmentation Method\n\n"
        "## Approaches compared\n\n"
        f"- Business-linked: Low < {selected_threshold / 2:.2f}; Medium "
        f"{selected_threshold / 2:.2f} to < {selected_threshold:.2f}; High >= {selected_threshold:.2f}.\n"
        "- Probability quantiles: three approximately equal-sized score groups.\n\n"
        "## Observed holdout quality\n\n"
        f"Business-linked actual churn rates: Low {linked_summary.loc['Low Risk', 'actual_churn_rate']:.2%}, "
        f"Medium {linked_summary.loc['Medium Risk', 'actual_churn_rate']:.2%}, "
        f"High {linked_summary.loc['High Risk', 'actual_churn_rate']:.2%}.\n\n"
        f"Quantile actual churn rates: Low {quantile_summary.loc['Low Risk', 'actual_churn_rate']:.2%}, "
        f"Medium {quantile_summary.loc['Medium Risk', 'actual_churn_rate']:.2%}, "
        f"High {quantile_summary.loc['High Risk', 'actual_churn_rate']:.2%}.\n\n"
        "## Recommendation\n\n"
        "Use the business-linked method. Both approaches produce monotonic risk separation, but "
        "the business-linked boundaries keep High Risk identical to the validated intervention "
        "population. Quantiles remain a useful portfolio comparison, not the operational rule.\n"
    )
    (INSIGHTS_DIR / "risk_segmentation_method.md").write_text(segmentation_note, encoding="utf-8")

    indicator_evidence = validate_observed_indicators(featured_df)
    indicator_evidence.to_csv(INSIGHTS_DIR / "observed_risk_indicators.csv", index=False)
    value_reference = {
        "monthly_charges_p95": round(float(X_train["MonthlyCharges"].quantile(0.95)), 4),
        "total_charges_p95": round(float(X_train["TotalCharges"].quantile(0.95)), 4),
    }
    priority_note = (
        "# Retention Priority Method\n\n"
        "The priority score is a transparent decision heuristic, not another trained model.\n\n"
        "## Formula\n\n"
        "- 60% churn probability.\n"
        "- 25% customer-value proxy.\n"
        "- 15% intervention urgency.\n\n"
        "The value proxy combines 60% MonthlyCharges and 40% TotalCharges after each is "
        "scaled to its training-data 95th percentile and capped at 100%. "
        f"Reference values: MonthlyCharges={value_reference['monthly_charges_p95']:.4f}; "
        f"TotalCharges={value_reference['total_charges_p95']:.4f}. It is not Customer Lifetime Value.\n\n"
        "Urgency combines 60% inverse tenure (capped at 72 months) and 40% "
        "month-to-month contract status.\n\n"
        "## Priority mapping\n\n"
        "- Priority 1: High Risk and priority score >= 70.\n"
        "- Priority 2: other High Risk customers.\n"
        "- Priority 3: Medium Risk customers.\n"
        "- Monitor: Low Risk customers.\n\n"
        "The score-70 boundary is an explainable portfolio rule, not a proven economic optimum. "
        "It must be revisited when real capacity, customer value, and intervention outcomes exist.\n"
    )
    (INSIGHTS_DIR / "retention_priority_method.md").write_text(priority_note, encoding="utf-8")
    best_result.pipeline.retention_value_reference_ = value_reference
    retention_queue = build_retention_queue(
        prediction_df, selected_threshold, indicator_evidence, value_reference
    )
    retention_queue.to_csv(INSIGHTS_DIR / "customer_retention_queue.csv", index=False)

    risk_summary = summarize_risk_segments(retention_queue)
    risk_summary.to_csv(INSIGHTS_DIR / "risk_segment_summary.csv", index=False)
    risk_summary.to_csv(POWERBI_DIR / "risk_segment_summary.csv", index=False)
    save_risk_figure(risk_summary, FIGURE_DIR)

    drivers = build_global_churn_drivers(featured_df, importance_df)
    drivers.to_csv(INSIGHTS_DIR / "global_churn_drivers.csv", index=False)
    drivers.to_csv(POWERBI_DIR / "churn_driver_summary.csv", index=False)
    save_driver_figures(drivers, FIGURE_DIR)

    errors = build_error_analysis(retention_queue, selected_threshold)
    errors.to_csv(INSIGHTS_DIR / "error_analysis.csv", index=False)
    write_error_summary(errors, INSIGHTS_DIR / "error_analysis_summary.md")
    write_retention_recommendations(
        retention_queue,
        indicator_evidence,
        INSIGHTS_DIR / "retention_recommendations.md",
    )
    action_rules = (
        retention_queue.groupby(["suggested_action", "risk_segment", "priority_level"], observed=True)
        .agg(
            assigned_customer_count=("customerID", "size"),
            observed_evidence=("primary_reason", lambda values: values.value_counts().index[0]),
            expected_mechanism=("expected_mechanism", "first"),
            suggested_success_metric=("suggested_success_metric", "first"),
        )
        .reset_index()
        .sort_values("assigned_customer_count", ascending=False)
    )
    action_rules.to_csv(INSIGHTS_DIR / "retention_action_rules.csv", index=False)

    strategy_simulation = simulate_retention_strategies(retention_queue, selected_threshold)
    strategy_simulation.to_csv(METRICS_DIR / "retention_strategy_simulation.csv", index=False)
    write_simulation_assumptions(
        strategy_simulation,
        METRICS_DIR / "retention_strategy_simulation_assumptions.md",
    )

    queue_columns = [
        "customerID", "churn_probability", "predicted_churn_flag", "risk_segment",
        "retention_priority_score", "priority_level", "customer_value_proxy",
        "intervention_urgency", "suggested_action", "primary_reason", "secondary_reason",
        "reason_type", "expected_mechanism", "suggested_success_metric", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges", "InternetService", "PaymentMethod",
        "TechSupport", "OnlineSecurity", "SupportServiceCount",
    ]
    retention_queue[queue_columns].to_csv(POWERBI_DIR / "customer_retention_queue.csv", index=False)
    metrics_df.to_csv(POWERBI_DIR / "model_performance_summary.csv", index=False)

    joblib.dump(best_result.pipeline, MODEL_DIR / "best_churn_pipeline.joblib")
    save_model_metadata(
        MODEL_DIR / "model_metadata.json",
        best_result.name,
        selected_threshold,
        X_train.columns.tolist(),
        best_result.metrics,
        RANDOM_SEED,
        value_reference,
    )
    build_powerbi_summary(featured_df)
    save_business_recommendations(metrics_df, importance_df, featured_df)

    print("Project pipeline completed successfully.")
    print(f"Selected model: {best_result.name}")
    print(f"Selected threshold: {best_result.threshold:.2f}")
    print("Metrics summary:")
    print(metrics_df.to_string(index=False))
