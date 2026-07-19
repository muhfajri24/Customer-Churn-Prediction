# %%
"""
Notebook-style walkthrough for the Customer Churn Prediction project.

This file uses the `# %%` format so it can be executed like Google Colab
inside VS Code:
- one cell at a time,
- tables displayed inline,
- charts displayed inline,
- and the full analysis can be followed from start to finish.

How to use it in VS Code:
1. Open this file.
2. Click "Run Cell" on each `# %%` block.
3. If prompted, select the same Python environment used for this project.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


# Add the project root to the Python path so imports from `src` work correctly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.churn_pipeline import (  # noqa: E402
    add_features,
    build_models,
    build_powerbi_summary,
    build_preprocessor,
    clean_data,
    download_dataset,
    ensure_directories,
    evaluate_model,
    load_dataset,
    prepare_train_test_data,
    save_business_recommendations,
    save_feature_importance,
    save_metrics,
    save_model_artifacts,
    save_prediction_outputs,
    select_model_and_threshold,
    train_final_models,
)


sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def show_section(title: str, subtitle: str = "") -> None:
    """Display a section heading to make the notebook easier to follow."""
    text = f"## {title}"
    if subtitle:
        text += f"\n\n{subtitle}"
    display(Markdown(text))


def show_missing_value_chart(df: pd.DataFrame, title: str) -> None:
    """Display a bar chart of missing values by column."""
    missing_df = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
    )

    if missing_df.empty:
        print("No missing values were detected.")
        return

    plt.figure(figsize=(10, 5))
    sns.barplot(data=missing_df, x="column", y="missing_count", hue="column", palette="flare", legend=False)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def show_numeric_distribution(df: pd.DataFrame, columns: list[str], title_prefix: str) -> None:
    """Display histograms for selected numeric features."""
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]

    for axis, column in zip(axes, columns):
        sns.histplot(df[column], kde=True, bins=30, ax=axis, color="#1f77b4")
        axis.set_title(f"{title_prefix}: {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Count")

    plt.tight_layout()
    plt.show()


def show_churn_comparison(df: pd.DataFrame, category_column: str, title: str) -> None:
    """Display churn rate for a categorical feature."""
    summary = (
        df.groupby(category_column, observed=False)["ChurnFlag"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"ChurnFlag": "churn_rate"})
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary, x=category_column, y="churn_rate", hue=category_column, palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel(category_column)
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    display(summary)


def summarize_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> None:
    """Display a quick preview of the transformed feature matrix."""
    transformed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    preview_df = pd.DataFrame(
        transformed[:5, :10].toarray() if hasattr(transformed, "toarray") else transformed[:5, :10],
        columns=feature_names[:10],
    )

    print("Shape after preprocessing:", transformed.shape)
    print("Preview of the first 10 transformed features:")
    display(preview_df.round(3))


# %%
show_section("1. Dataset Preparation", "Load the dataset and display the initial preview.")
ensure_directories()
csv_path = download_dataset()
raw_df = load_dataset(csv_path)

print("Dataset path:", csv_path)
print("Raw dataset shape:", raw_df.shape)
display(raw_df.head())


# %%
show_section("2. Dataset Overview", "Inspect data types, column counts, and basic dataset structure.")
dataset_overview = pd.DataFrame(
    {
        "column": raw_df.columns,
        "dtype": raw_df.dtypes.astype(str).values,
        "non_null_count": raw_df.notna().sum().values,
        "unique_count": raw_df.nunique().values,
    }
)
display(dataset_overview)
raw_df.info()


# %%
show_section("3. Missing Values and Duplicates", "Identify data quality issues before cleaning.")
missing_before = raw_df.isna().sum().sort_values(ascending=False)
duplicate_before = raw_df.duplicated().sum()

print("Duplicate rows before cleaning:", duplicate_before)
display(missing_before[missing_before > 0].to_frame(name="missing_count"))
show_missing_value_chart(raw_df, "Missing Values Before Cleaning")


# %%
show_section("4. Initial Numeric Feature Distributions", "Review the distribution of key numeric features before cleaning.")
raw_numeric = raw_df.copy()
raw_numeric["TotalCharges"] = pd.to_numeric(raw_numeric["TotalCharges"], errors="coerce")
show_numeric_distribution(raw_numeric, ["tenure", "MonthlyCharges", "TotalCharges"], "Initial Distribution")


# %%
show_section("5. Data Cleaning", "Run the cleaning function and display the result.")
cleaned_df = clean_data(raw_df)

print("Shape before cleaning :", raw_df.shape)
print("Shape after cleaning  :", cleaned_df.shape)
display(cleaned_df.head())


# %%
show_section("6. Before vs After Cleaning", "Re-check missing values and numeric summaries after cleaning.")
missing_after = cleaned_df.isna().sum().sort_values(ascending=False)
display(missing_after[missing_after > 0].to_frame(name="missing_count"))
show_missing_value_chart(cleaned_df, "Missing Values After Cleaning")
display(cleaned_df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))


# %%
show_section("7. Target Distribution", "Visualize churn vs non-churn customers.")
plt.figure(figsize=(7, 5))
sns.countplot(data=cleaned_df, x="Churn", hue="Churn", palette="Set2", legend=False)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

churn_ratio = cleaned_df["Churn"].value_counts(normalize=True).mul(100).round(2)
display(churn_ratio.to_frame(name="percentage"))


# %%
show_section("8. Numeric EDA by Churn", "Compare tenure and monthly charges across churn classes.")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=cleaned_df, x="Churn", y="tenure", hue="Churn", ax=axes[0], legend=False, palette="Set2")
axes[0].set_title("Tenure vs Churn")

sns.boxplot(data=cleaned_df, x="Churn", y="MonthlyCharges", hue="Churn", ax=axes[1], legend=False, palette="Set2")
axes[1].set_title("Monthly Charges vs Churn")

plt.tight_layout()
plt.show()


# %%
show_section("9. Categorical EDA", "Compare churn rate across key business segments.")
cleaned_with_flag = cleaned_df.copy()
cleaned_with_flag["ChurnFlag"] = cleaned_with_flag["Churn"].map({"No": 0, "Yes": 1})
show_churn_comparison(cleaned_with_flag, "Contract", "Churn Rate by Contract")
show_churn_comparison(cleaned_with_flag, "InternetService", "Churn Rate by Internet Service")
show_churn_comparison(cleaned_with_flag, "PaymentMethod", "Churn Rate by Payment Method")


# %%
show_section("10. Feature Engineering", "Create additional features to enrich the model input.")
featured_df = add_features(cleaned_df)

print("Shape after feature engineering:", featured_df.shape)
display(featured_df.head())


# %%
show_section("11. Newly Created Features", "Preview the engineered columns added for modeling.")
display(
    featured_df[
        [
            "tenure",
            "TenureGroup",
            "MonthlyCharges",
            "TotalCharges",
            "AvgMonthlySpend",
            "SupportServiceCount",
            "ChurnFlag",
        ]
    ].head(10)
)


# %%
show_section("12. Engineered Feature Visualizations", "Check how the engineered features are distributed.")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=featured_df, x="TenureGroup", hue="TenureGroup", ax=axes[0], palette="crest", legend=False)
axes[0].set_title("Tenure Group Distribution")
axes[0].tick_params(axis="x", rotation=20)

sns.countplot(data=featured_df, x="SupportServiceCount", hue="SupportServiceCount", ax=axes[1], palette="mako", legend=False)
axes[1].set_title("Support Service Count Distribution")

plt.tight_layout()
plt.show()


# %%
show_section("13. Numeric Correlation Heatmap", "Review relationships among the main numeric features.")
numeric_corr = featured_df[["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "SupportServiceCount", "ChurnFlag"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# %%
show_section("14. Train-Test Split", "Split the dataset and display the resulting shapes.")
X_train, X_test, y_train, y_test, numeric_features, categorical_features, customer_reference = prepare_train_test_data(featured_df)

split_summary = pd.DataFrame(
    {
        "dataset": ["X_train", "X_test", "y_train", "y_test"],
        "rows": [X_train.shape[0], X_test.shape[0], y_train.shape[0], y_test.shape[0]],
        "columns": [X_train.shape[1], X_test.shape[1], 1, 1],
    }
)
display(split_summary)

target_distribution = pd.DataFrame(
    {
        "train_churn_rate": [y_train.mean()],
        "test_churn_rate": [y_test.mean()],
    }
).mul(100).round(2)
display(target_distribution)


# %%
show_section("15. Preprocessing Setup", "Display numeric and categorical feature groups plus transformed output preview.")
print("Number of numeric features    :", len(numeric_features))
print("Number of categorical features:", len(categorical_features))
print("Sample numeric features       :", numeric_features[:5])
print("Sample categorical features   :", categorical_features[:5])

preprocessor = build_preprocessor(numeric_features, categorical_features)
summarize_preprocessor(preprocessor, X_train)


# %%
show_section("16. Model Setup", "Compare a baseline and four classification models.")
models = build_models(preprocessor)
model_summary = pd.DataFrame(
    {
        "model_name": list(models.keys()),
        "description": [
            "No-skill baseline using the training class prior",
            "Unweighted linear model",
            "Class-balanced linear model",
            "Class-balanced tree ensemble",
            "Gradient-boosted tree model",
        ],
    }
)
display(model_summary)


# %%
show_section("17. Model Training", "Train each model and display its immediate evaluation metrics.")
selected_model, selected_threshold, threshold_df = select_model_and_threshold(X_train, y_train, preprocessor)
results = train_final_models(X_train, X_test, y_train, y_test, preprocessor, threshold_df)
for result in results:
    display(pd.DataFrame([result.metrics], index=[result.name]))


# %%
show_section("18. Metrics Summary", "Compare accuracy, precision, recall, F1-score, and ROC-AUC.")
metrics_df = save_metrics(results, selected_model)
display(metrics_df)


# %%
show_section("19. Model Comparison Chart", "Use a grouped bar chart to compare model performance across all metrics.")
metrics_long = metrics_df.melt(
    id_vars="model",
    value_vars=["precision", "recall", "f1_score", "roc_auc", "pr_auc"],
    var_name="metric",
    value_name="score",
)
plt.figure(figsize=(11, 6))
sns.barplot(data=metrics_long, x="metric", y="score", hue="model", palette="viridis")
plt.title("Model Metrics Comparison")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Model")
plt.tight_layout()
plt.show()


# %%
show_section("20. ROC Curve", "Compare how well the models separate churn and non-churn customers.")
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
plt.show()


# %%
show_section("21. Confusion Matrices", "Inspect the classification outcomes of each model.")
for result in results:
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, result.y_pred, cmap="Blues", ax=axis, colorbar=False)
    axis.set_title(f"Confusion Matrix - {result.name}")
    plt.tight_layout()
    plt.show()


# %%
show_section("22. Model and Threshold Selection", "Selection uses training-validation evidence and a retention-oriented score.")
best_result = next(result for result in results if result.name == selected_model)
print("Selected model:", best_result.name)
print("Selected threshold:", best_result.threshold)
display(pd.DataFrame([best_result.metrics], index=[best_result.name]))


# %%
show_section("23. Prediction and Scoring", "Save the trained models and preview customer scoring results.")
save_model_artifacts(results)
prediction_df = save_prediction_outputs(best_result, X_test, y_test, customer_reference)
display(prediction_df.head(10))


# %%
show_section("24. Probability and Risk Segment Distribution", "Visualize churn probabilities and business risk segments.")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(prediction_df["churn_probability"], bins=20, kde=True, ax=axes[0], color="#d62728")
axes[0].set_title("Churn Probability Distribution")
axes[0].set_xlabel("Churn Probability")

sns.countplot(data=prediction_df, x="risk_segment", hue="risk_segment", ax=axes[1], palette="coolwarm", legend=False)
axes[1].set_title("Risk Segment Distribution")
axes[1].set_xlabel("Risk Segment")
axes[1].set_ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# %%
show_section("25. Highest-Risk Customers", "Use this table to identify customers who should be prioritized for retention.")
high_risk_customers = prediction_df.sort_values("churn_probability", ascending=False).head(20)
display(
    high_risk_customers[
        [
            "customerID",
            "Contract",
            "InternetService",
            "PaymentMethod",
            "tenure",
            "MonthlyCharges",
            "churn_probability",
            "risk_segment",
        ]
    ]
)


# %%
show_section("26. Feature Importance", "Identify which features matter most for churn prediction.")
importance_df = save_feature_importance(best_result, X_test, y_test)
display(importance_df.head(10))


# %%
show_section("27. Feature Importance Chart", "Display the top 10 most influential features from the best model.")
top_importance = importance_df.head(10).sort_values("importance_mean", ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(top_importance["feature"], top_importance["importance_mean"], color="#1f77b4")
plt.title("Top 10 Feature Importance - Best Model")
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# %%
show_section("28. Power BI Output", "Prepare the summary files used to build the dashboard.")
build_powerbi_summary(featured_df)
powerbi_files = pd.DataFrame(
    {
        "file_name": [
            "customer_churn_scoring.csv",
            "summary_by_contract.csv",
            "summary_by_payment_method.csv",
            "summary_by_internet_service.csv",
        ],
        "purpose": [
            "Customer-level test set scoring output",
            "Contract-level churn summary",
            "Payment-method churn summary",
            "Internet-service churn summary",
        ],
    }
)
display(powerbi_files)


# %%
show_section("29. Business Recommendations", "Translate model outputs into actionable business suggestions.")
save_business_recommendations(metrics_df, importance_df, featured_df)
recommendation_file = PROJECT_ROOT / "reports" / "business_recommendations.txt"
print(recommendation_file.read_text(encoding="utf-8"))


# %%
show_section("30. Final Summary", "Close the walkthrough by showing the final model comparison.")
print("Analysis completed.")
print("Selected model:", best_result.name)
display(metrics_df)
