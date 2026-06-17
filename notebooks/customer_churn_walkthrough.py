# %%
"""
Notebook-style walkthrough untuk Customer Churn Prediction.

File ini dibuat dengan format `# %%` agar bisa dijalankan seperti Google Colab
di VS Code:
- per cell,
- hasil tabel langsung tampil,
- grafik tampil inline,
- dan alur analisis bisa diikuti dari awal sampai akhir.

Cara pakai di VS Code:
1. Buka file ini.
2. Klik "Run Cell" pada setiap blok `# %%`.
3. Jika diminta, pilih environment Python yang sama dengan project ini.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve


# Menambahkan root project ke path Python agar import dari folder `src` berjalan.
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
)


sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def show_section(title: str, subtitle: str = "") -> None:
    """Menampilkan judul bagian agar notebook lebih enak dibaca."""
    text = f"## {title}"
    if subtitle:
        text += f"\n\n{subtitle}"
    display(Markdown(text))


def show_missing_value_chart(df: pd.DataFrame, title: str) -> None:
    """Menampilkan bar chart missing value per kolom."""
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
        print("Tidak ada missing value yang terdeteksi.")
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
    """Menampilkan histogram untuk beberapa fitur numerik utama."""
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
    """Menampilkan churn rate berdasarkan satu kolom kategorikal."""
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
    """Menampilkan ringkasan hasil preprocessing."""
    transformed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    preview_df = pd.DataFrame(
        transformed[:5, :10].toarray() if hasattr(transformed, "toarray") else transformed[:5, :10],
        columns=feature_names[:10],
    )

    print("Bentuk data setelah preprocessing:", transformed.shape)
    print("Contoh 10 fitur hasil transformasi:")
    display(preview_df.round(3))


# %%
show_section("1. Persiapan Dataset", "Load dataset dan tampilkan preview awal.")
ensure_directories()
csv_path = download_dataset()
raw_df = load_dataset(csv_path)

print("Path dataset:", csv_path)
print("Ukuran data mentah:", raw_df.shape)
display(raw_df.head())


# %%
show_section("2. Overview Dataset", "Lihat tipe data, jumlah kolom, dan sampel isi dataset.")
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
show_section("3. Missing Value dan Duplikasi", "Tahap ini membantu kita melihat masalah data sebelum cleaning.")
missing_before = raw_df.isna().sum().sort_values(ascending=False)
duplicate_before = raw_df.duplicated().sum()

print("Jumlah duplikasi sebelum cleaning:", duplicate_before)
display(missing_before[missing_before > 0].to_frame(name="missing_count"))
show_missing_value_chart(raw_df, "Missing Value Sebelum Cleaning")


# %%
show_section("4. Distribusi Fitur Numerik Awal", "Melihat sebaran fitur numerik penting sebelum cleaning.")
raw_numeric = raw_df.copy()
raw_numeric["TotalCharges"] = pd.to_numeric(raw_numeric["TotalCharges"], errors="coerce")
show_numeric_distribution(raw_numeric, ["tenure", "MonthlyCharges", "TotalCharges"], "Distribusi Awal")


# %%
show_section("5. Data Cleaning", "Jalankan fungsi cleaning lalu tampilkan hasilnya.")
cleaned_df = clean_data(raw_df)

print("Ukuran data sebelum cleaning :", raw_df.shape)
print("Ukuran data setelah cleaning:", cleaned_df.shape)
display(cleaned_df.head())


# %%
show_section("6. Perbandingan Sebelum dan Sesudah Cleaning", "Cek missing value dan statistik numerik setelah cleaning.")
missing_after = cleaned_df.isna().sum().sort_values(ascending=False)
display(missing_after[missing_after > 0].to_frame(name="missing_count"))
show_missing_value_chart(cleaned_df, "Missing Value Setelah Cleaning")
display(cleaned_df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2))


# %%
show_section("7. Visual Target Churn", "Lihat proporsi customer churn dan non-churn.")
plt.figure(figsize=(7, 5))
sns.countplot(data=cleaned_df, x="Churn", hue="Churn", palette="Set2", legend=False)
plt.title("Distribusi Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Jumlah Customer")
plt.tight_layout()
plt.show()

churn_ratio = cleaned_df["Churn"].value_counts(normalize=True).mul(100).round(2)
display(churn_ratio.to_frame(name="percentage"))


# %%
show_section("8. EDA Numerik terhadap Churn", "Bandingkan distribusi tenure dan biaya pada customer churn vs non-churn.")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=cleaned_df, x="Churn", y="tenure", hue="Churn", ax=axes[0], legend=False, palette="Set2")
axes[0].set_title("Tenure vs Churn")

sns.boxplot(data=cleaned_df, x="Churn", y="MonthlyCharges", hue="Churn", ax=axes[1], legend=False, palette="Set2")
axes[1].set_title("Monthly Charges vs Churn")

plt.tight_layout()
plt.show()


# %%
show_section("9. EDA Kategorikal", "Bandingkan churn rate untuk beberapa segmen bisnis utama.")
cleaned_with_flag = cleaned_df.copy()
cleaned_with_flag["ChurnFlag"] = cleaned_with_flag["Churn"].map({"No": 0, "Yes": 1})
show_churn_comparison(cleaned_with_flag, "Contract", "Churn Rate berdasarkan Contract")
show_churn_comparison(cleaned_with_flag, "InternetService", "Churn Rate berdasarkan Internet Service")
show_churn_comparison(cleaned_with_flag, "PaymentMethod", "Churn Rate berdasarkan Payment Method")


# %%
show_section("10. Feature Engineering", "Tambahkan fitur turunan agar model punya informasi yang lebih kaya.")
featured_df = add_features(cleaned_df)

print("Ukuran data setelah feature engineering:", featured_df.shape)
display(featured_df.head())


# %%
show_section("11. Fitur Baru yang Dihasilkan", "Preview kolom turunan yang dibuat pada tahap feature engineering.")
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
show_section("12. Visualisasi Fitur Baru", "Lihat bagaimana fitur baru terdistribusi.")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=featured_df, x="TenureGroup", hue="TenureGroup", ax=axes[0], palette="crest", legend=False)
axes[0].set_title("Distribusi Tenure Group")
axes[0].tick_params(axis="x", rotation=20)

sns.countplot(data=featured_df, x="SupportServiceCount", hue="SupportServiceCount", ax=axes[1], palette="mako", legend=False)
axes[1].set_title("Distribusi Support Service Count")

plt.tight_layout()
plt.show()


# %%
show_section("13. Korelasi Fitur Numerik", "Heatmap ini membantu melihat hubungan antar fitur numerik utama.")
numeric_corr = featured_df[["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "SupportServiceCount", "ChurnFlag"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# %%
show_section("14. Train-Test Split", "Pisahkan data train dan test lalu tampilkan komposisinya.")
X_train, X_test, y_train, y_test, numeric_features, categorical_features, _ = prepare_train_test_data(featured_df)

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
show_section("15. Preprocessing Setup", "Tampilkan fitur numerik, kategorikal, dan contoh hasil transformasi.")
print("Jumlah fitur numerik   :", len(numeric_features))
print("Jumlah fitur kategorikal:", len(categorical_features))
print("Contoh fitur numerik   :", numeric_features[:5])
print("Contoh fitur kategorikal:", categorical_features[:5])

preprocessor = build_preprocessor(numeric_features, categorical_features)
summarize_preprocessor(preprocessor, X_train)


# %%
show_section("16. Model Setup", "Project ini melatih tiga model klasifikasi.")
models = build_models(preprocessor)
model_summary = pd.DataFrame(
    {
        "model_name": list(models.keys()),
        "description": [
            "Baseline linear model yang mudah dijelaskan",
            "Model ensemble berbasis banyak decision tree",
            "Boosting model yang kuat untuk data tabular",
        ],
    }
)
display(model_summary)


# %%
show_section("17. Training Model", "Latih model satu per satu dan tampilkan hasil metrik awalnya.")
results = []

for model_name, model_pipeline in models.items():
    print(f"Melatih model: {model_name}")
    model_pipeline.fit(X_train, y_train)
    result = evaluate_model(model_name, model_pipeline, X_test, y_test)
    results.append(result)
    display(pd.DataFrame([result.metrics], index=[model_name]))


# %%
show_section("18. Ringkasan Metrik", "Bandingkan accuracy, precision, recall, F1-score, dan ROC-AUC.")
metrics_df = save_metrics(results)
display(metrics_df)


# %%
show_section("19. Visualisasi Perbandingan Model", "Grafik bar memudahkan membaca kekuatan masing-masing model.")
metrics_long = metrics_df.melt(id_vars="model", var_name="metric", value_name="score")
plt.figure(figsize=(11, 6))
sns.barplot(data=metrics_long, x="metric", y="score", hue="model", palette="viridis")
plt.title("Perbandingan Metrik Seluruh Model")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Model")
plt.tight_layout()
plt.show()


# %%
show_section("20. ROC Curve", "Kurva ROC membantu membandingkan kemampuan model memisahkan kelas churn dan non-churn.")
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
show_section("21. Confusion Matrix", "Confusion matrix menunjukkan salah dan benar prediksi tiap model.")
for result in results:
    fig, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, result.y_pred, cmap="Blues", ax=axis, colorbar=False)
    axis.set_title(f"Confusion Matrix - {result.name}")
    plt.tight_layout()
    plt.show()


# %%
show_section("22. Menentukan Best Model", "Best model saat ini dipilih berdasarkan ROC-AUC tertinggi.")
best_result = sorted(results, key=lambda item: item.metrics["roc_auc"], reverse=True)[0]
print("Model terbaik berdasarkan ROC-AUC:", best_result.name)
display(pd.DataFrame([best_result.metrics], index=[best_result.name]))


# %%
show_section("23. Prediksi dan Scoring", "Simpan model lalu tampilkan hasil scoring customer pada data test.")
save_model_artifacts(results)
prediction_df = save_prediction_outputs(best_result, X_test, y_test)
display(prediction_df.head(10))


# %%
show_section("24. Distribusi Probabilitas dan Risk Segment", "Visual ini membantu membaca hasil scoring churn secara bisnis.")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(prediction_df["churn_probability"], bins=20, kde=True, ax=axes[0], color="#d62728")
axes[0].set_title("Distribusi Churn Probability")
axes[0].set_xlabel("Churn Probability")

sns.countplot(data=prediction_df, x="risk_segment", hue="risk_segment", ax=axes[1], palette="coolwarm", legend=False)
axes[1].set_title("Distribusi Risk Segment")
axes[1].set_xlabel("Risk Segment")
axes[1].set_ylabel("Jumlah Customer")

plt.tight_layout()
plt.show()


# %%
show_section("25. Customer Risiko Tertinggi", "Tabel ini cocok untuk bahan rekomendasi retensi.")
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
show_section("26. Feature Importance", "Cari fitur yang paling memengaruhi prediksi churn.")
importance_df = save_feature_importance(best_result, X_test, y_test)
display(importance_df.head(10))


# %%
show_section("27. Visualisasi Feature Importance", "Top 10 feature importance dari model terbaik.")
top_importance = importance_df.head(10).sort_values("importance_mean", ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(top_importance["feature"], top_importance["importance_mean"], color="#1f77b4")
plt.title("Top 10 Feature Importance - Best Model")
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# %%
show_section("28. Output untuk Power BI", "Siapkan file summary agar dashboard bisa langsung dibuat.")
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
            "Hasil scoring customer test set",
            "Ringkasan churn per kontrak",
            "Ringkasan churn per metode pembayaran",
            "Ringkasan churn per layanan internet",
        ],
    }
)
display(powerbi_files)


# %%
show_section("29. Rekomendasi Bisnis", "Hasil modeling diterjemahkan menjadi rekomendasi yang bisa dipakai tim bisnis.")
save_business_recommendations(metrics_df, importance_df, featured_df)
recommendation_file = PROJECT_ROOT / "reports" / "business_recommendations.txt"
print(recommendation_file.read_text(encoding="utf-8"))


# %%
show_section("30. Kesimpulan", "Tahap akhir: tampilkan ringkasan model dan simpulkan hasil analisis.")
print("Analisis selesai.")
print("Model terbaik berdasarkan ROC-AUC:", best_result.name)
display(metrics_df)
