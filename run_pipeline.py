"""
Entry point project Customer Churn Prediction.

File ini dibuat sederhana agar proses eksekusi mudah dipahami:
1. Download / salin dataset ke folder project.
2. Jalankan data cleaning dan feature engineering.
3. Latih beberapa model klasifikasi.
4. Evaluasi performa model.
5. Simpan artefak untuk analisis lanjutan dan Power BI.
"""

from src.churn_pipeline import run_project_pipeline


if __name__ == "__main__":
    run_project_pipeline()
