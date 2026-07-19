# Customer Retention Intelligence

## From Churn Probability to Retention Priority

Memprediksi churn belum cukup untuk membantu tim retensi. Tim juga perlu mengetahui pelanggan
mana yang harus ditinjau lebih dahulu, seberapa mendesak risikonya, indikator apa yang terlihat
pada profilnya, dan tindakan apa yang layak diuji.

Proyek ini mengubah data pelanggan telekomunikasi menjadi **churn probability**, threshold
keputusan yang divalidasi, risk segment, retention priority, observed risk indicators, dan
dataset siap Power BI. Output tindakan adalah hipotesis yang dapat diuji, bukan klaim bahwa
intervensi tertentu sudah terbukti mencegah churn.

## The Retention Decision

Keputusan yang didukung adalah:

> Pelanggan mana yang perlu diprioritaskan untuk outreach retensi ketika kapasitas kontak terbatas?

Objektif model adalah menemukan proporsi tinggi dari pelanggan yang benar-benar churn sambil
menjaga jumlah outreach tetap dapat dikelola. Karena false negative berarti churner terlewat,
recall menjadi pertimbangan penting. Precision tetap dijaga agar biaya menghubungi pelanggan
yang sebenarnya bertahan tidak menjadi tidak terkendali.

## How a Customer Becomes a Retention Priority

```text
Customer profile
      ↓
Churn probability
      ↓
Validated decision threshold
      ↓
Low / Medium / High Risk
      ↓
Transparent priority score
      ↓
Observed risk indicators
      ↓
Suggested intervention hypothesis
      ↓
Power BI retention queue
```

`customerID` dipertahankan sebagai traceability key pada output, tetapi secara eksplisit
dikeluarkan dari model bersama `Churn` dan `ChurnFlag`.

## Retention Snapshot

Dataset sumber memiliki 7.043 pelanggan dan churn rate historis 26,54%. Evaluasi final memakai
1.409 pelanggan pada holdout test set yang tidak digunakan untuk memilih model atau threshold.

| Ukuran | Hasil terverifikasi |
|---|---:|
| Selected model | Random Forest |
| Selected threshold | 0,30 |
| Churn recall | 86,90% |
| Churn precision | 45,90% |
| Churn F1 | 60,07% |
| PR-AUC | 64,70% |
| ROC-AUC | 84,07% |
| Churners detected | 325 |
| Churners missed | 49 |
| High Risk customers | 708 |
| Priority 1 customers | 220 |

Metrik lengkap seluruh model tersedia di
[`reports/metrics/model_comparison.csv`](reports/metrics/model_comparison.csv).

## Customer Risk Portfolio

Risk segment dihubungkan langsung dengan threshold intervensi:

- **Low Risk:** probability `< 0,15`
- **Medium Risk:** probability `0,15–<0,30`
- **High Risk:** probability `≥ 0,30`

| Risk segment | Customers | Portfolio share | Average probability | Actual churn rate |
|---|---:|---:|---:|---:|
| Low Risk | 478 | 33,92% | 5,64% | 3,14% |
| Medium Risk | 223 | 15,83% | 21,52% | 15,25% |
| High Risk | 708 | 50,25% | 61,85% | 45,90% |

![Held-out customers by churn-risk segment](reports/figures/risk_segment_distribution.png)

Probability quantiles juga diuji. Keduanya menghasilkan pemisahan risiko yang monotonik, tetapi
business-linked boundaries dipilih karena definisi High Risk tetap identik dengan populasi yang
melewati threshold intervensi. Perbandingan lengkap terdapat di
[`reports/insights/risk_segmentation_method.md`](reports/insights/risk_segmentation_method.md).

## The Decision Threshold

Threshold tidak dioptimalkan pada test set. Pipeline membuat validation subset dari training
data, membandingkan threshold `0,25–0,70` untuk lima model, lalu menilai kombinasi:

- churn recall: 30%;
- churn F1: 30%;
- PR-AUC: 25%;
- churn precision: 15%.

Kandidat juga harus mencapai precision minimal 45% dan menandai maksimal 50% validation
customers. Proses ini memilih Random Forest pada threshold `0,30`. Setelah pemilihan selesai,
model dilatih ulang pada seluruh training set dan dievaluasi satu kali pada holdout test set.

![Threshold precision-recall trade-off](reports/figures/threshold_tradeoff.png)

Threshold `0,30` mengutamakan coverage churner. Konsekuensinya adalah 383 false positives:
pelanggan yang sebenarnya tidak churn tetapi masuk outreach. Sebaliknya, 49 false negatives
merupakan pelanggan churn yang tidak terdeteksi dan tidak masuk intervensi berbasis threshold.

## Retention Action Queue

Priority score tidak berasal dari model tambahan yang tidak transparan:

```text
Priority score = 60% churn probability
               + 25% customer-value proxy
               + 15% intervention urgency
```

Customer-value proxy menggabungkan `MonthlyCharges` dan `TotalCharges` yang diskalakan terhadap
persentil ke-95 training data. Nilai ini **bukan Customer Lifetime Value**. Urgency menggunakan
tenure dan status month-to-month contract.

| Priority | Rule | Customers |
|---|---|---:|
| Priority 1 | High Risk dan priority score ≥ 70 | 220 |
| Priority 2 | High Risk lainnya | 488 |
| Priority 3 | Medium Risk | 223 |
| Monitor | Low Risk | 478 |

Contoh berikut dianonimkan dari antrean yang dihasilkan:

| Customer | Churn probability | Risk | Priority score | Priority | Suggested action | Primary observed indicator |
|---|---:|---|---:|---|---|---|
| Customer A | 95,68% | High Risk | 85,76 | Priority 1 | Contract-upgrade incentive | Tenure ≤ 12 months |
| Customer B | 93,94% | High Risk | 85,74 | Priority 1 | Contract-upgrade incentive | Tenure ≤ 12 months |
| Customer C | 95,50% | High Risk | 85,60 | Priority 1 | Contract-upgrade incentive | Tenure ≤ 12 months |

Queue lengkap tersedia di
[`powerbi/customer_retention_queue.csv`](powerbi/customer_retention_queue.csv).

## What Raises the Risk Signal

Global driver analysis menggabungkan permutation importance model dengan asosiasi churn historis.
Hasil dipisahkan antara indikator yang berasosiasi dengan churn lebih tinggi dan indikator retensi.
Analisis ini tidak menyatakan hubungan sebab-akibat.

![Observed indicators associated with higher churn](reports/figures/top_churn_indicators.png)

Indikator risiko pelanggan hanya ditampilkan ketika mempunyai minimal 100 pelanggan dan observed
churn rate setidaknya lima percentage points di atas churn rate keseluruhan. Contohnya meliputi
short tenure, electronic-check payment, month-to-month contract, fiber-optic service, serta tidak
memiliki technical support atau online security.

![Observed indicators associated with lower churn](reports/figures/top_retention_indicators.png)

Karena exact local model explanation tidak digunakan, output customer-level diberi label
**Observed risk indicators**, bukan “reasons the model predicted churn.”

## Retention Actions

Action rules menghubungkan risk segment dan karakteristik pelanggan dengan hipotesis tindakan:

- contract-upgrade incentive;
- onboarding-support outreach;
- technical-support outreach;
- payment-method assistance;
- retention call;
- targeted retention email;
- monitor only.

Setiap tindakan menyertakan expected mechanism dan suggested success metric. Efektivitasnya belum
dibuktikan karena dataset tidak berisi eksperimen atau hasil intervensi. Bukti dan aturan lengkap
terdapat di [`reports/insights/retention_recommendations.md`](reports/insights/retention_recommendations.md).

## Hypothetical Strategy Simulation

Simulasi membandingkan beberapa strategi menggunakan asumsi hipotetis: outreach cost `5`, incentive
cost `20`, remaining-churn cost `300`, dan intervention success rate `25%`. Semua angka memakai
**hypothetical value units**, bukan biaya atau pendapatan perusahaan telekomunikasi nyata.

| Strategy | Contacted | Churners captured | Churners missed | False-positive interventions | Hypothetical net value |
|---|---:|---:|---:|---:|---:|
| No intervention | 0 | 0 | 374 | 0 | 0 |
| Contact all customers | 1.409 | 374 | 0 | 1.035 | -7.175 |
| Default threshold 0,50 | 498 | 272 | 102 | 226 | 7.950 |
| Selected threshold 0,30 | 708 | 325 | 49 | 383 | 6.675 |
| Priority 1 only | 220 | 149 | 225 | 71 | 5.675 |

Pada asumsi ini, threshold `0,50` memiliki hypothetical net value tertinggi, sedangkan threshold
`0,30` menangkap lebih banyak churner. Ini menunjukkan bahwa model-selection objective dan strategi
ekonomi bukan keputusan yang identik. Ranking dapat berubah bila biaya, nilai pelanggan, kapasitas,
atau intervention success rate berubah.

## Power BI Decision Layer

Pipeline menghasilkan:

- `customer_retention_queue.csv` untuk tindakan customer-level;
- `risk_segment_summary.csv` untuk portfolio risk;
- `model_performance_summary.csv` untuk model evidence;
- `churn_driver_summary.csv` untuk indikator churn dan retensi;
- ringkasan berdasarkan contract, payment method, dan internet service.

Empat halaman Power BI yang direkomendasikan:

1. Executive Retention Overview
2. Customer Risk Portfolio
3. Churn Drivers
4. Retention Action Queue

Definisi grain dan kolom tersedia di [`powerbi/DATA_DICTIONARY.md`](powerbi/DATA_DICTIONARY.md),
sedangkan desain dashboard tersedia di [`powerbi/DASHBOARD_GUIDE.md`](powerbi/DASHBOARD_GUIDE.md).
Repository ini tidak mengklaim file `.pbix` sudah tersedia.

## Where the System Can Be Wrong

- False negatives membuat pelanggan churn tidak masuk outreach berbasis threshold.
- False positives menimbulkan outreach dan insentif yang tidak diperlukan.
- Probabilitas belum dikalibrasi secara khusus; score digunakan terutama untuk ranking dan threshold.
- Risk dan priority boundaries adalah aturan keputusan yang transparan, bukan batas universal.
- Historical association dan feature importance tidak membuktikan kausalitas.
- Queue customer-level memakai holdout test set agar contoh keputusan tetap out-of-sample; summary
  historis lain dapat menggunakan seluruh dataset dan tidak boleh dijumlahkan sebagai populasi yang sama.

Error records dan konsentrasinya tersedia di
[`reports/insights/error_analysis_summary.md`](reports/insights/error_analysis_summary.md).

## Reproducing the Decision Pipeline

```bash
pip install -r requirements.txt
python run_pipeline.py
python -m pytest -q
```

Pipeline akan memvalidasi schema, membersihkan data, membuat fitur, melatih lima model, memilih
model dan threshold pada validation data, mengevaluasi holdout, lalu menghasilkan model artifacts,
insights, figures, retention queue, dan Power BI datasets.

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Pipeline menggunakan copy lokal bila tersedia dan KaggleHub sebagai fallback.

## Repository Structure

```text
├── data/                  # raw and generated processed data
├── models/                # complete pipelines and model metadata
├── notebooks/             # secondary walkthrough
├── powerbi/               # decision datasets, dictionary, and dashboard guide
├── reports/
│   ├── figures/           # evaluation and decision visuals
│   ├── insights/          # risk, priority, actions, drivers, and errors
│   └── metrics/           # model, threshold, and strategy comparisons
├── src/
│   ├── churn_pipeline.py
│   └── retention_intelligence.py
├── tests/
├── run_pipeline.py
└── requirements.txt
```

## Limitations

- Dataset adalah public historical snapshot dan tidak memiliki timestamp untuk drift monitoring.
- Tidak ada hasil kampanye atau eksperimen retensi nyata.
- Tidak ada klaim kausal mengenai churn indicators atau suggested actions.
- Customer-value score adalah proxy, bukan CLV.
- Seluruh nilai simulasi bisnis bersifat hipotetis.
- Tidak ada production deployment, automated retraining, atau data-drift monitoring.
- Probability calibration belum menjadi bagian pipeline.
- Priority score dan action rules perlu divalidasi ulang dengan kapasitas operasional dan hasil intervensi nyata.
