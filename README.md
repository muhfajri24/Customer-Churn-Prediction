# Customer Churn Prediction

This project aims to predict customers who are likely to stop using the service (churn), and then translate the model results into business insights that can support customer retention strategies.

## Repository Summary

- End-to-end churn prediction workflow for a telecom customer dataset
- Three classification models: Logistic Regression, Random Forest, and XGBoost
- Evaluation outputs, saved models, business recommendations, and Power BI-ready files
- Notebook-style walkthrough for step-by-step visuals similar to Google Colab

## Project Objectives

- Perform data cleaning on the Telco Customer Churn dataset.
- Create feature engineering relevant to churn prediction.
- Train 3 classification models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Evaluate model performance using classification metrics.
- Prepare output files for a Power BI dashboard.
- Provide business recommendations based on the analysis results.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- Power BI

## Dataset

The dataset comes from Kaggle:

`blastchar/telco-customer-churn`

Inside the pipeline, the dataset is downloaded automatically using `kagglehub`, then copied into:

`data/raw/telco_customer_churn.csv`

## Project Structure

```text
Customer Churn Prediction/
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|-- notebooks/
|-- powerbi/
|-- reports/
|   |-- figures/
|-- src/
|   |-- churn_pipeline.py
|-- run_pipeline.py
|-- requirements.txt
|-- README.md
```

## Project Workflow

### 1. Data Collection

The pipeline downloads the dataset using `kagglehub`.

### 2. Data Cleaning

The cleaning steps include:

- Converting `TotalCharges` into numeric format.
- Removing unnecessary spaces from categorical columns.
- Replacing empty strings with missing values.
- Removing duplicate rows if any exist.

### 3. Feature Engineering

Additional features created in this project:

- `TenureGroup`: customer subscription length segmentation.
- `AvgMonthlySpend`: estimated average spending per month.
- `SupportServiceCount`: number of support/protection services used.
- `ChurnFlag`: numeric target label (`Yes=1`, `No=0`).

### 4. Preprocessing

Preprocessing is handled using `ColumnTransformer`:

- Numeric features:
  - median imputation
  - standard scaling
- Categorical features:
  - most frequent imputation
  - one-hot encoding

### 5. Modeling

The models trained in this project:

#### Logistic Regression

Used as a baseline model because it is:

- interpretable,
- fast to train,
- easy to explain.

#### Random Forest

Used because it is:

- strong for non-linear relationships,
- relatively stable,
- able to capture feature interactions.

#### XGBoost

Used because it is:

- often strong on tabular datasets,
- effective at handling complex patterns,
- widely used in practical machine learning projects.

### 6. Evaluation Metrics

The evaluation metrics used are:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Evaluation files are saved in:

- `reports/metrics_summary.csv`
- `reports/classification_report_*.json`

### 7. Visualizations

Generated visual outputs include:

- Churn distribution
- ROC curve comparison
- Confusion matrix for each model

All visual outputs are saved in:

`reports/figures/`

### 8. Power BI Output

Files prepared for Power BI:

- `powerbi/customer_churn_scoring.csv`
- `powerbi/summary_by_contract.csv`
- `powerbi/summary_by_payment_method.csv`
- `powerbi/summary_by_internet_service.csv`

See also:

- `powerbi/README.md`

## Current Best Result

Based on the latest run saved in `reports/metrics_summary.csv`, the best model is currently:

- `Logistic Regression` by `ROC-AUC`

Latest comparison:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7622 | 0.5371 | 0.7540 | 0.6274 | 0.8415 |
| XGBoost | 0.7991 | 0.6532 | 0.5187 | 0.5782 | 0.8408 |
| Random Forest | 0.7147 | 0.4778 | 0.8048 | 0.5996 | 0.8221 |

Why Logistic Regression is selected as the best model:

- It has the highest `ROC-AUC`, which is the primary selection metric in this project.
- It also has the strongest `F1-Score`, giving the best balance between precision and recall.
- `XGBoost` has higher accuracy, and `Random Forest` has higher recall, but neither beats Logistic Regression on overall ranking performance.

## How to Run the Project

### 1. Go to the project folder

```powershell
cd "d:\code\project portfolio\Customer Churn Prediction"
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the main pipeline

```powershell
python run_pipeline.py
```

### 4. Run the notebook-style walkthrough

```powershell
python notebooks/customer_churn_walkthrough.py
```

For the intended notebook experience, open the file in VS Code and run it cell by cell with the Jupyter extension.

## Google Colab-like Display Mode

If you want to see the process from beginning to end while displaying tables and charts step by step, use:

- `notebooks/customer_churn_walkthrough.py`
- `notebooks/customer_churn_walkthrough.ipynb`

The `.py` file uses the `# %%` format, so it can be executed like a notebook in VS Code. The `.ipynb` file is included for direct upload to Google Colab.

### How to run the notebook-style mode in VS Code

1. Open `notebooks/customer_churn_walkthrough.py`
2. Make sure the Python and Jupyter extensions are enabled in VS Code
3. Click `Run Cell` on each block
4. Tables, cleaning results, and charts will be displayed inline similar to Google Colab

### How to use it in Google Colab

1. Upload `notebooks/customer_churn_walkthrough.ipynb`
2. Upload or mount the project folder so `src/` is available
3. Adjust the `PROJECT_ROOT` cell if needed
4. Run the notebook from top to bottom

### What will be displayed in this mode

- raw dataset preview
- column information and missing values
- missing value chart
- initial numeric feature histograms
- cleaned data preview
- before vs after cleaning comparison
- feature engineering results
- new feature visualizations
- boxplots of churn against tenure and monthly charges
- correlation heatmap for numeric features
- churn rate by contract, internet service, and payment method
- preprocessing output preview
- churn distribution chart
- model training results
- metrics comparison table
- full metrics comparison chart
- confusion matrices
- ROC curve
- churn probability distribution
- risk segment distribution
- top high-risk customers
- top feature importance
- business recommendation summary

If you want the closest experience to Google Colab, run the notebook-style file cell by cell instead of using `python ...` in the terminal.

## Expected Outputs

After the pipeline finishes, the project folder will contain:

- Raw and cleaned datasets
- Saved models in `.joblib` format
- Model performance summary
- Classification reports
- Model evaluation visualizations
- High-risk customer predictions
- Power BI-ready datasets
- Business recommendation file

## Code File Explanation

### `run_pipeline.py`

This is the project entry point. It is intentionally kept very short so it is easy to understand: it only calls the main pipeline from `src/churn_pipeline.py`.

### `src/churn_pipeline.py`

This file contains the full project logic, separated into small functions:

- `ensure_directories()`  
  Creates the output folder structure.

- `download_dataset()`  
  Downloads the dataset and copies it into the project folder.

- `clean_data()`  
  Cleans the raw dataset.

- `add_features()`  
  Creates engineered features for modeling.

- `prepare_train_test_data()`  
  Splits features, target, and train-test data.

- `build_preprocessor()`  
  Builds numeric and categorical preprocessing transformations.

- `build_models()`  
  Creates the 3 model pipelines.

- `train_and_evaluate_models()`  
  Trains and evaluates all models.

- `save_metrics()`  
  Saves the model metrics comparison.

- `save_prediction_outputs()`  
  Saves prediction results from the best model.

- `build_powerbi_summary()`  
  Creates summary tables for the Power BI dashboard.

- `save_business_recommendations()`  
  Writes business recommendations based on the analysis results.

## Business Recommendation

In general, this project is designed to help businesses:

- identify customers with high churn risk,
- prioritize retention interventions,
- understand which customer segments are most vulnerable to churn,
- and prepare a recurring churn monitoring dashboard.

Retention efforts can usually focus on:

- customers with `Month-to-month` contracts,
- customers with high churn probability,
- customers with low support service adoption,
- and customers with payment behavior associated with higher churn.

## Important Notes

- Power BI Desktop is not generated automatically from Python, but all dashboard source files are already prepared in the `powerbi/` folder.
- The code contains explanatory comments in important sections so it is easier to study and present.
- For further improvements, you can add hyperparameter tuning, cross-validation, SHAP analysis, or simple deployment.
