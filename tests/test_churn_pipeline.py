import joblib
import pandas as pd
import pytest

from src.churn_pipeline import (
    EXCLUDED_MODEL_COLUMNS,
    add_features,
    build_models,
    build_preprocessor,
    clean_data,
    metrics_at_threshold,
    prepare_train_test_data,
    validate_required_columns,
)


@pytest.fixture
def raw_customers() -> pd.DataFrame:
    rows = []
    for index in range(40):
        churn = "Yes" if index % 4 == 0 else "No"
        rows.append(
            {
                "customerID": f"C-{index:03d}",
                "gender": "Female" if index % 2 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 3 else "No",
                "Dependents": "No",
                "tenure": index + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic" if churn == "Yes" else "DSL",
                "OnlineSecurity": "No" if churn == "Yes" else "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No" if churn == "Yes" else "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if churn == "Yes" else "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check" if churn == "Yes" else "Credit card (automatic)",
                "MonthlyCharges": 50.0 + index,
                "TotalCharges": str((50.0 + index) * (index + 1)),
                "Churn": churn,
            }
        )
    return pd.DataFrame(rows)


def test_required_columns_accept_valid_input(raw_customers):
    validate_required_columns(raw_customers)


def test_required_columns_reject_missing_column(raw_customers):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(raw_customers.drop(columns="Contract"))


def test_target_mapping_and_supported_values(raw_customers):
    featured = add_features(clean_data(raw_customers))
    assert set(featured["ChurnFlag"].unique()) == {0, 1}

    malformed = raw_customers.copy()
    malformed.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Unsupported Churn values"):
        clean_data(malformed)


def test_total_charges_conversion(raw_customers):
    malformed_charge = raw_customers.copy()
    malformed_charge.loc[0, "TotalCharges"] = " "
    cleaned = clean_data(malformed_charge)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert pd.isna(cleaned.loc[0, "TotalCharges"])


def test_duplicate_customer_id_is_rejected(raw_customers):
    duplicate_id = raw_customers.copy()
    duplicate_id.loc[1, "customerID"] = duplicate_id.loc[0, "customerID"]
    with pytest.raises(ValueError, match="customerID must be unique"):
        clean_data(duplicate_id)


def test_identifiers_and_targets_are_excluded(raw_customers):
    featured = add_features(clean_data(raw_customers))
    X_train, X_test, _, _, numeric, categorical, reference = prepare_train_test_data(featured)
    for column in EXCLUDED_MODEL_COLUMNS:
        assert column not in X_train.columns
        assert column not in X_test.columns
        assert column not in numeric
        assert column not in categorical
    assert "customerID" in reference.columns


def test_split_is_reproducible_and_stratified(raw_customers):
    featured = add_features(clean_data(raw_customers))
    first = prepare_train_test_data(featured)
    second = prepare_train_test_data(featured)
    assert first[0].index.tolist() == second[0].index.tolist()
    assert first[2].mean() == pytest.approx(first[3].mean())


def test_pipeline_prediction_and_probability_range(raw_customers):
    featured = add_features(clean_data(raw_customers))
    X_train, X_test, y_train, _, numeric, categorical, _ = prepare_train_test_data(featured)
    preprocessor = build_preprocessor(numeric, categorical)
    pipeline = build_models(preprocessor)["balanced_logistic_regression"]
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    assert len(pipeline.predict(X_test)) == len(X_test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    assert not any("customerID" in name or "Churn" in name for name in transformed_names)


def test_threshold_mapping_and_confusion_counts():
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.10, 0.70, 0.40, 0.90])
    predictions, metrics = metrics_at_threshold(y_true, probabilities, 0.50)
    assert predictions.tolist() == [0, 1, 0, 1]
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1


def test_model_artifact_round_trip(raw_customers, tmp_path):
    featured = add_features(clean_data(raw_customers))
    X_train, X_test, y_train, _, numeric, categorical, _ = prepare_train_test_data(featured)
    pipeline = build_models(build_preprocessor(numeric, categorical))["logistic_regression"]
    pipeline.fit(X_train, y_train)
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, artifact)
    loaded = joblib.load(artifact)
    assert loaded.predict_proba(X_test).shape == (len(X_test), 2)
