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
from src.retention_intelligence import (
    add_observed_indicators,
    apply_action_rules,
    assign_risk_segments,
    calculate_priority_components,
    score_retention_customers,
    simulate_retention_strategies,
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


def test_business_linked_risk_segmentation():
    segments = assign_risk_segments([0.00, 0.1499, 0.15, 0.2999, 0.30, 0.90], 0.30)
    assert segments.astype(str).tolist() == [
        "Low Risk", "Low Risk", "Medium Risk", "Medium Risk", "High Risk", "High Risk"
    ]


def test_retention_priority_mapping_is_transparent():
    scored = pd.DataFrame(
        {
            "churn_probability": [0.90, 0.40, 0.20, 0.05],
            "risk_segment": ["High Risk", "High Risk", "Medium Risk", "Low Risk"],
            "MonthlyCharges": [100, 50, 40, 20],
            "TotalCharges": [5000, 1000, 500, 100],
            "tenure": [6, 24, 36, 72],
            "Contract": ["Month-to-month", "One year", "One year", "Two year"],
        }
    )
    prioritized = calculate_priority_components(scored)
    assert prioritized["priority_level"].tolist() == [
        "Priority 1", "Priority 2", "Priority 3", "Monitor"
    ]
    assert prioritized["customer_value_proxy"].between(0, 100).all()
    assert prioritized["intervention_urgency"].between(0, 100).all()


def test_readable_indicator_and_action_output():
    scored = pd.DataFrame(
        {
            "Contract": ["Month-to-month"],
            "tenure": [3],
            "InternetService": ["Fiber optic"],
            "PaymentMethod": ["Electronic check"],
            "TechSupport": ["No"],
            "OnlineSecurity": ["No"],
            "SupportServiceCount": [0],
            "risk_segment": ["High Risk"],
            "priority_level": ["Priority 1"],
        }
    )
    evidence = pd.DataFrame(
        {
            "indicator_code": ["month_to_month", "short_tenure"],
            "validated": [True, True],
            "churn_rate_uplift": [0.16, 0.20],
        }
    )
    explained = add_observed_indicators(scored, evidence)
    acted = apply_action_rules(explained)
    assert acted.loc[0, "reason_type"] == "Observed risk indicators"
    assert acted.loc[0, "primary_reason"] == "Tenure of 12 months or less"
    assert acted.loc[0, "suggested_action"] == "Contract-upgrade incentive"
    assert isinstance(acted.loc[0, "expected_mechanism"], str)


def test_one_customer_inference_and_malformed_handling(raw_customers):
    featured = add_features(clean_data(raw_customers))
    X_train, _, y_train, _, numeric, categorical, _ = prepare_train_test_data(featured)
    pipeline = build_models(build_preprocessor(numeric, categorical))["logistic_regression"]
    pipeline.fit(X_train, y_train)

    customer = raw_customers.drop(columns="Churn").iloc[[0]].copy()
    result = score_retention_customers(customer, pipeline, 0.30)
    required = {
        "churn_probability", "predicted_churn_flag", "risk_segment", "priority_level",
        "suggested_action", "primary_reason", "secondary_reason",
    }
    assert required.issubset(result.columns)
    assert result.loc[result.index[0], "churn_probability"] >= 0
    assert result.loc[result.index[0], "churn_probability"] <= 1

    with pytest.raises(ValueError, match="Missing required inference columns"):
        score_retention_customers(customer.drop(columns="Contract"), pipeline, 0.30)


def test_hypothetical_strategy_simulation_reconciles_counts():
    queue = pd.DataFrame(
        {
            "actual_churn_flag": [1, 1, 0, 0],
            "churn_probability": [0.80, 0.40, 0.60, 0.10],
            "priority_level": ["Priority 1", "Priority 2", "Priority 1", "Monitor"],
        }
    )
    assumptions = {
        "outreach_cost_per_customer": 1.0,
        "incentive_cost_per_customer": 1.0,
        "missed_churn_cost_per_customer": 10.0,
        "intervention_success_rate": 0.50,
    }
    result = simulate_retention_strategies(queue, 0.30, assumptions).set_index("strategy")
    no_action = result.loc["No intervention"]
    assert no_action["customers_contacted"] == 0
    assert no_action["churners_missed"] == 2
    assert no_action["hypothetical_net_value_vs_no_intervention"] == 0

    selected = result.loc["Selected threshold 0.30"]
    assert selected["customers_contacted"] == 3
    assert selected["churners_captured"] == 2
    assert selected["false_positive_interventions"] == 1
    assert selected["expected_retained_customers"] == 1
