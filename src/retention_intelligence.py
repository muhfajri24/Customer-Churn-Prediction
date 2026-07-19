"""Decision-layer utilities for customer-retention prioritization.

The functions in this module translate validated churn probabilities into
transparent portfolio segments, priorities, observed indicators, actions, and
Power BI-ready outputs. They do not train an additional decision model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


RISK_METHOD = "business_threshold_linked"
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
PRIORITY_LABELS = ["Priority 1", "Priority 2", "Priority 3", "Monitor"]
DEFAULT_SIMULATION_ASSUMPTIONS = {
    "outreach_cost_per_customer": 5.0,
    "incentive_cost_per_customer": 20.0,
    "missed_churn_cost_per_customer": 300.0,
    "intervention_success_rate": 0.25,
}


def assign_risk_segments(probabilities: Iterable[float], threshold: float) -> pd.Series:
    """Map probabilities to segments anchored to the intervention threshold."""
    probability = pd.Series(probabilities, dtype=float)
    if not probability.between(0, 1).all():
        raise ValueError("Churn probabilities must be between 0 and 1.")
    if not 0 < threshold < 1:
        raise ValueError("Decision threshold must be between 0 and 1.")

    medium_boundary = threshold / 2
    values = np.select(
        [probability < medium_boundary, probability < threshold],
        ["Low Risk", "Medium Risk"],
        default="High Risk",
    )
    return pd.Series(
        pd.Categorical(values, categories=RISK_LABELS, ordered=True),
        index=probability.index,
        name="risk_segment",
    )


def assign_quantile_segments(probabilities: Iterable[float]) -> pd.Series:
    """Create equal-sized comparison segments without using them operationally."""
    probability = pd.Series(probabilities, dtype=float)
    ranked = probability.rank(method="first")
    segments = pd.qcut(ranked, q=3, labels=RISK_LABELS)
    return pd.Series(segments, index=probability.index, name="risk_segment")


def summarize_risk_segments(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize segment size, calibration evidence, economics, and contracts."""
    total = len(scored_df)
    summary = (
        scored_df.groupby("risk_segment", observed=False)
        .agg(
            customer_count=("customerID", "size"),
            average_churn_probability=("churn_probability", "mean"),
            actual_churn_rate=("actual_churn_flag", "mean"),
            average_tenure=("tenure", "mean"),
            average_monthly_charges=("MonthlyCharges", "mean"),
        )
        .reset_index()
    )
    summary["customer_percentage"] = summary["customer_count"] / total

    contract_mix = pd.crosstab(
        scored_df["risk_segment"], scored_df["Contract"], normalize="index"
    ).reindex(index=RISK_LABELS, fill_value=0)
    contract_mix.columns = [
        f"contract_{str(column).lower().replace(' ', '_').replace('-', '_')}_percentage"
        for column in contract_mix.columns
    ]
    summary = summary.merge(contract_mix.reset_index(), on="risk_segment", how="left")
    numeric_columns = summary.select_dtypes(include="number").columns
    summary[numeric_columns] = summary[numeric_columns].round(4)
    return summary


def compare_segmentation_methods(scored_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Compare decision-linked boundaries with equal-sized probability groups."""
    comparisons = []
    for method, segments in {
        RISK_METHOD: assign_risk_segments(scored_df["churn_probability"], threshold),
        "probability_quantiles": assign_quantile_segments(scored_df["churn_probability"]),
    }.items():
        candidate = scored_df.copy()
        candidate["risk_segment"] = segments.values
        summary = summarize_risk_segments(candidate)
        summary.insert(0, "method", method)
        comparisons.append(summary)
    return pd.concat(comparisons, ignore_index=True)


def calculate_priority_components(
    scored_df: pd.DataFrame,
    value_reference: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Calculate a transparent score; the value component is explicitly a proxy."""
    result = scored_df.copy()
    if value_reference is None:
        value_reference = {
            "monthly_charges_p95": float(result["MonthlyCharges"].quantile(0.95)),
            "total_charges_p95": float(result["TotalCharges"].quantile(0.95)),
        }
    monthly_scale = value_reference["monthly_charges_p95"]
    total_scale = value_reference["total_charges_p95"]
    if monthly_scale <= 0 or total_scale <= 0:
        raise ValueError("Customer-value reference scales must be positive.")
    monthly_component = (result["MonthlyCharges"] / monthly_scale).clip(0, 1)
    total_component = (result["TotalCharges"].fillna(0) / total_scale).clip(0, 1)
    result["customer_value_proxy"] = (100 * (0.60 * monthly_component + 0.40 * total_component)).round(2)

    short_tenure = 1 - result["tenure"].clip(lower=0, upper=72) / 72
    flexible_contract = result["Contract"].eq("Month-to-month").astype(float)
    result["intervention_urgency"] = (100 * (0.60 * short_tenure + 0.40 * flexible_contract)).round(2)
    result["retention_priority_score"] = (
        100
        * (
            0.60 * result["churn_probability"]
            + 0.25 * result["customer_value_proxy"] / 100
            + 0.15 * result["intervention_urgency"] / 100
        )
    ).round(2)

    result["priority_level"] = np.select(
        [
            result["risk_segment"].eq("High Risk") & result["retention_priority_score"].ge(70),
            result["risk_segment"].eq("High Risk"),
            result["risk_segment"].eq("Medium Risk"),
        ],
        ["Priority 1", "Priority 2", "Priority 3"],
        default="Monitor",
    )
    return result


def indicator_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return interpretable customer characteristics considered as indicators."""
    return {
        "month_to_month": df["Contract"].eq("Month-to-month"),
        "short_tenure": df["tenure"].le(12),
        "fiber_optic": df["InternetService"].eq("Fiber optic"),
        "electronic_check": df["PaymentMethod"].eq("Electronic check"),
        "no_tech_support": df["TechSupport"].eq("No"),
        "no_online_security": df["OnlineSecurity"].eq("No"),
        "limited_support_adoption": df["SupportServiceCount"].le(1),
    }


INDICATOR_LABELS = {
    "month_to_month": "Month-to-month contract",
    "short_tenure": "Tenure of 12 months or less",
    "fiber_optic": "Fiber-optic internet service",
    "electronic_check": "Electronic-check payment",
    "no_tech_support": "No technical-support subscription",
    "no_online_security": "No online-security subscription",
    "limited_support_adoption": "Limited support-service adoption",
}


def validate_observed_indicators(historical_df: pd.DataFrame) -> pd.DataFrame:
    """Keep candidate indicators supported by prevalence and observed churn uplift."""
    overall_rate = historical_df["ChurnFlag"].mean()
    rows = []
    for code, mask in indicator_masks(historical_df).items():
        count = int(mask.sum())
        churn_rate = historical_df.loc[mask, "ChurnFlag"].mean()
        rows.append(
            {
                "indicator_code": code,
                "business_label": INDICATOR_LABELS[code],
                "customer_count": count,
                "observed_churn_rate": churn_rate,
                "overall_churn_rate": overall_rate,
                "churn_rate_uplift": churn_rate - overall_rate,
                "validated": count >= 100 and churn_rate >= overall_rate + 0.05,
            }
        )
    result = pd.DataFrame(rows).sort_values("churn_rate_uplift", ascending=False)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(4)
    return result


def add_observed_indicators(scored_df: pd.DataFrame, indicator_evidence: pd.DataFrame) -> pd.DataFrame:
    """Attach up to two readable observed indicators to every scored customer."""
    result = scored_df.copy()
    valid_codes = indicator_evidence.loc[indicator_evidence["validated"], "indicator_code"].tolist()
    masks = indicator_masks(result)
    uplift_order = indicator_evidence.set_index("indicator_code")["churn_rate_uplift"].to_dict()
    valid_codes.sort(key=lambda code: uplift_order[code], reverse=True)

    primary, secondary = [], []
    for row_index in result.index:
        labels = [INDICATOR_LABELS[code] for code in valid_codes if bool(masks[code].loc[row_index])]
        primary.append(labels[0] if labels else "Elevated model probability")
        secondary.append(labels[1] if len(labels) > 1 else "No additional validated indicator")
    result["primary_reason"] = primary
    result["secondary_reason"] = secondary
    result["reason_type"] = "Observed risk indicators"
    return result


def apply_action_rules(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, readable action rules instead of another model."""
    result = scored_df.copy()
    high = result["risk_segment"].eq("High Risk")
    conditions = [
        high & result["Contract"].eq("Month-to-month"),
        high & result["tenure"].le(12),
        high & result["TechSupport"].eq("No"),
        high & result["PaymentMethod"].eq("Electronic check"),
        high,
        result["risk_segment"].eq("Medium Risk"),
    ]
    result["suggested_action"] = np.select(
        conditions,
        [
            "Contract-upgrade incentive",
            "Onboarding support outreach",
            "Technical-support outreach",
            "Payment-method assistance",
            "Retention call",
            "Targeted retention email",
        ],
        default="Monitor only",
    )
    result["expected_mechanism"] = np.select(
        conditions,
        [
            "Encourage a longer commitment while reducing avoidable switching friction.",
            "Resolve early-life experience gaps before disengagement increases.",
            "Reduce unresolved service and support friction.",
            "Reduce payment-process friction and offer a more stable payment option.",
            "Review needs and service friction with the customer.",
            "Use a lower-cost channel to test engagement before direct outreach.",
        ],
        default="Observe risk movement without intervention cost.",
    )
    result["suggested_success_metric"] = np.select(
        conditions,
        [
            "Contract conversion rate and 90-day retained rate",
            "Outreach completion and 90-day retained rate",
            "Support activation and 90-day retained rate",
            "Payment-method change and 90-day retained rate",
            "Contact rate and 90-day retained rate",
            "Email engagement and 90-day retained rate",
        ],
        default="Risk-segment migration and observed churn rate",
    )
    return result


def build_retention_queue(
    scored_df: pd.DataFrame,
    threshold: float,
    indicator_evidence: pd.DataFrame,
    value_reference: Dict[str, float],
) -> pd.DataFrame:
    """Build the customer-level decision queue from held-out model scores."""
    result = scored_df.copy()
    result["risk_segment"] = assign_risk_segments(result["churn_probability"], threshold).values
    result = calculate_priority_components(result, value_reference)
    result = add_observed_indicators(result, indicator_evidence)
    result = apply_action_rules(result)
    priority_order = pd.Categorical(result["priority_level"], categories=PRIORITY_LABELS, ordered=True)
    result = result.assign(_priority_order=priority_order).sort_values(
        ["_priority_order", "retention_priority_score", "churn_probability"],
        ascending=[True, False, False],
    ).drop(columns="_priority_order")
    return result


def build_global_churn_drivers(
    historical_df: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine model importance with non-causal observed outcome associations."""
    overall = historical_df["ChurnFlag"].mean()
    importance = importance_df.set_index("feature")["importance_mean"].clip(lower=0).to_dict()
    rows: List[Dict[str, Any]] = []
    for feature, importance_value in importance.items():
        if importance_value <= 0 or feature not in historical_df.columns:
            continue
        series = historical_df[feature]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 5:
            lower, upper = series.quantile([0.25, 0.75])
            groups = [
                (f"Low {feature} (<= Q1)", series.le(lower)),
                (f"High {feature} (>= Q3)", series.ge(upper)),
            ]
        else:
            rates = historical_df.groupby(feature, observed=False)["ChurnFlag"].agg(["mean", "size"])
            rates = rates[rates["size"] >= 50]
            if rates.empty:
                continue
            groups = [
                (f"{feature}: {rates['mean'].idxmax()}", series.eq(rates["mean"].idxmax())),
                (f"{feature}: {rates['mean'].idxmin()}", series.eq(rates["mean"].idxmin())),
            ]
        for label, mask in groups:
            rate = historical_df.loc[mask, "ChurnFlag"].mean()
            delta = rate - overall
            rows.append(
                {
                    "feature": feature,
                    "business_indicator": label,
                    "indicator_type": "Higher churn indicator" if delta >= 0 else "Retention indicator",
                    "model_importance": importance_value,
                    "customer_count": int(mask.sum()),
                    "observed_churn_rate": rate,
                    "overall_churn_rate": overall,
                    "association_delta": delta,
                    "evidence_basis": "Permutation importance plus observed historical association; not causal",
                }
            )
    drivers = pd.DataFrame(rows)
    drivers["decision_relevance"] = drivers["model_importance"] * drivers["association_delta"].abs()
    numeric = drivers.select_dtypes(include="number").columns
    drivers[numeric] = drivers[numeric].round(4)
    return drivers.sort_values("decision_relevance", ascending=False)


def save_driver_figures(drivers: pd.DataFrame, figure_dir: Path) -> None:
    """Save separate non-causal churn and retention indicator figures."""
    specifications = [
        ("Higher churn indicator", "top_churn_indicators.png", "Observed Indicators Associated with Higher Churn", "#B45309"),
        ("Retention indicator", "top_retention_indicators.png", "Observed Indicators Associated with Lower Churn", "#2563EB"),
    ]
    for indicator_type, filename, title, color in specifications:
        chart = drivers[drivers["indicator_type"] == indicator_type].head(8).copy()
        chart = chart.sort_values("decision_relevance")
        plt.figure(figsize=(10, 6))
        plt.barh(chart["business_indicator"], chart["decision_relevance"], color=color, edgecolor="#1F2937")
        plt.title(title)
        plt.xlabel("Decision relevance (importance × absolute churn-rate difference)")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=200)
        plt.close()


def save_risk_figure(summary: pd.DataFrame, figure_dir: Path) -> None:
    """Save an honest zero-based segment-count chart."""
    ordered = summary.set_index("risk_segment").reindex(RISK_LABELS).reset_index()
    plt.figure(figsize=(8, 5))
    bars = plt.bar(ordered["risk_segment"], ordered["customer_count"], color=["#CBD5E1", "#F59E0B", "#B45309"], edgecolor="#1F2937")
    plt.bar_label(bars, labels=[f"{int(value):,}" for value in ordered["customer_count"]], padding=3)
    plt.ylim(0, ordered["customer_count"].max() * 1.18)
    plt.title("Held-out Customers by Churn-Risk Segment")
    plt.xlabel("Business-linked boundaries: Low < threshold/2; Medium < threshold; High >= threshold")
    plt.ylabel("Customer count")
    plt.tight_layout()
    plt.savefig(figure_dir / "risk_segment_distribution.png", dpi=200)
    plt.close()


def build_error_analysis(queue: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return customer-level false negatives and false positives."""
    errors = queue[
        queue["actual_churn_flag"].ne(queue["predicted_churn_flag"])
    ].copy()
    errors["error_type"] = np.where(
        errors["actual_churn_flag"].eq(1), "False Negative", "False Positive"
    )
    errors["distance_from_threshold"] = (errors["churn_probability"] - threshold).abs().round(4)
    columns = [
        "customerID", "error_type", "actual_churn_flag", "predicted_churn_flag",
        "churn_probability", "distance_from_threshold", "risk_segment", "priority_level",
        "Contract", "tenure", "TenureGroup", "PaymentMethod", "InternetService",
        "TechSupport", "OnlineSecurity", "SupportServiceCount", "MonthlyCharges",
        "primary_reason", "secondary_reason",
    ]
    return errors[columns].sort_values(["error_type", "distance_from_threshold"])


def simulate_retention_strategies(
    queue: pd.DataFrame,
    selected_threshold: float,
    assumptions: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compare contact strategies under explicitly hypothetical assumptions.

    Net value is measured relative to the no-intervention churn-loss baseline:
    baseline churn loss minus intervention cost and expected remaining churn loss.
    """
    assumptions = dict(DEFAULT_SIMULATION_ASSUMPTIONS if assumptions is None else assumptions)
    required = {
        "outreach_cost_per_customer",
        "incentive_cost_per_customer",
        "missed_churn_cost_per_customer",
        "intervention_success_rate",
    }
    missing = sorted(required - set(assumptions))
    if missing:
        raise ValueError(f"Missing simulation assumptions: {', '.join(missing)}")
    if not 0 <= assumptions["intervention_success_rate"] <= 1:
        raise ValueError("Intervention success rate must be between 0 and 1.")

    actual_churn = queue["actual_churn_flag"].eq(1)
    strategies = {
        "No intervention": pd.Series(False, index=queue.index),
        "Contact all customers": pd.Series(True, index=queue.index),
        "Default threshold 0.50": queue["churn_probability"].ge(0.50),
        f"Selected threshold {selected_threshold:.2f}": queue["churn_probability"].ge(selected_threshold),
        "Priority 1 customers only": queue["priority_level"].eq("Priority 1"),
    }
    total_churners = int(actual_churn.sum())
    baseline_churn_loss = total_churners * assumptions["missed_churn_cost_per_customer"]
    rows = []
    for strategy, contacted in strategies.items():
        contacted = contacted.astype(bool)
        customers_contacted = int(contacted.sum())
        churners_captured = int((contacted & actual_churn).sum())
        churners_missed = int((~contacted & actual_churn).sum())
        false_positive_interventions = int((contacted & ~actual_churn).sum())
        expected_retained = churners_captured * assumptions["intervention_success_rate"]
        expected_remaining_churners = total_churners - expected_retained
        intervention_cost = customers_contacted * (
            assumptions["outreach_cost_per_customer"]
            + assumptions["incentive_cost_per_customer"]
        )
        expected_churn_loss = expected_remaining_churners * assumptions["missed_churn_cost_per_customer"]
        total_cost = intervention_cost + expected_churn_loss
        rows.append(
            {
                "strategy": strategy,
                "customers_contacted": customers_contacted,
                "contact_rate": customers_contacted / len(queue),
                "churners_captured": churners_captured,
                "churners_missed": churners_missed,
                "false_positive_interventions": false_positive_interventions,
                "expected_retained_customers": expected_retained,
                "intervention_cost": intervention_cost,
                "expected_remaining_churn_loss": expected_churn_loss,
                "hypothetical_total_cost": total_cost,
                "hypothetical_net_value_vs_no_intervention": baseline_churn_loss - total_cost,
                **assumptions,
            }
        )
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].round(2)
    return result


def write_simulation_assumptions(
    simulation: pd.DataFrame,
    output_path: Path,
) -> None:
    """Document the hypothetical scenario and calculation boundaries."""
    first = simulation.iloc[0]
    lines = [
        "# Hypothetical Retention Strategy Simulation",
        "",
        "All monetary-style values are hypothetical value units for scenario comparison. "
        "They are not telecom-company costs, revenue, or measured intervention outcomes.",
        "",
        "## Assumptions",
        "",
        f"- Outreach cost per contacted customer: {first['outreach_cost_per_customer']:.2f}",
        f"- Incentive cost per contacted customer: {first['incentive_cost_per_customer']:.2f}",
        f"- Cost assigned to one remaining churner: {first['missed_churn_cost_per_customer']:.2f}",
        f"- Intervention success rate among contacted actual churners: {first['intervention_success_rate']:.0%}",
        "",
        "## Calculation",
        "",
        "Expected retained customers = contacted actual churners × success rate.",
        "",
        "Hypothetical total cost = contact and incentive cost + expected remaining churn loss.",
        "",
        "Hypothetical net value = no-intervention churn-loss baseline − strategy total cost.",
        "",
        "## Interpretation",
        "",
        "This simulation tests the mechanics and trade-offs of targeting strategies only. "
        "It does not estimate realized ROI. Different assumptions can change the ranking.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_error_summary(errors: pd.DataFrame, output_path: Path) -> None:
    """Document where prediction errors concentrate without fairness claims."""
    lines = [
        "# Error Analysis Summary",
        "",
        "This analysis covers held-out false negatives and false positives. It does not establish bias or causality.",
    ]
    for error_type in ["False Negative", "False Positive"]:
        subset = errors[errors["error_type"] == error_type]
        lines.extend(["", f"## {error_type}s", "", f"Customer count: {len(subset)}"])
        for column in ["Contract", "TenureGroup", "PaymentMethod", "InternetService", "SupportServiceCount"]:
            counts = subset[column].value_counts(dropna=False)
            if not counts.empty:
                lines.append(f"- Largest {column} group: {counts.index[0]} ({int(counts.iloc[0])} customers)")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_retention_recommendations(
    queue: pd.DataFrame,
    indicator_evidence: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write action hypotheses tied to observed evidence and measurable outcomes."""
    evidence = indicator_evidence.set_index("business_label")
    action_map = {
        "Contract-upgrade incentive": "Month-to-month contract",
        "Onboarding support outreach": "Tenure of 12 months or less",
        "Technical-support outreach": "No technical-support subscription",
        "Payment-method assistance": "Electronic-check payment",
    }
    lines = [
        "# Retention Recommendations",
        "",
        "These are testable intervention hypotheses, not actions proven to prevent churn.",
    ]
    for action in queue["suggested_action"].drop_duplicates():
        sample = queue[queue["suggested_action"] == action].iloc[0]
        indicator = action_map.get(action)
        if indicator in evidence.index:
            row = evidence.loc[indicator]
            observed = f"{indicator}: {row['observed_churn_rate']:.2%} observed churn versus {row['overall_churn_rate']:.2%} overall."
        else:
            observed = f"Assigned from {sample['risk_segment']} and the selected churn-probability threshold."
        lines.extend(
            [
                "",
                f"## {action}",
                "",
                f"- Target segment: {sample['risk_segment']} / {sample['priority_level']}",
                f"- Observed evidence: {observed}",
                f"- Suggested action: {action}",
                f"- Expected mechanism: {sample['expected_mechanism']}",
                f"- Suggested success metric: {sample['suggested_success_metric']}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_model_metadata(
    output_path: Path,
    model_name: str,
    threshold: float,
    feature_columns: List[str],
    metrics: Dict[str, Any],
    random_seed: int,
    value_reference: Dict[str, float],
) -> None:
    """Persist the minimum information needed to audit inference decisions."""
    metadata = {
        "model_name": model_name,
        "selected_threshold": threshold,
        "target_mapping": {"No": 0, "Yes": 1},
        "feature_columns": feature_columns,
        "excluded_columns": ["customerID", "Churn", "ChurnFlag"],
        "risk_segmentation_method": RISK_METHOD,
        "risk_boundaries": {
            "low_risk": f"probability < {threshold / 2:.2f}",
            "medium_risk": f"{threshold / 2:.2f} <= probability < {threshold:.2f}",
            "high_risk": f"probability >= {threshold:.2f}",
        },
        "priority_score": "60% churn probability + 25% customer value proxy + 15% intervention urgency",
        "customer_value_proxy_reference": value_reference,
        "evaluation_metrics": metrics,
        "random_seed": random_seed,
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_inference_features(raw_df: pd.DataFrame, expected_features: List[str]) -> Tuple[pd.Series, pd.DataFrame]:
    """Validate and engineer unlabeled customer records for pipeline inference."""
    required_raw = {
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
    }
    missing = sorted(required_raw - set(raw_df.columns))
    if missing:
        raise ValueError(f"Missing required inference columns: {', '.join(missing)}")
    if raw_df.empty:
        raise ValueError("Inference input is empty.")
    if raw_df["customerID"].isna().any() or raw_df["customerID"].astype(str).str.strip().eq("").any():
        raise ValueError("customerID contains missing or empty values.")
    if raw_df["customerID"].duplicated().any():
        raise ValueError("customerID must be unique for inference.")

    featured = raw_df.copy()
    featured["TotalCharges"] = pd.to_numeric(featured["TotalCharges"], errors="coerce")
    for column in featured.select_dtypes(include="object").columns:
        featured[column] = featured[column].astype(str).str.strip()
    featured["TenureGroup"] = pd.cut(
        featured["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"],
    )
    featured["AvgMonthlySpend"] = featured["TotalCharges"] / featured["tenure"].replace(0, 1)
    support_columns = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    featured["SupportServiceCount"] = featured[support_columns].eq("Yes").sum(axis=1)
    missing_features = sorted(set(expected_features) - set(featured.columns))
    if missing_features:
        raise ValueError(f"Unable to create model features: {', '.join(missing_features)}")
    return featured["customerID"].copy(), featured.reindex(columns=expected_features)


def score_retention_customers(
    raw_df: pd.DataFrame,
    pipeline: Pipeline,
    threshold: float,
) -> pd.DataFrame:
    """Run reusable inference without duplicating the fitted preprocessing."""
    expected_features = list(pipeline.feature_names_in_)
    customer_ids, features = prepare_inference_features(raw_df, expected_features)
    probabilities = pipeline.predict_proba(features)[:, 1]
    output = raw_df.copy().reset_index(drop=True)
    output["customerID"] = customer_ids.reset_index(drop=True)
    output["churn_probability"] = probabilities
    output["predicted_churn_flag"] = (probabilities >= threshold).astype(int)
    output["risk_segment"] = assign_risk_segments(probabilities, threshold).values
    engineered = features.reset_index(drop=True)
    output["TotalCharges"] = engineered["TotalCharges"]
    for column in ["TenureGroup", "AvgMonthlySpend", "SupportServiceCount"]:
        output[column] = engineered[column]
    evidence = pd.DataFrame(
        {"indicator_code": list(INDICATOR_LABELS), "validated": True, "churn_rate_uplift": range(len(INDICATOR_LABELS), 0, -1)}
    )
    value_reference = getattr(pipeline, "retention_value_reference_", None)
    output = calculate_priority_components(output, value_reference)
    output = add_observed_indicators(output, evidence)
    return apply_action_rules(output)


def load_and_score_retention_customers(
    raw_df: pd.DataFrame,
    model_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    """Load the complete trained pipeline and its threshold for inference."""
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return score_retention_customers(raw_df, pipeline, float(metadata["selected_threshold"]))
