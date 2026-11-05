import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, brier_score_loss, confusion_matrix
)
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

np.random.seed(42)

def generate_unbanked_dataset(n_samples=8000):
    """Generates realistic alternative financial telemetry for unbanked individuals."""
    # Demographic attribute for Fairness Auditing (NOT used directly for unfair discrimination)
    is_rural = np.random.binomial(1, 0.55, n_samples)

    # 1. Utility & Telecom Bill Timeliness (0 to 100% on-time over 12 months)
    utility_on_time_pct = np.clip(np.random.normal(78, 18, n_samples), 10, 100)

    # 2. Digital Wallet / UPI Monthly Transaction Consistency (CV of inflows: lower is steadier)
    income_volatility_index = np.clip(np.random.gamma(2.2, 0.15, n_samples), 0.05, 1.2)

    # 3. Essential vs. Discretionary Spend Ratio (higher = more disciplined spending)
    essential_spend_ratio = np.clip(np.random.beta(5, 2.5, n_samples), 0.20, 0.95)

    # 4. Micro-Savings Retention Rate (% of monthly inflow retained at month-end)
    savings_retention_pct = np.clip(np.random.exponential(12, n_samples), 0, 45)

    # 5. Gig / Co-op / Vendor Tenure & Platform Rating (Months active: 1 to 60)
    livelihood_tenure_months = np.clip(np.random.weibull(1.8, n_samples) * 24, 1, 60)

    # 6. Community / SHG (Self-Help Group) or Peer Vouching Score (0 to 100)
    community_trust_score = np.clip(
        np.random.normal(70 + (is_rural * 6), 15, n_samples), 20, 100
    )

    # 7. Prepaid Recharge Emergency Gap Days (Days phone went un-recharged due to cash crunch)
    telecom_gap_days = np.clip(np.random.poisson(3.5, n_samples), 0, 25)

    # 8. Cash-Flow Shock Recovery Speed (Days to recover balance after a major expense)
    shock_recovery_days = np.clip(
        np.random.normal(14 - (savings_retention_pct * 0.25), 5, n_samples), 2, 35
    )

    # Non-linear ground-truth creditworthiness log-odds
    logit = (
        0.045 * (utility_on_time_pct - 70)
        - 2.40 * (income_volatility_index - 0.35)
        + 2.80 * (essential_spend_ratio - 0.60)
        + 0.085 * (savings_retention_pct - 10)
        + 0.035 * (livelihood_tenure_months - 18)
        + 0.030 * (community_trust_score - 65)
        - 0.160 * (telecom_gap_days - 3)
        - 0.090 * (shock_recovery_days - 12)
        # Non-linear interaction: high utility discipline + high savings protects against volatility
        + 0.025 * ((utility_on_time_pct / 100.0) * savings_retention_pct)
        + np.random.normal(0, 0.45, n_samples)
    )

    prob_repay = 1 / (1 + np.exp(-logit))
    repaid_on_time = (prob_repay >= 0.50).astype(int)

    df = pd.DataFrame({
        "utility_on_time_pct": np.round(utility_on_time_pct, 1),
        "income_volatility_index": np.round(income_volatility_index, 3),
        "essential_spend_ratio": np.round(essential_spend_ratio, 2),
        "savings_retention_pct": np.round(savings_retention_pct, 1),
        "livelihood_tenure_months": np.round(livelihood_tenure_months, 1),
        "community_trust_score": np.round(community_trust_score, 1),
        "telecom_gap_days": telecom_gap_days.astype(int),
        "shock_recovery_days": np.round(shock_recovery_days, 1),
        "is_rural": is_rural,
        "repaid_on_time": repaid_on_time
    })
    return df

def run_training_pipeline():
    print("1. Generating alternative credit dataset (8,000 unbanked profiles)...")
    df = generate_unbanked_dataset(8000)
    feature_cols = [
        "utility_on_time_pct",
        "income_volatility_index",
        "essential_spend_ratio",
        "savings_retention_pct",
        "livelihood_tenure_months",
        "community_trust_score",
        "telecom_gap_days",
        "shock_recovery_days"
    ]

    X = df[feature_cols]
    y = df["repaid_on_time"]
    rural_flag = df["is_rural"]

    X_train, X_test, y_train, y_test, rural_train, rural_test = train_test_split(
        X, y, rural_flag, test_size=0.20, random_state=42, stratify=y
    )

    print("2. Training candidate models & Stacking Ensemble...")
    xgb_model = XGBClassifier(
        n_estimators=220, max_depth=4, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, random_state=42, eval_metric="logloss"
    )
    lgb_model = LGBMClassifier(
        n_estimators=220, max_depth=4, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1
    )
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    lr_model = LogisticRegression(max_iter=500)

    ensemble = StackingClassifier(
        estimators=[("xgb", xgb_model), ("lgb", lgb_model), ("rf", rf_model)],
        final_estimator=LogisticRegression(),
        cv=5
    )

    models = {
        "Logistic Regression (Baseline)": lr_model,
        "Random Forest": rf_model,
        "LightGBM": lgb_model,
        "XGBoost (Primary XAI Tree)": xgb_model,
        "CrediTrust Stacking Ensemble": ensemble
    }

    benchmark_results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        probs = m.predict_proba(X_test)[:, 1]
        benchmark_results[name] = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            "f1_score": round(float(f1_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds)), 4),
            "recall": round(float(recall_score(y_test, preds)), 4),
            "brier_loss": round(float(brier_score_loss(y_test, probs)), 4)
        }

    # Fairness Audit on Test Set (Disparate Impact Ratio: Rural vs Urban)
    ens_preds = ensemble.predict(X_test)
    rural_approval_rate = float(ens_preds[rural_test == 1].mean())
    urban_approval_rate = float(ens_preds[rural_test == 0].mean())
    disparate_impact_ratio = round(rural_approval_rate / max(urban_approval_rate, 1e-5), 4)

    # Equal Opportunity Difference (True Positive Rate in Rural vs Urban)
    rural_tpr = float(ens_preds[(rural_test == 1) & (y_test == 1)].mean())
    urban_tpr = float(ens_preds[(rural_test == 0) & (y_test == 1)].mean())
    eod = round(abs(rural_tpr - urban_tpr), 4)

    cm = confusion_matrix(y_test, ens_preds).tolist()

    print("3. Computing fast TreeSHAP explainer on primary XGBoost estimator...")
    explainer = shap.TreeExplainer(xgb_model)

    bundle = {
        "ensemble_model": ensemble,
        "xgb_model": xgb_model,
        "explainer": explainer,
        "feature_cols": feature_cols,
        "feature_medians": X_train.median().to_dict(),
        "feature_bounds": {
            col: (float(X_train[col].min()), float(X_train[col].max())) for col in feature_cols
        }
    }
    joblib.dump(bundle, "model_bundle.pkl")

    report = {
        "dataset_size": len(df),
        "test_size": len(X_test),
        "benchmarks": benchmark_results,
        "confusion_matrix": cm,
        "fairness_audit": {
            "rural_approval_rate": round(rural_approval_rate, 4),
            "urban_approval_rate": round(urban_approval_rate, 4),
            "disparate_impact_ratio": disparate_impact_ratio,
            "equal_opportunity_diff": eod,
            "four_fifths_rule_passed": bool(0.80 <= disparate_impact_ratio <= 1.25)
        }
    }
    with open("metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Training Complete! Summary Metrics:")
    print(json.dumps(report["benchmarks"]["CrediTrust Stacking Ensemble"], indent=2))
    print("\n✅ Fairness Audit (80% Rule Check):")
    print(json.dumps(report["fairness_audit"], indent=2))

if __name__ == "__main__":
    run_training_pipeline()
