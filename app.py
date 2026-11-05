import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from train_engine import run_training_pipeline, generate_unbanked_dataset

st.set_page_config(
    page_title="CrediTrust XAI | Unbanked Credit Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom enterprise styling
st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .badge-pass {
        background-color: #DCFCE7; color: #166534;
        padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
    .badge-warn {
        background-color: #FEF9C3; color: #854D0E;
        padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #FEE2E2; color: #991B1B;
        padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_META = {
    "utility_on_time_pct": {
        "label": "Utility & Telecom On-Time Payment (%)",
        "unit": "%",
        "direction": "higher",
        "step": 5.0,
        "max_realistic_ improvement": 22.0,
        "action_template": "Pay electricity & mobile bills before the due date to boost on-time rate from {cur:.0f}% to {target:.0f}%."
    },
    "income_volatility_index": {
        "label": "UPI / Mobile Wallet Inflow Volatility (CV)",
        "unit": "CV",
        "direction": "lower",
        "step": -0.05,
        "max_realistic_improvement": 0.25,
        "action_template": "Deposit daily cash earnings into UPI/mobile wallet consistently to reduce inflow volatility from {cur:.2f} to {target:.2f}."
    },
    "essential_spend_ratio": {
        "label": "Essential vs. Total Spend Ratio",
        "unit": "ratio",
        "direction": "higher",
        "step": 0.05,
        "max_realistic_improvement": 0.20,
        "action_template": "Route inventory/household essentials through tracked digital payments to raise essential spend ratio from {cur:.2f} to {target:.2f}."
    },
    "savings_retention_pct": {
        "label": "Monthly Micro-Savings Retention (%)",
        "unit": "%",
        "direction": "higher",
        "step": 3.0,
        "max_realistic_improvement": 12.0,
        "action_template": "Maintain an automated month-end micro-savings buffer of {target:.1f}% (up from {cur:.1f}%)."
    },
    "livelihood_tenure_months": {
        "label": "Verified Livelihood / Platform Tenure (Months)",
        "unit": "mo",
        "direction": "higher",
        "step": 3.0,
        "max_realistic_improvement": 6.0,
        "action_template": "Maintain continuous active platform/vendor registration for {target:.0f} months (currently {cur:.0f} mo)."
    },
    "community_trust_score": {
        "label": "SHG / Co-op / Peer Vouching Score (0-100)",
        "unit": "pts",
        "direction": "higher",
        "step": 5.0,
        "max_realistic_improvement": 15.0,
        "action_template": "Complete Self-Help Group (SHG) or merchant co-op peer endorsement to lift trust score from {cur:.0f} to {target:.0f}."
    },
    "telecom_gap_days": {
        "label": "Prepaid Recharge Gap Days (Last 12m)",
        "unit": "days",
        "direction": "lower",
        "step": -1.0,
        "max_realistic_improvement": 6.0,
        "action_template": "Avoid prepaid SIM disconnection gaps; reduce un-recharged days from {cur:.0f} to {target:.0f} days."
    },
    "shock_recovery_days": {
        "label": "Cash-Flow Shock Recovery Speed (Days)",
        "unit": "days",
        "direction": "lower",
        "step": -2.0,
        "max_realistic_improvement": 8.0,
        "action_template": "Build a small emergency float to shorten wallet balance recovery time from {cur:.0f} to {target:.0f} days."
    }
}

PERSONAS = {
    "Ramesh (Rural Dairy Farmer — Prime Unbanked)": {
        "desc": "No bank credit card or CIBIL score, but 3+ years in a village dairy co-op with steady utility payments and strong SHG peer vouching.",
        "values": {
            "utility_on_time_pct": 91.0,
            "income_volatility_index": 0.24,
            "essential_spend_ratio": 0.78,
            "savings_retention_pct": 18.5,
            "livelihood_tenure_months": 38.0,
            "community_trust_score": 86.0,
            "telecom_gap_days": 1,
            "shock_recovery_days": 7.0
        },
        "requested_loan_inr": 35000
    },
    "Priya (Urban Delivery Gig Worker — Borderline / Near-Prime)": {
        "desc": "Active delivery partner for 14 months. High daily UPI inflows, but seasonal income swings and low month-end savings retention.",
        "values": {
            "utility_on_time_pct": 74.0,
            "income_volatility_index": 0.49,
            "essential_spend_ratio": 0.58,
            "savings_retention_pct": 7.5,
            "livelihood_tenure_months": 14.0,
            "community_trust_score": 66.0,
            "telecom_gap_days": 4,
            "shock_recovery_days": 15.0
        },
        "requested_loan_inr": 25000
    },
    "Suresh (Street Food Vendor — High Risk / Needs Recourse)": {
        "desc": "Cash-heavy vendor with frequent prepaid recharge gaps, high wallet volatility, and minimal emergency buffer. Ideal candidate for 30-Day AI Recourse.",
        "values": {
            "utility_on_time_pct": 54.0,
            "income_volatility_index": 0.72,
            "essential_spend_ratio": 0.44,
            "savings_retention_pct": 3.0,
            "livelihood_tenure_months": 9.0,
            "community_trust_score": 52.0,
            "telecom_gap_days": 9,
            "shock_recovery_days": 22.0
        },
        "requested_loan_inr": 20000
    }
}

@st.cache_resource
def load_engine():
    if not (os.path.exists("model_bundle.pkl") and os.path.exists("metrics_report.json")):
        with st.spinner("Initializing CrediTrust XAI Ensemble & Fairness Engine..."):
            run_training_pipeline()
    bundle = joblib.load("model_bundle.pkl")
    with open("metrics_report.json", "r") as f:
        report = json.load(f)
    return bundle, report

bundle, report = load_engine()
ensemble_model = bundle["ensemble_model"]
xgb_model = bundle["xgb_model"]
explainer = bundle["explainer"]
feature_cols = bundle["feature_cols"]
feature_bounds = bundle["feature_bounds"]

def compute_credit_assessment(input_dict, loan_amount=25000):
    X_row = pd.DataFrame([input_dict])[feature_cols]
    repay_prob = float(ensemble_model.predict_proba(X_row)[0, 1])

    # Map probability [0, 1] -> Alternative Credit Score [300, 850]
    alt_score = int(np.clip(round(300 + repay_prob * 550), 300, 850))

    if repay_prob >= 0.72:
        tier = "Tier 1: Prime Micro-Credit (Auto-Approved)"
        badge_class = "badge-pass"
        apr = round(11.5 + (1.0 - repay_prob) * 10.0, 2)
        approved = True
        max_limit = int(loan_amount * 1.4)
    elif repay_prob >= 0.50:
        tier = "Tier 2: Conditional Micro-Credit (SHG Co-Sign / Step-Up)"
        badge_class = "badge-warn"
        apr = round(15.5 + (0.72 - repay_prob) * 18.0, 2)
        approved = True
        max_limit = int(loan_amount * 1.0)
    else:
        tier = "Tier 3: High Risk — Enrolled in 30-Day AI Recourse Plan"
        badge_class = "badge-fail"
        apr = 24.0
        approved = False
        max_limit = int(loan_amount * 0.4)

    monthly_rate = (apr / 100.0) / 12.0
    emi_12m = int(round((loan_amount * monthly_rate * (1 + monthly_rate)**12) / ((1 + monthly_rate)**12 - 1)))

    # Compute exact local TreeSHAP values
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list):
        shap_row = np.array(shap_vals[1])[0]
    else:
        shap_row = np.array(shap_vals)[0]

    return {
        "repay_prob": repay_prob,
        "alt_score": alt_score,
        "tier": tier,
        "badge_class": badge_class,
        "apr": apr,
        "approved": approved,
        "max_limit": max_limit,
        "emi_12m": emi_12m,
        "shap_row": shap_row,
        "X_row": X_row
    }

def generate_counterfactual_recourse(input_dict, shap_row, target_prob=0.72):
    """Greedy counterfactual optimizer that finds actionable behavioral changes."""
    current = input_dict.copy()
    X_cur = pd.DataFrame([current])[feature_cols]
    cur_prob = float(ensemble_model.predict_proba(X_cur)[0, 1])

    # If already above 0.72, set target to 0.85 for prime APR reduction
    goal_prob = target_prob if cur_prob < target_prob else min(0.92, cur_prob + 0.10)

    # Rank features by lowest (most negative / weakest) SHAP contribution
    feature_order = np.argsort(shap_row)
    actions = []

    for idx in feature_order:
        if cur_prob >= goal_prob and len(actions) >= 2:
            break
        col = feature_cols[idx]
        # Skip tenure since a user cannot fast-forward time in 30 days
        if col == "livelihood_tenure_months":
            continue

        meta = FEATURE_META[col]
        f_min, f_max = feature_bounds[col]
        orig_val = current[col]
        best_val = orig_val

        # Step toward realistic improvement bound
        for _ in range(4):
            candidate = best_val + meta["step"]
            if meta["direction"] == "higher":
                candidate = min(f_max, min(orig_val + meta.get("max_realistic_improvement", 20.0), candidate))
            else:
                candidate = max(f_min, max(orig_val - meta.get("max_realistic_improvement", 6.0), candidate))
            best_val = candidate
            current[col] = best_val
            new_prob = float(ensemble_model.predict_proba(pd.DataFrame([current])[feature_cols])[0, 1])
            if new_prob >= goal_prob:
                break

        if abs(best_val - orig_val) > 1e-4:
            new_prob = float(ensemble_model.predict_proba(pd.DataFrame([current])[feature_cols])[0, 1])
            prob_gain = (new_prob - cur_prob) * 100.0
            if prob_gain > 0.4:
                actions.append({
                    "feature": meta["label"],
                    "instruction": meta["action_template"].format(cur=orig_val, target=best_val),
                    "prob_gain_pct": round(prob_gain, 1),
                    "new_score": int(np.clip(round(300 + new_prob * 550), 300, 850))
                })
                cur_prob = new_prob
            else:
                current[col] = orig_val

        if len(actions) >= 3:
            break

    projected_score = int(np.clip(round(300 + cur_prob * 550), 300, 850))
    return actions, cur_prob, projected_score

# Initialize session state with default persona
if "active_persona" not in st.session_state:
    st.session_state.active_persona = "Priya (Urban Delivery Gig Worker — Borderline / Near-Prime)"
    for k, v in PERSONAS[st.session_state.active_persona]["values"].items():
        st.session_state[k] = v
    st.session_state["requested_loan_inr"] = PERSONAS[st.session_state.active_persona]["requested_loan_inr"]

def load_persona(persona_name):
    st.session_state.active_persona = persona_name
    for k, v in PERSONAS[persona_name]["values"].items():
        st.session_state[k] = v
    st.session_state["requested_loan_inr"] = PERSONAS[persona_name]["requested_loan_inr"]

# Header Banner
st.markdown("## 🛡️ CrediTrust XAI — Alternative Credit Scoring & Recourse Engine")
st.caption(
    "**Neural Storm 2026 | Track 3 (HI-04):** Empowering unbanked gig workers, rural farmers, and street vendors "
    "with Calibrated Ensemble ML, Local TreeSHAP Attribution, and Counterfactual Financial Recourse."
)

# Quick Demo Persona Selector Bar
st.markdown("##### ⚡ Instant Judge Demo: Load an Unbanked Applicant Persona")
p_cols = st.columns(3)
for idx, p_name in enumerate(PERSONAS.keys()):
    with p_cols[idx]:
        is_selected = (st.session_state.active_persona == p_name)
        btn_type = "primary" if is_selected else "secondary"
        if st.button(p_name, key=f"btn_{idx}", use_container_width=True, type=btn_type):
            load_persona(p_name)
            st.rerun()

st.info(f"**Active Persona Context:** {PERSONAS[st.session_state.active_persona]['desc']}")

# Sidebar Sliders for Real-Time What-If Simulation
with st.sidebar:
    st.markdown("### 🎛️ Applicant Telemetry Inputs")
    st.caption("Adjust sliders to simulate real-time credit score, SHAP attribution, and interest rate shifts.")

    requested_loan = st.number_input(
        "Requested Micro-Loan Amount (₹ INR)",
        min_value=5000, max_value=200000, step=5000, key="requested_loan_inr"
    )
    st.divider()

    u_pct = st.slider(FEATURE_META["utility_on_time_pct"]["label"], 10.0, 100.0, step=1.0, key="utility_on_time_pct")
    inc_vol = st.slider(FEATURE_META["income_volatility_index"]["label"], 0.05, 1.20, step=0.01, key="income_volatility_index")
    ess_rat = st.slider(FEATURE_META["essential_spend_ratio"]["label"], 0.20, 0.95, step=0.01, key="essential_spend_ratio")
    sav_pct = st.slider(FEATURE_META["savings_retention_pct"]["label"], 0.0, 45.0, step=0.5, key="savings_retention_pct")
    ten_mo = st.slider(FEATURE_META["livelihood_tenure_months"]["label"], 1.0, 60.0, step=1.0, key="livelihood_tenure_months")
    com_scr = st.slider(FEATURE_META["community_trust_score"]["label"], 20.0, 100.0, step=1.0, key="community_trust_score")
    tel_gap = st.slider(FEATURE_META["telecom_gap_days"]["label"], 0, 25, step=1, key="telecom_gap_days")
    shk_rec = st.slider(FEATURE_META["shock_recovery_days"]["label"], 2.0, 35.0, step=0.5, key="shock_recovery_days")

current_inputs = {
    "utility_on_time_pct": u_pct,
    "income_volatility_index": inc_vol,
    "essential_spend_ratio": ess_rat,
    "savings_retention_pct": sav_pct,
    "livelihood_tenure_months": ten_mo,
    "community_trust_score": com_scr,
    "telecom_gap_days": tel_gap,
    "shock_recovery_days": shk_rec
}

assessment = compute_credit_assessment(current_inputs, requested_loan)

tab1, tab2, tab3 = st.tabs([
    "🔍 Live XAI Underwriting & Recourse",
    "📂 Batch Microfinance Portfolio Triage",
    "⚖️ Model Benchmarks & Fairness Audit"
])

with tab1:
    # Top KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Alternative Credit Score", f"{assessment['alt_score']} / 850", f"{assessment['repay_prob']*100:.1f}% Repay Prob")
    k2.metric("Risk-Adjusted Micro-APR", f"{assessment['apr']:.2f}%", "Fair-Cap Microfinance Rate")
    k3.metric("Estimated 12-Month EMI", f"₹{assessment['emi_12m']:,} / mo", f"On ₹{requested_loan:,} Principal")
    k4.metric("Recommended Credit Limit", f"₹{assessment['max_limit']:,}", "Dynamic Exposure Cap")

    st.markdown(
        f"**Underwriting Decision:** <span class='{assessment['badge_class']}'>{assessment['tier']}</span>",
        unsafe_allow_html=True
    )
    st.write("")

    col_left, col_right = st.columns([1.0, 1.35])

    with col_left:
        # Credit Score Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=assessment["alt_score"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "CrediTrust Alternative Score (300–850)", "font": {"size": 16}},
            gauge={
                "axis": {"range": [300, 850], "tickwidth": 1},
                "bar": {"color": "#0F766E"},
                "steps": [
                    {"range": [300, 575], "color": "#FEE2E2"},
                    {"range": [575, 695], "color": "#FEF9C3"},
                    {"range": [695, 850], "color": "#DCFCE7"}
                ],
                "threshold": {
                    "line": {"color": "#0F172A", "width": 4},
                    "thickness": 0.8,
                    "value": assessment["alt_score"]
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=45, b=15))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        # Local TreeSHAP Attribution Chart
        shap_df = pd.DataFrame({
            "Feature": [FEATURE_META[c]["label"] for c in feature_cols],
            "SHAP": assessment["shap_row"],
            "Value": [current_inputs[c] for c in feature_cols]
        }).sort_values("SHAP", ascending=True)

        colors = ["#DC2626" if val < 0 else "#0D9488" for val in shap_df["SHAP"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f} (val: {val})" for v, val in zip(shap_df["SHAP"], shap_df["Value"])],
            textposition="auto"
        ))
        fig_shap.update_layout(
            title="Local TreeSHAP Feature Attribution (Green = Boosts Score, Red = Increases Risk)",
            xaxis_title="SHAP Log-Odds Impact on Creditworthiness",
            yaxis_title="",
            height=300,
            margin=dict(l=10, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    st.divider()

    # Counterfactual Financial Recourse Engine
    st.markdown("### 🧭 Counterfactual Financial Recourse Plan (30-Day Path to Better Credit)")
    st.caption(
        "Unlike black-box models that reject unbanked applicants without explanation, our Counterfactual Optimizer "
        "calculates the smallest, most realistic behavioral steps to unlock approval or reduce interest rates."
    )

    actions, proj_prob, proj_score = generate_counterfactual_recourse(
        current_inputs, assessment["shap_row"]
    )

    r_col1, r_col2 = st.columns([2.2, 1.0])
    with r_col1:
        if actions:
            for i, act in enumerate(actions, 1):
                st.success(
                    f"**Step {i} ({act['feature']}):** {act['instruction']}  \n"
                    f"📈 *Impact:* **+{act['prob_gain_pct']}%** repayment probability → Score reaches **{act['new_score']}**"
                )
        else:
            st.success("🎉 Applicant is already in the top Prime Tier! Continue maintaining current payment and savings discipline.")

    with r_col2:
        score_delta = proj_score - assessment["alt_score"]
        st.metric(
            "Projected Score After 30-Day Plan",
            f"{proj_score} / 850",
            f"+{score_delta} pts ({proj_prob*100:.1f}% Repay Prob)"
        )
        new_apr = round(11.5 + (1.0 - proj_prob) * 10.0, 2) if proj_prob >= 0.72 else round(15.5 + (0.72 - proj_prob) * 18.0, 2)
        apr_saved = max(0.0, round(assessment["apr"] - new_apr, 2))
        st.metric("Projected Micro-Loan APR", f"{new_apr:.2f}%", f"-{apr_saved:.2f}% APR Reduction", delta_color="normal")

with tab2:
    st.markdown("### 📂 Batch Microfinance Cooperative / NGO Underwriting")
    st.caption(
        "Field officers can upload a CSV of unbanked applicants (or generate a synthetic village cohort below) "
        "to triage an entire Self-Help Group (SHG) portfolio in milliseconds."
    )

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        sample_size = st.slider("Synthetic Village Cohort Size", 25, 250, 50, step=25)
        generate_btn = st.button("⚡ Generate & Score Sample Village Cohort", type="primary", use_container_width=True)
    with b_col2:
        uploaded_csv = st.file_uploader("Or upload custom applicant CSV with the 8 telemetry columns", type=["csv"])

    if uploaded_csv is not None:
        batch_df = pd.read_csv(uploaded_csv)
    else:
        batch_df = generate_unbanked_dataset(sample_size).drop(columns=["repaid_on_time"])

    X_batch = batch_df[feature_cols]
    probs = ensemble_model.predict_proba(X_batch)[:, 1]
    batch_df["Alt_Credit_Score"] = np.clip(np.round(300 + probs * 550), 300, 850).astype(int)
    batch_df["Repay_Probability_%"] = np.round(probs * 100, 1)
    batch_df["Underwriting_Tier"] = np.where(
        probs >= 0.72, "Tier 1: Prime Approved",
        np.where(probs >= 0.50, "Tier 2: Conditional SHG", "Tier 3: AI Recourse Plan")
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Applicants Scored", len(batch_df))
    m2.metric("Auto-Approved (Tier 1 & 2)", f"{int((probs >= 0.50).sum())} ({((probs >= 0.50).mean()*100):.1f}%)")
    m3.metric("Cohort Average Credit Score", int(batch_df["Alt_Credit_Score"].mean()))

    st.dataframe(
        batch_df[["Alt_Credit_Score", "Repay_Probability_%", "Underwriting_Tier"] + feature_cols],
        use_container_width=True,
        height=340
    )

    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Underwritten Portfolio CSV",
        data=csv_bytes,
        file_name="creditrust_underwritten_cohort.csv",
        mime="text/csv"
    )

with tab3:
    st.markdown("### ⚖️ Model Architecture Benchmarks & Algorithmic Fairness Audit")
    st.caption(
        "Evaluated on a stratified 20% holdout test set (1,600 profiles out of 8,000 total). "
        "Audited for Disparate Impact & Equal Opportunity across Rural vs. Urban cohorts."
    )

    bench_df = pd.DataFrame(report["benchmarks"]).T.reset_index()
    bench_df.columns = ["Model Architecture", "Accuracy", "ROC-AUC", "F1-Score", "Precision", "Recall", "Brier Loss (Calibration)"]
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("#### 🛡️ Algorithmic Fairness Audit (Rural vs. Urban)")
        fa = report["fairness_audit"]
        f1, f2, f3 = st.columns(3)
        f1.metric("Disparate Impact Ratio", f"{fa['disparate_impact_ratio']:.3f}", "Target: 0.80 – 1.25")
        f2.metric("Equal Opportunity Diff", f"{fa['equal_opportunity_diff']:.3f}", "Target: < 0.05")
        f3.metric("80% Rule Compliance", "PASSED ✅" if fa["four_fifths_rule_passed"] else "REVIEW ⚠️")

        fig_fair = go.Figure(data=[
            go.Bar(
                name="Approval Rate",
                x=["Rural Applicants", "Urban Applicants"],
                y=[fa["rural_approval_rate"] * 100, fa["urban_approval_rate"] * 100],
                marker_color=["#0F766E", "#0284C7"],
                text=[f"{fa['rural_approval_rate']*100:.1f}%", f"{fa['urban_approval_rate']*100:.1f}%"],
                textposition="auto"
            )
        ])
        fig_fair.update_layout(
            title="Demographic Approval Parity (Rural vs. Urban Holdout Test Set)",
            yaxis_title="Approval Rate (%)",
            yaxis_range=[0, 100],
            height=290,
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_fair, use_container_width=True)

    with f_col2:
        st.markdown("#### 🎯 Holdout Confusion Matrix (Stacking Ensemble)")
        cm = np.array(report["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Default (0)", "Predicted Repay (1)"],
            y=["Actual Default (0)", "Actual Repay (1)"],
            colorscale="Teal",
            text=[[f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"], [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"]],
            texttemplate="%{text}",
            textfont={"size": 15}
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix (n = {report['test_size']} Test Profiles)",
            height=290,
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)