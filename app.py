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

# Clean, self-explanatory UI CSS (Prevents metric truncation like "345 / ...")
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px; }
    .how-it-works {
        background: #F0FDFA;
        border: 1px solid #CCFBF1;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 16px;
        color: #134E4A;
        font-size: 0.92rem;
    }
    .profile-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #0F766E;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }
    .kpi-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .kpi-title { font-size: 0.80rem; color: #64748B; font-weight: 600; text-transform: uppercase; }
    .kpi-val { font-size: 1.55rem; color: #0F172A; font-weight: 800; margin: 4px 0; }
    .kpi-sub { font-size: 0.80rem; color: #0F766E; font-weight: 500; }
    .decision-banner-pass {
        background: #DCFCE7; border: 1px solid #86EFAC; color: #14532D;
        padding: 14px 18px; border-radius: 10px; font-weight: 700; font-size: 1.05rem; margin-bottom: 16px;
    }
    .decision-banner-warn {
        background: #FEF9C3; border: 1px solid #FDE047; color: #713F12;
        padding: 14px 18px; border-radius: 10px; font-weight: 700; font-size: 1.05rem; margin-bottom: 16px;
    }
    .decision-banner-fail {
        background: #FEE2E2; border: 1px solid #FCA5A5; color: #7F1D1D;
        padding: 14px 18px; border-radius: 10px; font-weight: 700; font-size: 1.05rem; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_META = {
    "utility_on_time_pct": {
        "label": "Electricity & Phone Bill Timeliness (%)",
        "short_label": "On-Time Utility Bills (%)",
        "help": "Percentage of electricity, water, and mobile bills paid before the due date over the last 12 months.",
        "good_text": "Pays electricity & phone bills on time ({val:.0f}%)",
        "bad_text": "Late electricity/utility bill payments ({val:.0f}% on-time)",
        "direction": "higher",
        "step": 5.0,
        "max_realistic_improvement": 22.0,
        "action_template": "Pay electricity & mobile bills before the due date to boost your on-time rate from {cur:.0f}% to {target:.0f}%."
    },
    "income_volatility_index": {
        "label": "Daily Income Fluctuation (Lower = Steadier)",
        "short_label": "UPI Income Volatility",
        "help": "Measures how much daily/weekly mobile wallet earnings fluctuate. Lower number means steadier income.",
        "good_text": "Steady, predictable mobile wallet / UPI earnings ({val:.2f})",
        "bad_text": "Irregular, fluctuating cash-flow in mobile wallet ({val:.2f})",
        "direction": "lower",
        "step": -0.05,
        "max_realistic_improvement": 0.25,
        "action_template": "Deposit daily cash sales into your UPI wallet consistently to steady your income fluctuation from {cur:.2f} down to {target:.2f}."
    },
    "essential_spend_ratio": {
        "label": "Business & Household Essential Spend Ratio",
        "short_label": "Essential Spend Ratio",
        "help": "Share of wallet outflow spent on inventory, groceries, fuel, and school fees vs. non-essentials.",
        "good_text": "Disciplined spending on business & family essentials ({val:.0%})",
        "bad_text": "Low tracked ratio of essential business/household spending ({val:.0%})",
        "direction": "higher",
        "step": 0.05,
        "max_realistic_improvement": 0.20,
        "action_template": "Pay for business inventory & household essentials digitally via UPI to raise essential spend ratio from {cur:.0%} to {target:.0%}."
    },
    "savings_retention_pct": {
        "label": "Month-End Money Saved (%)",
        "short_label": "Monthly Savings Kept (%)",
        "help": "Percentage of monthly income left over as savings at the end of each month.",
        "good_text": "Keeps a healthy month-end savings buffer ({val:.1f}%)",
        "bad_text": "Very little money left in savings at month-end ({val:.1f}%)",
        "direction": "higher",
        "step": 3.0,
        "max_realistic_improvement": 12.0,
        "action_template": "Keep an automated month-end micro-savings balance of {target:.1f}% of income (up from {cur:.1f}%)."
    },
    "livelihood_tenure_months": {
        "label": "Months Working in Current Trade / App",
        "short_label": "Work / Platform Tenure (Mo)",
        "help": "How many months the applicant has been active in their dairy co-op, street stall, or delivery app.",
        "good_text": "Long, verified work stability ({val:.0f} months active)",
        "bad_text": "Short verified work tenure ({val:.0f} months active)",
        "direction": "higher",
        "step": 3.0,
        "max_realistic_improvement": 6.0,
        "action_template": "Maintain continuous active platform/vendor registration for {target:.0f} months."
    },
    "community_trust_score": {
        "label": "Village Co-op / Self-Help Group Trust (0-100)",
        "short_label": "Community / SHG Trust Score",
        "help": "Endorsement score from local Women's Self-Help Group (SHG), dairy cooperative, or merchant association.",
        "good_text": "Strong peer vouching from local Co-op / Self-Help Group ({val:.0f}/100)",
        "bad_text": "Low community / Self-Help Group peer endorsement ({val:.0f}/100)",
        "direction": "higher",
        "step": 5.0,
        "max_realistic_improvement": 15.0,
        "action_template": "Complete local Self-Help Group (SHG) or merchant co-op verification to lift peer trust from {cur:.0f} to {target:.0f}."
    },
    "telecom_gap_days": {
        "label": "Days Phone Went Un-Recharged (Last 12m)",
        "short_label": "SIM Recharge Gap Days",
        "help": "Number of days the applicant's prepaid SIM expired without recharge—an indicator of severe cash crunches.",
        "good_text": "Phone SIM stays continuously active ({val:.0f} gap days)",
        "bad_text": "Prepaid SIM went un-recharged for {val:.0f} days during cash crunches",
        "direction": "lower",
        "step": -1.0,
        "max_realistic_improvement": 6.0,
        "action_template": "Avoid prepaid SIM disconnection gaps; reduce un-recharged days from {cur:.0f} down to {target:.0f} days."
    },
    "shock_recovery_days": {
        "label": "Days to Recover Wallet Balance After Expense",
        "short_label": "Emergency Recovery Speed (Days)",
        "help": "Average days needed to rebuild mobile wallet balance after a sudden medical or repair expense.",
        "good_text": "Recovers cash balance quickly after expenses ({val:.0f} days)",
        "bad_text": "Slow recovery after sudden cash expenses ({val:.0f} days)",
        "direction": "lower",
        "step": -2.0,
        "max_realistic_improvement": 8.0,
        "action_template": "Keep a small emergency cash float to shorten wallet recovery time from {cur:.0f} down to {target:.0f} days."
    }
}

PERSONAS = {
    "🌾 Ramesh (Rural Dairy Farmer — Approved)": {
        "name": "Ramesh Kumar",
        "role": "Smallholder Dairy Farmer (Rural Co-op Member)",
        "loan_purpose": "Wants ₹35,000 micro-loan to purchase a second milch buffalo & automated fodder chaff cutter.",
        "story": "Ramesh has NO bank credit card or CIBIL score. Traditional banks auto-reject him. However, he has paid village electricity bills on time for 3+ years and has strong dairy co-op peer trust.",
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
    "🛵 Priya (Urban Delivery Partner — Conditional)": {
        "name": "Priya Sharma",
        "role": "App-Based Food & Grocery Delivery Partner",
        "loan_purpose": "Wants ₹25,000 micro-loan to upgrade from a petrol scooter to an Electric Vehicle (EV) two-wheeler.",
        "story": "Priya earns daily payouts via UPI and has worked for 22 months. Her income dips slightly during monsoon months, making her a borderline applicant who qualifies for Step-Up credit.",
        "values": {
            "utility_on_time_pct": 80.0,
            "income_volatility_index": 0.33,
            "essential_spend_ratio": 0.66,
            "savings_retention_pct": 12.0,
            "livelihood_tenure_months": 22.0,
            "community_trust_score": 72.0,
            "telecom_gap_days": 2,
            "shock_recovery_days": 11.0
        },
        "requested_loan_inr": 25000
    },
    "🍲 Suresh (Street Food Vendor — Needs 30-Day Coaching)": {
        "name": "Suresh Thapa",
        "role": "Roadside Momo & Tea Stall Vendor",
        "loan_purpose": "Wants ₹20,000 working-capital loan to buy commercial steamer equipment and bulk ingredients.",
        "story": "Suresh deals mostly in cash, forgets utility bill due dates, and had 9 days where his phone SIM went un-recharged. Instead of a dead-end rejection, CrediTrust gives him a 30-day coaching plan to get approved!",
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
        with st.spinner("Training CrediTrust XAI Ensemble & Fairness Engine (takes ~4 seconds)..."):
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
    alt_score = int(np.clip(round(300 + repay_prob * 550), 300, 850))

    if repay_prob >= 0.72:
        tier = "✅ LOAN AUTO-APPROVED (Tier 1: Prime Micro-Credit)"
        banner_class = "decision-banner-pass"
        plain_verdict = "This applicant shows strong real-world payment discipline and qualifies for instant low-interest micro-credit without needing a traditional bank score."
        apr = round(11.5 + (1.0 - repay_prob) * 10.0, 2)
        max_limit = int(loan_amount * 1.4)
    elif repay_prob >= 0.50:
        tier = "🟡 CONDITIONALLY APPROVED (Tier 2: Co-op / SHG Step-Up Loan)"
        banner_class = "decision-banner-warn"
        plain_verdict = "This applicant shows promising digital cash-flow and qualifies for a starter micro-loan with Self-Help Group (SHG) peer co-signing."
        apr = round(15.5 + (0.72 - repay_prob) * 18.0, 2)
        max_limit = int(loan_amount * 1.0)
    else:
        tier = "⏳ NOT APPROVED YET — Enrolled in 30-Day AI Credit Coaching Plan"
        banner_class = "decision-banner-fail"
        plain_verdict = "Current cash-flow risk is too high for immediate disbursement. Follow the 3-step AI Coaching Plan below for 30 days to unlock approval!"
        apr = 24.0
        max_limit = int(loan_amount * 0.4)

    monthly_rate = (apr / 100.0) / 12.0
    emi_12m = int(round((loan_amount * monthly_rate * (1 + monthly_rate)**12) / ((1 + monthly_rate)**12 - 1)))

    shap_vals = explainer.shap_values(X_row)
    shap_row = np.array(shap_vals[1])[0] if isinstance(shap_vals, list) else np.array(shap_vals)[0]

    return {
        "repay_prob": repay_prob,
        "alt_score": alt_score,
        "tier": tier,
        "banner_class": banner_class,
        "plain_verdict": plain_verdict,
        "apr": apr,
        "max_limit": max_limit,
        "emi_12m": emi_12m,
        "shap_row": shap_row
    }

def generate_counterfactual_recourse(input_dict, shap_row, target_prob=0.72):
    current = input_dict.copy()
    X_cur = pd.DataFrame([current])[feature_cols]
    cur_prob = float(ensemble_model.predict_proba(X_cur)[0, 1])
    goal_prob = target_prob if cur_prob < target_prob else min(0.92, cur_prob + 0.10)

    feature_order = np.argsort(shap_row)
    actions = []

    for idx in feature_order:
        if cur_prob >= goal_prob and len(actions) >= 2:
            break
        col = feature_cols[idx]
        if col == "livelihood_tenure_months":
            continue

        meta = FEATURE_META[col]
        f_min, f_max = feature_bounds[col]
        orig_val = current[col]
        best_val = orig_val

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
                    "feature": meta["short_label"],
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

# Default session state
default_persona = "🌾 Ramesh (Rural Dairy Farmer — Approved)"
if "active_persona" not in st.session_state or st.session_state.active_persona not in PERSONAS:
    st.session_state.active_persona = default_persona
    for k, v in PERSONAS[default_persona]["values"].items():
        st.session_state[k] = v
    st.session_state["requested_loan_inr"] = PERSONAS[default_persona]["requested_loan_inr"]

def load_persona(persona_name):
    st.session_state.active_persona = persona_name
    for k, v in PERSONAS[persona_name]["values"].items():
        st.session_state[k] = v
    st.session_state["requested_loan_inr"] = PERSONAS[persona_name]["requested_loan_inr"]

# Top Title & 5-Second Plain-English Guide
st.markdown("## 🛡️ CrediTrust XAI — Fair Micro-Loan Underwriting for the Unbanked")
st.markdown("""
<div class="how-it-works">
    <b>💡 What does this app do?</b> 1.4 billion rural farmers, street vendors, and gig workers have <b>no bank credit card history</b>, so banks reject them. 
    <b>CrediTrust XAI</b> uses 8 everyday signals (like electricity bill payments & mobile wallet steadiness) to approve fair micro-loans—and gives rejected borrowers a <b>30-Day Step-by-Step Coaching Plan</b> to become eligible!
</div>
""", unsafe_allow_html=True)

# Step 1: Persona Picker
st.markdown("#### 👇 Step 1: Click a Real-World Unbanked Applicant to Test")
p_cols = st.columns(3)
for idx, p_name in enumerate(PERSONAS.keys()):
    with p_cols[idx]:
        is_selected = (st.session_state.active_persona == p_name)
        if st.button(p_name, key=f"btn_{idx}", use_container_width=True, type="primary" if is_selected else "secondary"):
            load_persona(p_name)
            st.rerun()

p_info = PERSONAS[st.session_state.active_persona]

# Sidebar with clear human labels & tooltips
with st.sidebar:
    st.markdown("### 🎛️ Adjust Applicant Data")
    st.caption("Move any slider below to see how changing financial habits updates the credit score and loan decision live!")

    requested_loan = st.number_input(
        "Requested Loan Amount (₹ INR)",
        min_value=5000, max_value=200000, step=5000, key="requested_loan_inr"
    )
    st.divider()

    u_pct = st.slider(FEATURE_META["utility_on_time_pct"]["label"], 10.0, 100.0, step=1.0, key="utility_on_time_pct", help=FEATURE_META["utility_on_time_pct"]["help"])
    inc_vol = st.slider(FEATURE_META["income_volatility_index"]["label"], 0.05, 1.20, step=0.01, key="income_volatility_index", help=FEATURE_META["income_volatility_index"]["help"])
    ess_rat = st.slider(FEATURE_META["essential_spend_ratio"]["label"], 0.20, 0.95, step=0.01, key="essential_spend_ratio", help=FEATURE_META["essential_spend_ratio"]["help"])
    sav_pct = st.slider(FEATURE_META["savings_retention_pct"]["label"], 0.0, 45.0, step=0.5, key="savings_retention_pct", help=FEATURE_META["savings_retention_pct"]["help"])
    ten_mo = st.slider(FEATURE_META["livelihood_tenure_months"]["label"], 1.0, 60.0, step=1.0, key="livelihood_tenure_months", help=FEATURE_META["livelihood_tenure_months"]["help"])
    com_scr = st.slider(FEATURE_META["community_trust_score"]["label"], 20.0, 100.0, step=1.0, key="community_trust_score", help=FEATURE_META["community_trust_score"]["help"])
    tel_gap = st.slider(FEATURE_META["telecom_gap_days"]["label"], 0, 25, step=1, key="telecom_gap_days", help=FEATURE_META["telecom_gap_days"]["help"])
    shk_rec = st.slider(FEATURE_META["shock_recovery_days"]["label"], 2.0, 35.0, step=0.5, key="shock_recovery_days", help=FEATURE_META["shock_recovery_days"]["help"])

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
    "🔍 1. Loan Decision & 30-Day Coaching Plan",
    "📂 2. Village / NGO Batch Loan Triage",
    "⚖️ 3. AI Accuracy & Fairness Proof"
])

with tab1:
    # Applicant Story Banner
    st.markdown(f"""
    <div class="profile-banner">
        <div style="font-size:1.1rem; font-weight:700; color:#0F172A;">👤 Applicant: {p_info['name']} — <span style="color:#0F766E;">{p_info['role']}</span></div>
        <div style="margin-top:4px; font-size:0.92rem; color:#334155;"><b>🎯 Loan Goal:</b> {p_info['loan_purpose']}</div>
        <div style="margin-top:4px; font-size:0.88rem; color:#64748B;"><b>📋 Background:</b> {p_info['story']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Clear Decision Banner
    st.markdown(f"""
    <div class="{assessment['banner_class']}">
        {assessment['tier']}
        <div style="font-weight:400; font-size:0.9rem; margin-top:4px;">{assessment['plain_verdict']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Non-truncating HTML KPI Grid (Never cuts off numbers with "...")
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-box">
            <div class="kpi-title">Alternative Credit Score</div>
            <div class="kpi-val">{assessment['alt_score']} <span style="font-size:1rem;color:#64748B;">/ 850</span></div>
            <div class="kpi-sub">{assessment['repay_prob']*100:.1f}% On-Time Repayment Likelihood</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Recommended Interest (APR)</div>
            <div class="kpi-val">{assessment['apr']:.2f}% <span style="font-size:0.95rem;color:#64748B;">/ yr</span></div>
            <div class="kpi-sub">vs. 60%–120% Informal Moneylenders</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Monthly Installment (12-Mo EMI)</div>
            <div class="kpi-val">₹{assessment['emi_12m']:,} <span style="font-size:0.95rem;color:#64748B;">/ mo</span></div>
            <div class="kpi-sub">For ₹{requested_loan:,} Requested Loan</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">Safe Maximum Loan Limit</div>
            <div class="kpi-val">₹{assessment['max_limit']:,}</div>
            <div class="kpi-sub">AI Risk-Adjusted Exposure Cap</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plain-English Summary of Why (Top Helping vs Hurting Factors)
    shap_pairs = list(zip(feature_cols, assessment["shap_row"]))
    pos_factors = sorted([p for p in shap_pairs if p[1] > 0], key=lambda x: x[1], reverse=True)[:3]
    neg_factors = sorted([p for p in shap_pairs if p[1] < 0], key=lambda x: x[1])[:3]

    why_col1, why_col2 = st.columns(2)
    with why_col1:
        st.markdown("##### 👍 What is Helping This Applicant's Score")
        if pos_factors:
            for col_name, s_val in pos_factors:
                txt = FEATURE_META[col_name]["good_text"].format(val=current_inputs[col_name])
                st.markdown(f"- ✅ **{txt}** *(+{s_val:.2f} AI impact)*")
        else:
            st.caption("No strong positive indicators yet—see the 30-Day Coaching Plan below.")

    with why_col2:
        st.markdown("##### ⚠️ What is Holding Back This Applicant's Score")
        if neg_factors:
            for col_name, s_val in neg_factors:
                txt = FEATURE_META[col_name]["bad_text"].format(val=current_inputs[col_name])
                st.markdown(f"- 🔻 **{txt}** *({s_val:.2f} AI impact)*")
        else:
            st.markdown("- 🎉 **No major risk flags detected!** All behavioral signals are strong.")

    # Visual Charts Row
    col_left, col_right = st.columns([1.0, 1.35])
    with col_left:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=assessment["alt_score"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Alternative Credit Score (300–850)", "font": {"size": 15}},
            gauge={
                "axis": {"range": [300, 850], "tickwidth": 1},
                "bar": {"color": "#0F766E"},
                "steps": [
                    {"range": [300, 575], "color": "#FEE2E2"},
                    {"range": [575, 695], "color": "#FEF9C3"},
                    {"range": [695, 850], "color": "#DCFCE7"}
                ],
                "threshold": {"line": {"color": "#0F172A", "width": 4}, "thickness": 0.8, "value": assessment["alt_score"]}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_right:
        shap_df = pd.DataFrame({
            "Feature": [FEATURE_META[c]["short_label"] for c in feature_cols],
            "SHAP": assessment["shap_row"]
        }).sort_values("SHAP", ascending=True)

        colors = ["#DC2626" if val < 0 else "#0D9488" for val in shap_df["SHAP"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in shap_df["SHAP"]],
            textposition="auto"
        ))
        fig_shap.update_layout(
            title="Explainable AI (TreeSHAP): Green Bars Boost Score | Red Bars Lower Score",
            xaxis_title="Impact on Loan Approval",
            yaxis_title="",
            height=280,
            margin=dict(l=10, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    st.divider()

    # Step 3: Counterfactual Financial Recourse Plan
    st.markdown("### 🧭 Step 3: Automated 30-Day Credit Coaching Plan (Counterfactual AI)")
    st.caption(
        "Instead of rejecting unbanked workers without guidance, our AI calculates the smallest, realistic steps "
        "this applicant can take over the next 30 days to unlock loan approval or lower their interest rate."
    )

    actions, proj_prob, proj_score = generate_counterfactual_recourse(current_inputs, assessment["shap_row"])

    r_col1, r_col2 = st.columns([2.1, 1.0])
    with r_col1:
        if actions:
            for i, act in enumerate(actions, 1):
                st.success(
                    f"**Action Step {i} ({act['feature']}):** {act['instruction']}  \n"
                    f"📈 *Result:* Boosts approval odds by **+{act['prob_gain_pct']}%** → Credit Score rises to **{act['new_score']}**"
                )
        else:
            st.success("🎉 **Already in Top Prime Tier!** This applicant already qualifies for our lowest microfinance interest rate.")

    with r_col2:
        score_delta = proj_score - assessment["alt_score"]
        new_apr = round(11.5 + (1.0 - proj_prob) * 10.0, 2) if proj_prob >= 0.72 else round(15.5 + (0.72 - proj_prob) * 18.0, 2)
        apr_saved = max(0.0, round(assessment["apr"] - new_apr, 2))
        st.markdown(f"""
        <div class="kpi-box" style="background:#F0FDFA; border-color:#99F6E4;">
            <div class="kpi-title">Score After 30-Day Plan</div>
            <div class="kpi-val" style="color:#0F766E;">{proj_score} / 850 <span style="font-size:1rem;">(+{score_delta} pts)</span></div>
            <div class="kpi-sub">New Interest Rate: <b>{new_apr:.2f}% APR</b> (Saves {apr_saved:.2f}% interest)</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📂 Batch Village / NGO Microfinance Portfolio Triage")
    st.caption(
        "A microfinance field officer visiting a village can upload a spreadsheet of 50+ unbanked citizens "
        "to see who is ready for an immediate loan and who needs the 30-Day Coaching Plan."
    )

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        sample_size = st.slider("Sample Village Cohort Size", 25, 250, 50, step=25)
        st.button("⚡ Score Sample Village Cohort", type="primary", use_container_width=True)
    with b_col2:
        uploaded_csv = st.file_uploader("Or upload a custom CSV of applicants", type=["csv"])

    batch_df = pd.read_csv(uploaded_csv) if uploaded_csv is not None else generate_unbanked_dataset(sample_size).drop(columns=["repaid_on_time"])
    X_batch = batch_df[feature_cols]
    probs = ensemble_model.predict_proba(X_batch)[:, 1]
    batch_df["Alt_Credit_Score"] = np.clip(np.round(300 + probs * 550), 300, 850).astype(int)
    batch_df["Repay_Probability_%"] = np.round(probs * 100, 1)
    batch_df["Loan_Decision"] = np.where(
        probs >= 0.72, "✅ Tier 1: Auto-Approved",
        np.where(probs >= 0.50, "🟡 Tier 2: SHG Co-Sign", "⏳ Tier 3: 30-Day Coaching")
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Citizens Evaluated", len(batch_df))
    m2.metric("Approved for Micro-Loans (Tier 1 & 2)", f"{int((probs >= 0.50).sum())} Citizens ({((probs >= 0.50).mean()*100):.1f}%)")
    m3.metric("Village Average Credit Score", f"{int(batch_df['Alt_Credit_Score'].mean())} / 850")

    st.dataframe(
        batch_df[["Loan_Decision", "Alt_Credit_Score", "Repay_Probability_%"] + feature_cols],
        use_container_width=True,
        height=340
    )

    st.download_button(
        "⬇️ Download Scored Village Portfolio (CSV)",
        data=batch_df.to_csv(index=False).encode("utf-8"),
        file_name="creditrust_village_portfolio.csv",
        mime="text/csv"
    )

with tab3:
    st.markdown("### ⚖️ Model Accuracy Benchmarks & Rural vs. Urban Fairness Proof")
    st.caption(
        "Tested on 1,600 holdout profiles. Proves to regulators that our Stacking Ensemble outperforms single models "
        "and treats Rural and Urban citizens fairly (passing the 80% Disparate Impact Rule)."
    )

    bench_df = pd.DataFrame(report["benchmarks"]).T.reset_index()
    bench_df.columns = ["AI Model Architecture", "Accuracy", "ROC-AUC Score", "F1-Score", "Precision", "Recall", "Calibration Loss"]
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown("#### 🛡️ Demographic Fairness Audit (Rural vs. Urban)")
        fa = report["fairness_audit"]
        f1, f2, f3 = st.columns(3)
        f1.metric("Disparate Impact Ratio", f"{fa['disparate_impact_ratio']:.3f}", "Fair Range: 0.80–1.25")
        f2.metric("Equal Opportunity Gap", f"{fa['equal_opportunity_diff']:.3f}", "Target: < 0.05")
        f3.metric("EEOC 80% Fairness Rule", "PASSED ✅" if fa["four_fifths_rule_passed"] else "REVIEW ⚠️")

        fig_fair = go.Figure(data=[
            go.Bar(
                x=["Rural Citizens", "Urban Citizens"],
                y=[fa["rural_approval_rate"] * 100, fa["urban_approval_rate"] * 100],
                marker_color=["#0F766E", "#0284C7"],
                text=[f"{fa['rural_approval_rate']*100:.1f}% Approved", f"{fa['urban_approval_rate']*100:.1f}% Approved"],
                textposition="auto"
            )
        ])
        fig_fair.update_layout(
            title="Loan Approval Parity: Rural vs. Urban Citizens",
            yaxis_title="Approval Rate (%)",
            yaxis_range=[0, 100],
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_fair, use_container_width=True)

    with f_col2:
        st.markdown("#### 🎯 Holdout Test Set Confusion Matrix (1,600 Profiles)")
        cm = np.array(report["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["AI Predicted Default", "AI Predicted Repay"],
            y=["Actually Defaulted", "Actually Repaid"],
            colorscale="Teal",
            text=[[f"True Reject: {cm[0,0]}", f"False Approve: {cm[0,1]}"], [f"False Reject: {cm[1,0]}", f"True Approve: {cm[1,1]}"]],
            texttemplate="%{text}",
            textfont={"size": 14}
        ))
        fig_cm.update_layout(
            title=f"Accuracy Breakdown on {report['test_size']} Unseen Applicants",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
