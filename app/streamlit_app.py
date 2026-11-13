import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.preprocessing import clean_data

# Risk bands on the estimated default probability (average default rate is ~6.7%)
LOW_MAX = 0.05
MEDIUM_MAX = 0.15

st.set_page_config(page_title="Loan Default Predictor", layout="centered")


@st.cache_resource
def load_assets():
    model = joblib.load(ROOT / "models" / "loan_default_model.joblib")
    prep = model.named_steps["prep"]
    clf = model.named_steps["model"]
    names = [n.replace("num__", "").replace("missingindicator_", "missing: ")
             for n in prep.get_feature_names_out()]

    def predict_proba(X):
        return clf.predict_proba(X)[:, 1]

    # Scaled features are centred at 0, so a zero row is the "average borrower" reference
    explainer = shap.Explainer(predict_proba, np.zeros((1, len(names))), feature_names=names)
    return model, prep, explainer, names


# NEW: a slider and a number box that stay in sync (drag for speed, type for precision)
def synced_input(label, min_v, max_v, default, step, key):
    sld, num = f"{key}_sld", f"{key}_num"
    if sld not in st.session_state:
        st.session_state[sld] = default
        st.session_state[num] = default

    def snap(v):
        return float(min(max(round(round(v / step) * step, 4), min_v), max_v))

    def on_slider():
        st.session_state[num] = st.session_state[sld]

    def on_number():
        v = snap(st.session_state[num])
        st.session_state[sld] = v
        st.session_state[num] = v

    st.slider(label, min_v, max_v, step=step, key=sld, on_change=on_slider)
    st.number_input("Exact value", min_v, max_v, step=step, format="%.2f",
                    key=num, on_change=on_number)
    return st.session_state[sld]


model, prep, explainer, feature_names = load_assets()

st.title("Loan Default Predictor")
st.caption("Estimates the probability that a borrower becomes 90+ days late within two years. "
           "Educational project, not for real lending decisions.")
st.caption("Input ranges are limited to where the training data is dense; "
           "predictions outside them would be unreliable.")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 90, 45)
    income = st.number_input("Monthly income", 500, 50_000, 5_000, step=500)
    # CHANGED: the two sliders now come with a number box for exact values
    utilization = synced_input("Credit card utilization (balance / limit)",
                               0.0, 1.5, 0.30, 0.01, "util")
    debt_ratio = synced_input("Debt ratio (monthly debt / income)",
                              0.0, 5.0, 0.35, 0.05, "debt")
    dependents = st.number_input("Dependents", 0, 10, 0)
    
with col2:
    open_lines = st.number_input("Open credit lines and loans", 0, 40, 8)
    real_estate = st.number_input("Real estate loans", 0, 10, 1)
    late_30 = st.number_input("Times 30-59 days late", 0, 20, 0)
    late_60 = st.number_input("Times 60-89 days late", 0, 20, 0)
    late_90 = st.number_input("Times 90+ days late", 0, 20, 0)

row = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": utilization,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": income,
    "NumberOfOpenCreditLinesAndLoans": open_lines,
    "NumberOfTimes90DaysLate": late_90,
    "NumberRealEstateLoansOrLines": real_estate,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60,
    "NumberOfDependents": dependents,
}]).astype(float)
row = row[list(model.feature_names_in_)]

if st.button("Predict risk", type="primary"):
    row = clean_data(row)
    score = float(model.predict_proba(row)[0, 1])

    if score < LOW_MAX:
        band = "Low"
    elif score < MEDIUM_MAX:
        band = "Medium"
    else:
        band = "High"

    st.metric("Estimated default probability", f"{score:.1%}")
    st.subheader(f"Risk level: {band}")
    st.caption(
        "Average default rate in the training data is about 6.7%. "
        f"Risk levels: Low < {LOW_MAX:.0%}, Medium {LOW_MAX:.0%}-{MEDIUM_MAX:.0%}, "
        f"High > {MEDIUM_MAX:.0%}."
    )

    st.subheader("What drove this prediction")
    Xt = prep.transform(row)
    sv = explainer(Xt)

    # Show the borrower's real values in the chart labels instead of scaled ones.
    # Map by feature name, not by position: the last columns are the
    # missing-value indicators, which are 1 only when the matching input is
    # missing. (In the app every input is always filled in, so they are 0
    # here, but deriving them from the row keeps this correct if the code is
    # ever reused with incomplete rows.)
    raw = row.iloc[0]
    display = np.zeros(Xt.shape[1])
    for i, name in enumerate(feature_names):
        if name.startswith("missing: "):
            display[i] = 1.0 if pd.isna(raw[name.replace("missing: ", "")]) else 0.0
        elif name in raw.index:
            display[i] = raw[name]
    expl = shap.Explanation(
        values=sv.values[0],
        base_values=float(np.ravel(sv.base_values)[0]),
        data=display,
        feature_names=sv.feature_names,
    )
    shap.plots.waterfall(expl, max_display=8, show=False)
    st.caption("The chart starts from about 2.7%, the prediction for a borrower with "
               "all-average features, and each bar shows how a feature moves the estimate.")
    st.pyplot(plt.gcf())
    plt.close()