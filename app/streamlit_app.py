import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    return model, prep, explainer


model, prep, explainer = load_assets()

st.title("Loan Default Predictor")
st.caption("Estimates the probability that a borrower becomes 90+ days late within two years. "
           "Educational project, not for real lending decisions.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 45)
    income = st.number_input("Monthly income", 0, 50_000, 5_000, step=500)
    utilization = st.slider("Credit card utilization (balance / limit)", 0.0, 2.0, 0.30, 0.01)
    debt_ratio = st.slider("Debt ratio (monthly debt / income)", 0.0, 10.0, 0.35, 0.05)
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
    sv = explainer(prep.transform(row))
    shap.plots.waterfall(sv[0], max_display=8, show=False)
    st.pyplot(plt.gcf())
    plt.close()