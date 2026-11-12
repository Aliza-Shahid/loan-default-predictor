"""Tests for the saved model: it loads, is the calibrated (unweighted)
model, and its predictions move in the right direction."""
import joblib
import pandas as pd
import pytest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "loan_default_model.joblib"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


def _borrower(**overrides):
    row = {
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row]).astype(float)


def test_model_is_a_pipeline_with_prep_and_model(model):
    assert list(model.named_steps.keys()) == ["prep", "model"]


def test_final_model_is_unweighted(model):
    # The calibrated model must not use class weights (see README: the
    # weighted model scored Brier 0.143, the unweighted one 0.049).
    assert model.named_steps["model"].class_weight is None


def test_safe_borrower_scores_low(model):
    prob = float(model.predict_proba(_borrower())[0, 1])
    assert prob < 0.05


def test_risky_borrower_scores_high(model):
    prob = float(model.predict_proba(_borrower(
        NumberOfTimes90DaysLate=5,
        RevolvingUtilizationOfUnsecuredLines=1.4,
        MonthlyIncome=1500.0,
    ))[0, 1])
    assert prob > 0.5


def test_higher_utilization_raises_risk(model):
    low = float(model.predict_proba(_borrower(
        RevolvingUtilizationOfUnsecuredLines=0.1))[0, 1])
    high = float(model.predict_proba(_borrower(
        RevolvingUtilizationOfUnsecuredLines=1.4))[0, 1])
    assert high > low
