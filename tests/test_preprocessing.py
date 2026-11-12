"""Tests for the data cleaning, split and preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import build_preprocessor, clean_data, split_data, TARGET


def _raw_row(**overrides):
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
        TARGET: 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_placeholder_codes_become_missing():
    df = clean_data(_raw_row(**{"NumberOfTime30-59DaysPastDueNotWorse": 96,
                                "NumberOfTimes90DaysLate": 98}))
    assert pd.isna(df.loc[0, "NumberOfTime30-59DaysPastDueNotWorse"])
    assert pd.isna(df.loc[0, "NumberOfTimes90DaysLate"])


def test_impossible_age_becomes_missing():
    assert pd.isna(clean_data(_raw_row(age=12)).loc[0, "age"])
    assert clean_data(_raw_row(age=45)).loc[0, "age"] == 45


def test_debt_ratio_missing_when_income_missing():
    df = clean_data(_raw_row(MonthlyIncome=np.nan, DebtRatio=1234.0))
    assert pd.isna(df.loc[0, "DebtRatio"])
    df2 = clean_data(_raw_row(MonthlyIncome=5000.0, DebtRatio=0.35))
    assert df2.loc[0, "DebtRatio"] == 0.35


def test_extreme_values_are_capped():
    df = clean_data(_raw_row(RevolvingUtilizationOfUnsecuredLines=9.0,
                             DebtRatio=50.0, MonthlyIncome=200_000.0))
    assert df.loc[0, "RevolvingUtilizationOfUnsecuredLines"] == 2
    assert df.loc[0, "DebtRatio"] == 10
    assert df.loc[0, "MonthlyIncome"] == 50_000


def test_split_is_stratified():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(20, 70, 1000),
        "MonthlyIncome": rng.integers(1000, 9000, 1000).astype(float),
        TARGET: [0] * 930 + [1] * 70,
    })
    _, _, y_train, y_test = split_data(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.02
    assert abs(y_train.mean() - 0.07) < 0.01


def test_preprocessor_output_has_no_nan_and_adds_missing_flags():
    df = pd.DataFrame({
        "age": [30, 50, 65, 40],
        "MonthlyIncome": [3000.0, np.nan, 8000.0, 4500.0],
        "NumberOfDependents": [1, 0, np.nan, 2],
        "DebtRatio": [0.2, 0.5, 0.1, 0.9],
    })
    out = build_preprocessor().fit_transform(df)
    assert not np.isnan(out).any()
    # 4 features + 2 missing-value indicators (income, dependents)
    assert out.shape[1] == 6
