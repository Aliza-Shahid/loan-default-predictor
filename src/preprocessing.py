"""Data cleaning, train/test split, and preprocessing pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "SeriousDlqin2yrs"
LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "cs-training.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise fixes using fixed rules only (no statistics learned from data)."""
    df = df.copy()

    # 96 and 98 are placeholder codes, not real counts
    for col in LATE_COLS:
        df[col] = df[col].replace([96, 98], np.nan)

    # Impossible ages
    df.loc[df["age"] < 18, "age"] = np.nan

    # When income is missing, DebtRatio holds a raw amount, not a ratio
    df.loc[df["MonthlyIncome"].isna(), "DebtRatio"] = np.nan

    # Cap extreme values at fixed thresholds
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=2)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=10)
    df["MonthlyIncome"] = df["MonthlyIncome"].clip(upper=50_000)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified split so both sets keep the same ~6.7% default rate."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def build_preprocessor() -> ColumnTransformer:
    """Median imputation (with missing-value flags) plus scaling.
    Fitted only on training data when used inside a Pipeline."""
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric, make_numeric_selector())],
        remainder="drop",
    )


def make_numeric_selector():
    from sklearn.compose import make_column_selector

    return make_column_selector(dtype_include=np.number)


if __name__ == "__main__":
    data = clean_data(load_data())
    X_train, X_test, y_train, y_test = split_data(data)
    print("Train:", X_train.shape, "default rate:", round(y_train.mean(), 4))
    print("Test: ", X_test.shape, "default rate:", round(y_test.mean(), 4))
    Xt = build_preprocessor().fit_transform(X_train)
    print("Processed train shape:", Xt.shape)
    print("Any NaN left:", np.isnan(Xt).any())