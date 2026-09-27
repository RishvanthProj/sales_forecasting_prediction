"""
Model Training Module.
Implements chronological time-series splitting, baseline forecasts (Naive, Ridge),
and tree-based regressors (Random Forest, XGBoost) directly on the sales target.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split time series chronologically into Train, Validation, and Test sets.
    Preserves exact temporal sequence with zero random shuffling or data leakage.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    unique_dates = df_sorted[date_col].drop_duplicates().sort_values().values
    n_dates = len(unique_dates)

    if n_dates < 10:
        n_rows = len(df_sorted)
        train_end = int(n_rows * train_pct)
        val_end = int(n_rows * (train_pct + val_pct))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
    else:
        train_idx = int(n_dates * train_pct)
        val_idx = int(n_dates * (train_pct + val_pct))

        train_cut = unique_dates[train_idx]
        val_cut = unique_dates[val_idx]

        train_df = df_sorted[df_sorted[date_col] < train_cut].copy()
        val_df = df_sorted[(df_sorted[date_col] >= train_cut) & (df_sorted[date_col] < val_cut)].copy()
        test_df = df_sorted[df_sorted[date_col] >= val_cut].copy()

    split_info = {
        "train_start": str(train_df[date_col].min())[:10] if not train_df.empty else "N/A",
        "train_end": str(train_df[date_col].max())[:10] if not train_df.empty else "N/A",
        "train_rows": len(train_df),
        "val_start": str(val_df[date_col].min())[:10] if not val_df.empty else "N/A",
        "val_end": str(val_df[date_col].max())[:10] if not val_df.empty else "N/A",
        "val_rows": len(val_df),
        "test_start": str(test_df[date_col].min())[:10] if not test_df.empty else "N/A",
        "test_end": str(test_df[date_col].max())[:10] if not test_df.empty else "N/A",
        "test_rows": len(test_df),
        "total_rows": len(df_sorted),
    }

    return train_df, val_df, test_df, split_info


class NaiveForecaster:
    """Naive baseline predicting the most recent historical observation."""
    def __init__(self, lag_feature: str = "lag_1_Sales"):
        self.lag_feature = lag_feature

    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.lag_feature in X.columns:
            return np.maximum(0.0, X[self.lag_feature].fillna(0.0).values)
        return np.zeros(len(X))


class ModelWrapper:
    """Standardized wrapper executing predictions and ensuring non-negative sales."""
    def __init__(self, model, feature_cols: List[str]):
        self.model = model
        self.feature_cols = feature_cols

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X_mat = X[self.feature_cols].fillna(0.0)
        self.model.fit(X_mat, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = X[self.feature_cols].fillna(0.0)
        preds = self.model.predict(X_mat)
        return np.maximum(0.0, preds)


def build_regressor(model_name: str, random_state: int = 42):
    """Instantiate ML regressor with sensible hyperparameter defaults."""
    if model_name == "Ridge Baseline":
        return Ridge(alpha=1.0)
    elif model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
        return XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        # Fallback to Random Forest if XGBoost not present
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )


def train_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    sales_col: str = "Sales",
    models_to_train: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train specified models chronologically and calculate validation & test predictions.
    """
    if models_to_train is None:
        models_to_train = ["Naive Baseline", "Ridge Baseline", "Random Forest"]
        if XGBOOST_AVAILABLE:
            models_to_train.append("XGBoost")

    trained_models: Dict[str, Any] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    training_times: Dict[str, float] = {}

    y_train = train_df[sales_col].values
    y_val = val_df[sales_col].values
    y_test = test_df[sales_col].values

    lag_cands = [c for c in train_df.columns if c.startswith("lag_") and sales_col in c]
    lag1_col = lag_cands[0] if lag_cands else "lag_1_Sales"

    for name in models_to_train:
        t0 = time.time()

        if name == "Naive Baseline":
            forecaster = NaiveForecaster(lag_feature=lag1_col)
            val_p = forecaster.predict(val_df)
            test_p = forecaster.predict(test_df)

            trained_models[name] = forecaster
            predictions[name] = {"val_sales": val_p, "test_sales": test_p}
            training_times[name] = round(time.time() - t0, 3)
            continue

        raw_model = build_regressor(name)
        wrapper = ModelWrapper(raw_model, feature_cols)
        wrapper.fit(train_df, y_train)

        val_p = wrapper.predict(val_df)
        test_p = wrapper.predict(test_df)

        trained_models[name] = wrapper
        predictions[name] = {"val_sales": val_p, "test_sales": test_p}
        training_times[name] = round(time.time() - t0, 3)

    return {
        "models": trained_models,
        "predictions": predictions,
        "training_times": training_times,
        "feature_cols": feature_cols,
        "actuals": {
            "val_sales": y_val,
            "test_sales": y_test,
        },
    }
