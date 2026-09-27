"""
Model Hyperparameter Tuning Module.
Integrates Optuna optimization using TimeSeriesSplit cross-validation to search
optimal parameters for XGBoost, Random Forest, and Ridge.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def tune_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Sales",
    model_name: str = "XGBoost",
    n_trials: int = 15,
    n_splits: int = 3,
) -> Dict[str, Any]:
    """
    Tune model hyperparameters using Optuna and TimeSeriesSplit cross-validation.
    Returns best parameters, optimization history, and validation score.
    """
    if not OPTUNA_AVAILABLE:
        return {
            "status": "error",
            "message": "Optuna library is not installed.",
            "best_params": {},
            "best_score": None,
            "trials_df": pd.DataFrame(),
        }

    X = df[feature_cols].fillna(0.0).values
    y = df[target_col].values

    if len(X) < 30:
        return {
            "status": "error",
            "message": "Dataset too small for multi-fold TimeSeriesSplit tuning.",
            "best_params": {},
            "best_score": None,
            "trials_df": pd.DataFrame(),
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
            }
            model_cls = XGBRegressor
        elif model_name == "Random Forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
                "max_depth": trial.suggest_int("max_depth", 4, 14),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
                "random_state": 42,
                "n_jobs": -1,
            }
            model_cls = RandomForestRegressor
        elif model_name == "Ridge Baseline":
            params = {
                "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
            }
            model_cls = Ridge
        else:
            params = {"alpha": 1.0}
            model_cls = Ridge

        fold_errors = []
        for train_idx, val_idx in tscv.split(X):
            X_train_f, X_val_f = X[train_idx], X[val_idx]
            y_train_f, y_val_f = y[train_idx], y[val_idx]

            m = model_cls(**params)
            m.fit(X_train_f, y_train_f)
            preds = m.predict(X_val_f)

            # WAPE objective: sum(|y - y_hat|) / sum(y)
            denom = np.sum(np.abs(y_val_f))
            if denom > 1e-5:
                wape = np.sum(np.abs(y_val_f - preds)) / denom
            else:
                wape = mean_absolute_error(y_val_f, preds)
            fold_errors.append(wape)

        return float(np.mean(fold_errors))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Compile trial history dataframe
    records = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rec = {"Trial": t.number, "CV WAPE Score": round(t.value, 4)}
            rec.update(t.params)
            records.append(rec)

    trials_df = pd.DataFrame(records).sort_values("CV WAPE Score") if records else pd.DataFrame()

    return {
        "status": "success",
        "best_params": study.best_params,
        "best_score": round(study.best_value, 4),
        "trials_df": trials_df,
    }
