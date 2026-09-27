"""
Model Validation & Diagnostic Evaluation Module.
Computes regression metrics (MAE, RMSE, WAPE, R²), WAPE-based Forecast Accuracy,
produces comparative leaderboards, extracts tree feature importances,
and prepares actual-vs-predicted residual diagnostics.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate full suite of forecasting regression metrics.
    Computes WAPE and WAPE-based Forecast Accuracy = max(0, (1 - WAPE) * 100).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    valid = ~(np.isnan(y_t) | np.isnan(y_p))
    y_t = y_t[valid]
    y_p = y_p[valid]

    if len(y_t) == 0:
        return {
            "MAE": 0.0,
            "RMSE": 0.0,
            "MSE": 0.0,
            "WAPE_%": 0.0,
            "WAPE_Accuracy_%": 0.0,
            "Safe_MAPE_%": 0.0,
            "R2": 0.0,
        }

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))

    # WAPE: sum(|y - y_hat|) / sum(|y|)
    sum_actual = float(np.sum(np.abs(y_t)))
    wape_ratio = (float(np.sum(np.abs(y_t - y_p))) / sum_actual) if sum_actual > 1e-5 else 0.0
    wape_pct = wape_ratio * 100.0

    # WAPE-based Forecast Accuracy = (1 - WAPE) * 100
    wape_accuracy = max(0.0, (1.0 - wape_ratio) * 100.0)

    # Safe MAPE: handles actual zeros safely by bounding denominator
    safe_denom = np.maximum(np.abs(y_t), 1.0)
    safe_mape = float(np.mean(np.abs(y_t - y_p) / safe_denom) * 100.0)

    # R2 Score (coefficient of determination)
    r2 = float(r2_score(y_t, y_p)) if len(y_t) > 1 and np.var(y_t) > 1e-7 else 0.0

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MSE": round(mse, 2),
        "WAPE_%": round(wape_pct, 2),
        "WAPE_Accuracy_%": round(wape_accuracy, 2),
        "Safe_MAPE_%": round(safe_mape, 2),
        "R2": round(r2, 4),
    }


def build_comparison_table(
    training_results: Dict[str, Any],
    split: str = "test",
) -> pd.DataFrame:
    """
    Build a model comparison leaderboard table comparing all trained models on validation or test split.
    Ranked by highest WAPE-based Forecast Accuracy (lowest WAPE).
    """
    predictions = training_results.get("predictions", {})
    training_times = training_results.get("training_times", {})
    actuals = training_results.get("actuals", {})

    y_actual = actuals.get(f"{split}_sales")
    if y_actual is None:
        return pd.DataFrame()

    records = []
    for model_name, preds_dict in predictions.items():
        pred_sales = preds_dict.get(f"{split}_sales")
        if pred_sales is not None:
            m = calculate_metrics(y_actual, pred_sales)
            records.append({
                "Model": model_name,
                "WAPE-based Forecast Accuracy (%)": m["WAPE_Accuracy_%"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "WAPE (%)": m["WAPE_%"],
                "R² Score": m["R2"],
                "Train Time (s)": training_times.get(model_name, 0.0),
            })

    comp_df = pd.DataFrame(records)
    if not comp_df.empty:
        comp_df = comp_df.sort_values(by="WAPE-based Forecast Accuracy (%)", ascending=False).reset_index(drop=True)
    return comp_df


def generate_diagnostics(
    test_df: pd.DataFrame,
    date_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Construct aligned diagnostics dataframe with date, actual, predicted, residuals,
    and absolute error for test period evaluation.
    """
    n = min(len(test_df), len(y_true), len(y_pred))
    dates = pd.to_datetime(test_df[date_col].iloc[:n]).dt.strftime("%Y-%m-%d").values

    y_t = np.asarray(y_true[:n], dtype=float)
    y_p = np.asarray(y_pred[:n], dtype=float)

    residuals = y_t - y_p
    abs_errors = np.abs(residuals)
    pct_errors = np.abs(residuals) / np.maximum(y_t, 1.0) * 100.0

    diag_df = pd.DataFrame({
        "Date": dates,
        "Actual_Sales": np.round(y_t, 2),
        "Predicted_Sales": np.round(y_p, 2),
        "Residual": np.round(residuals, 2),
        "Absolute_Error": np.round(abs_errors, 2),
        "Percentage_Error": np.round(pct_errors, 2),
    })

    return diag_df


def extract_feature_importance(
    model_obj: Any,
    feature_cols: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importances from fitted tree regressors (XGBoost / Random Forest)
    or standardized coefficients from Ridge.
    """
    underlying = getattr(model_obj, "model", model_obj)

    importances = None
    if hasattr(underlying, "feature_importances_"):
        importances = underlying.feature_importances_
    elif hasattr(underlying, "coef_"):
        importances = np.abs(underlying.coef_)

    if importances is None or len(importances) != len(feature_cols):
        return pd.DataFrame()

    total = np.sum(importances)
    norm_importances = (importances / total * 100.0) if total > 1e-7 else importances

    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance_%": np.round(norm_importances, 2),
    }).sort_values(by="Importance_%", ascending=False).head(top_n).reset_index(drop=True)

    return fi_df


def get_human_metric_summary(metrics: Dict[str, float], currency_symbol: str = "$") -> List[Dict[str, str]]:
    """
    Produce plain-language, non-misleading interpretations of regression metrics.
    Explicitly clarifies R² vs 'accuracy'.
    """
    mae = metrics.get("MAE", 0.0)
    rmse = metrics.get("RMSE", 0.0)
    wape = metrics.get("WAPE_%", 0.0)
    acc = metrics.get("WAPE_Accuracy_%", max(0.0, 100.0 - wape))
    r2 = metrics.get("R2", 0.0)
    r2_pct = max(0.0, min(100.0, r2 * 100.0))

    return [
        {
            "metric": "WAPE-based Forecast Accuracy",
            "value": f"{acc:.2f}%",
            "interpretation": f"Derived as (1 - WAPE) × 100 on unseen chronological test data. WAPE is {wape:.2f}%.",
        },
        {
            "metric": "MAE (Mean Absolute Error)",
            "value": f"{currency_symbol}{mae:,.2f}",
            "interpretation": f"Average absolute error in sales units per observation across holdout test data.",
        },
        {
            "metric": "RMSE (Root Mean Squared Error)",
            "value": f"{currency_symbol}{rmse:,.2f}",
            "interpretation": f"Penalizes larger estimation mistakes more heavily ({currency_symbol}{rmse:,.2f}).",
        },
        {
            "metric": "R² Score (Model Performance)",
            "value": f"{r2:.4f}",
            "interpretation": f"The model explains approximately {r2_pct:.1f}% of the observed variation in holdout test sales.",
        },
    ]


def get_accuracy_calculation_explanation() -> str:
    """
    Return clean, visible, markdown-formatted explanation of how forecast accuracy is calculated.
    """
    return """
### How is Forecast Accuracy Calculated?

Because sales forecasting is a continuous **regression problem**, standard classification accuracy (e.g. correct labels / total labels) does not apply. Instead, we use industry-standard **WAPE (Weighted Absolute Percentage Error)** on unseen holdout test data:

1. **Chronological Hold-Out Split:** The dataset is strictly split in time (e.g. 70% train, 15% validation, 15% unseen test). Earlier data trains the model, while the latest period tests generalization.
2. **Out-of-Sample Predictions:** The trained model predicts the unseen test period without having seen its target sales.
3. **Absolute Errors:** For each observation $t$, the absolute error is calculated:
   $$\\text{Error}_t = |\\text{Actual}_t - \\text{Predicted}_t|$$
4. **WAPE Calculation:** The sum of absolute errors is divided by the sum of actual historical sales:
   $$\\text{WAPE} = \\frac{\\sum |\\text{Actual}_t - \\text{Predicted}_t|}{\\sum \\text{Actual}_t}$$
5. **Forecast Accuracy Percentage:** Forecast accuracy is directly derived from WAPE:
   $$\\text{Forecast Accuracy} = (1 - \\text{WAPE}) \\times 100$$
   *(Bounded at 0% minimum).*

#### Complementary Regression Metrics:
- **MAE (Mean Absolute Error):** The average absolute error in sales currency units. Easy to interpret as the expected dollar deviation.
- **RMSE (Root Mean Squared Error):** Gives higher weight to large prediction mistakes by squaring deviations before taking the square root.
- **R² Score (Coefficient of Determination):** Measures the proportion of variance in actual sales explained by model predictions ($0.0$ to $1.0$).
"""
