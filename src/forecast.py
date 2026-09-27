"""
Recursive Multi-Step Forecasting Engine.
Executes iterative future forecasting without data leakage by updating
historical lag buffers with generated step-by-step predictions at dataset scope.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from src.features import FeaturePipeline


def get_horizon_for_period(period_name: str, frequency: str = "Daily") -> int:
    """
    Convert simple user period selection ('1 Month', '3 Months', '6 Months')
    into the precise number of forecasting steps based on dataset frequency.
    """
    norm_p = str(period_name).strip().lower()
    norm_f = str(frequency).strip().lower()

    if "1" in norm_p:  # 1 Month
        if "daily" in norm_f:
            return 30
        elif "weekly" in norm_f:
            return 4
        elif "monthly" in norm_f:
            return 1
        else:  # irregular or fallback
            return 4
    elif "3" in norm_p:  # 3 Months
        if "daily" in norm_f:
            return 90
        elif "weekly" in norm_f:
            return 13
        elif "monthly" in norm_f:
            return 3
        else:
            return 13
    elif "6" in norm_p:  # 6 Months
        if "daily" in norm_f:
            return 180
        elif "weekly" in norm_f:
            return 26
        elif "monthly" in norm_f:
            return 6
        else:
            return 26
    else:
        # Default fallback
        return 4 if "weekly" in norm_f else (30 if "daily" in norm_f else 6)


def generate_recursive_forecast(
    historical_df: pd.DataFrame,
    pipeline: FeaturePipeline,
    model: Any,
    horizon: int = 7,
    frequency: str = "Daily",
    scenario_inputs: Optional[Dict[str, Any]] = None,
    **kwargs,  # Gracefully ignore any legacy entity_col/entity_val kwargs
) -> pd.DataFrame:
    """
    Generate multi-step recursive forecasts for a given horizon at dataset scope.
    Iteratively predicts period t, updates the historical lag buffer with the predicted value,
    recalculates required lag and rolling features, and advances to t+1.
    Guarantees mathematical continuity and scale consistency with the historical target.
    """
    date_col = pipeline.date_col
    sales_col = pipeline.sales_col
    order_col = pipeline.order_col

    working_hist = historical_df.copy()
    if working_hist.empty or date_col not in working_hist.columns or sales_col not in working_hist.columns:
        return pd.DataFrame()

    working_hist[date_col] = pd.to_datetime(working_hist[date_col])
    working_hist = working_hist.sort_values(date_col).reset_index(drop=True)

    # Ensure dataset-scope: if multiple records exist per date, aggregate them by date
    if working_hist[date_col].duplicated().any():
        agg_rules = {sales_col: "sum"}
        if order_col and order_col in working_hist.columns:
            agg_rules[order_col] = "sum"
        num_cols = working_hist.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if c not in [sales_col, order_col]:
                agg_rules[c] = "mean"
        working_hist = working_hist.groupby(date_col).agg(agg_rules).reset_index().sort_values(date_col).reset_index(drop=True)

    last_date = working_hist[date_col].max()

    # Determine future date sequence based on detected frequency
    freq_norm = str(frequency).strip().lower()
    if "weekly" in freq_norm:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=horizon, freq="W")
    elif "monthly" in freq_norm:
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")
    else:  # Daily or default
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Retain the minimal tail needed to compute maximum lags & rolling windows
    max_lag = max(pipeline.lags) if pipeline.lags else 1
    max_roll = max(pipeline.rolling_windows) if pipeline.rolling_windows else 7
    max_history_needed = max(max_lag, max_roll) + 30
    buffer_df = working_hist.tail(max_history_needed).copy().reset_index(drop=True)

    last_row = working_hist.iloc[-1].to_dict()
    forecast_records = []

    for step_date in future_dates:
        new_row = last_row.copy()
        new_row[date_col] = step_date
        new_row[sales_col] = np.nan
        if order_col and order_col in new_row:
            new_row[order_col] = np.nan

        # Apply user scenario inputs if provided
        if scenario_inputs:
            for k, v in scenario_inputs.items():
                if k in new_row:
                    new_row[k] = v

        step_df = pd.DataFrame([new_row])
        buffer_df = pd.concat([buffer_df, step_df], ignore_index=True)

        # Run feature pipeline over the expanded buffer to populate lags & shifted rolling features
        transformed_buffer = pipeline.transform(buffer_df)
        current_step_row = transformed_buffer.iloc[[-1]]

        # Generate prediction
        if hasattr(model, "predict"):
            pred_s_arr = model.predict(current_step_row)
            pred_s_val = float(pred_s_arr[0]) if hasattr(pred_s_arr, "__len__") else float(pred_s_arr)
        else:
            pred_s_val = 0.0

        pred_s_val = max(0.0, pred_s_val)

        # Update buffer target with newly predicted value so subsequent steps can use it as a lag!
        buffer_df.loc[buffer_df.index[-1], sales_col] = pred_s_val

        record = {
            "Date": step_date.strftime("%Y-%m-%d"),
            "Forecasted_Sales": round(pred_s_val, 2),
        }
        forecast_records.append(record)

    forecast_df = pd.DataFrame(forecast_records)

    # Rigorous Output Validation (Section 13 & 37)
    if not forecast_df.empty:
        preds = forecast_df["Forecasted_Sales"].values
        # 1. No NaNs or infinities
        if np.isnan(preds).any() or np.isinf(preds).any():
            raise ValueError("Model produced invalid NaN or Inf forecast values.")
        # 2. Correct number of prediction points
        if len(forecast_df) != horizon:
            raise ValueError(f"Forecast length mismatch: expected {horizon}, got {len(forecast_df)}.")
        # 3. All non-negative
        if (preds < 0).any():
            forecast_df["Forecasted_Sales"] = forecast_df["Forecasted_Sales"].clip(lower=0.0)

    return forecast_df


def compute_forecast_summary(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    sales_col: str = "Sales",
    date_col: str = "Date",
) -> Dict[str, Any]:
    """
    Calculate comprehensive forecast summary metrics and comparative trajectory indicators.
    """
    if forecast_df.empty or "Forecasted_Sales" not in forecast_df.columns:
        return {
            "total_forecast": 0.0,
            "avg_forecast": 0.0,
            "peak_period": "N/A",
            "peak_val": 0.0,
            "low_period": "N/A",
            "low_val": 0.0,
            "trend": "Stable",
            "change_vs_recent_pct": 0.0,
            "recent_actual_total": 0.0,
        }

    preds = forecast_df["Forecasted_Sales"].values
    total_forecast = float(np.sum(preds))
    avg_forecast = float(np.mean(preds))

    peak_idx = int(np.argmax(preds))
    low_idx = int(np.argmin(preds))

    peak_period = str(forecast_df["Date"].iloc[peak_idx])
    peak_val = float(preds[peak_idx])

    low_period = str(forecast_df["Date"].iloc[low_idx])
    low_val = float(preds[low_idx])

    # Compare against most recent historical period of same length
    horizon = len(preds)
    change_vs_recent_pct = 0.0
    recent_actual_total = 0.0

    if sales_col in historical_df.columns and date_col in historical_df.columns:
        hist_agg = historical_df.groupby(date_col)[sales_col].sum().sort_index()
        if len(hist_agg) >= horizon:
            recent_actual_total = float(hist_agg.tail(horizon).sum())
            if recent_actual_total > 1e-5:
                change_vs_recent_pct = ((total_forecast - recent_actual_total) / recent_actual_total) * 100.0

    # Trend categorization aligned with percentage change vs recent actuals
    if recent_actual_total > 1e-5:
        if change_vs_recent_pct >= 2.0:
            trend = "Increasing"
        elif change_vs_recent_pct <= -2.0:
            trend = "Decreasing"
        else:
            trend = "Stable"
    else:
        # Fallback to internal trajectory slope if historical comparison is unavailable
        if len(preds) >= 2:
            x = np.arange(len(preds))
            slope, _ = np.polyfit(x, preds, 1)
            rel_slope = (slope / max(avg_forecast, 1e-5)) * 100.0
            if rel_slope > 1.0:
                trend = "Increasing"
            elif rel_slope < -1.0:
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Stable"

    return {
        "total_forecast": round(total_forecast, 2),
        "avg_forecast": round(avg_forecast, 2),
        "peak_period": peak_period,
        "peak_val": round(peak_val, 2),
        "low_period": low_period,
        "low_val": round(low_val, 2),
        "trend": trend,
        "change_vs_recent_pct": round(change_vs_recent_pct, 2),
        "recent_actual_total": round(recent_actual_total, 2),
    }
