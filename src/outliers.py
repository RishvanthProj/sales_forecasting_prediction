"""
Outlier Analysis Module.
Implements robust IQR-based anomaly detection, threshold calculations,
and extraction of unusual observations with retail-calibrated domain explanations.
Provides coordinates for highlighting outliers on sales-over-time charts.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def analyze_outliers(
    df: pd.DataFrame,
    sales_col: str = "Sales",
    date_col: str = "Date",
    store_col: Optional[str] = None,
    product_col: Optional[str] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Perform rigorous IQR-based outlier analysis on the sales target.
    Calculates Q1, Q3, IQR, Lower/Upper bounds, outlier count, outlier percentage,
    and returns top outliers sorted by largest absolute deviation from normal range.
    """
    if sales_col not in df.columns or df[sales_col].dropna().empty:
        return {
            "has_outliers": False,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "extreme_upper": 0.0,
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "top_outliers_df": pd.DataFrame(),
            "all_outliers_df": pd.DataFrame(),
        }

    s_series = pd.to_numeric(df[sales_col], errors="coerce").dropna()
    total_count = len(s_series)

    q1 = float(s_series.quantile(0.25))
    median = float(s_series.median())
    q3 = float(s_series.quantile(0.75))
    iqr = q3 - q1

    lower_bound = max(0.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    extreme_upper = q3 + 3.0 * iqr

    outlier_mask = (df[sales_col] < lower_bound) | (df[sales_col] > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_pct = round((outlier_count / max(total_count, 1)) * 100, 2)

    # Contextual display columns
    display_cols = []
    if date_col in df.columns:
        display_cols.append(date_col)
    if store_col and store_col in df.columns:
        display_cols.append(store_col)
    if product_col and product_col in df.columns:
        display_cols.append(product_col)
    display_cols.append(sales_col)

    for ctx_col in ["Holiday", "Holiday_Flag", "Discount", "Temperature", "CPI", "Unemployment"]:
        if ctx_col in df.columns and ctx_col not in display_cols:
            display_cols.append(ctx_col)

    outliers_subset = df[outlier_mask].copy()
    if not outliers_subset.empty:
        # Calculate deviation from normal range
        high_mask = outliers_subset[sales_col] > upper_bound
        deviations = np.where(
            high_mask,
            outliers_subset[sales_col] - upper_bound,
            lower_bound - outliers_subset[sales_col]
        )
        outliers_subset["Deviation_From_Normal"] = np.round(deviations, 2)
        outliers_subset["Outlier_Type"] = np.where(
            outliers_subset[sales_col] > extreme_upper,
            "Extreme Spike (>3x IQR)",
            np.where(high_mask, "Moderate Spike (>1.5x IQR)", "Low Demand Drop")
        )
        outliers_subset["Domain_Interpretation"] = np.where(
            high_mask,
            "Seasonal surge / holiday demand spike / promotional peak",
            "Off-peak operational drop / irregular reporting",
        )

        # Sort by largest absolute deviation
        sorted_outliers = outliers_subset.sort_values(by="Deviation_From_Normal", ascending=False)
        cols_final = display_cols + ["Deviation_From_Normal", "Outlier_Type", "Domain_Interpretation"]
        cols_present = [c for c in cols_final if c in sorted_outliers.columns]
        top_outliers_df = sorted_outliers[cols_present].head(top_n).reset_index(drop=True)
        all_outliers_df = sorted_outliers[[c for c in [date_col, sales_col, "Deviation_From_Normal", "Outlier_Type"] if c in sorted_outliers.columns]].copy()
    else:
        top_outliers_df = pd.DataFrame()
        all_outliers_df = pd.DataFrame()

    return {
        "has_outliers": outlier_count > 0,
        "total_records": total_count,
        "q1": round(q1, 2),
        "median": round(median, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "extreme_upper": round(extreme_upper, 2),
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
        "top_outliers_df": top_outliers_df,
        "all_outliers_df": all_outliers_df,
    }
