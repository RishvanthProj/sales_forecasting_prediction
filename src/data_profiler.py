"""
Data Profiler Module.
Produces comprehensive dataset profiling metrics, missingness reports, outlier summaries,
and column categorization analysis.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def generate_profile(
    df: pd.DataFrame,
    date_col: Optional[str] = "Date",
    sales_col: Optional[str] = "Sales",
    order_col: Optional[str] = "Order",
    audit_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Generate a complete dataset profile dictionary containing structural, temporal,
    missingness, and outlier statistics.
    """
    total_rows = len(df)
    total_cols = df.shape[1]
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    duplicate_rows = int(df.duplicated().sum())

    # Date profiling
    date_info: Dict[str, Any] = {"available": False}
    if date_col and date_col in df.columns:
        try:
            date_series = pd.to_datetime(df[date_col].dropna())
            if not date_series.empty:
                min_date = date_series.min()
                max_date = date_series.max()
                unique_dates = date_series.nunique()
                total_span_days = (max_date - min_date).days + 1
                date_info = {
                    "available": True,
                    "min_date": str(min_date.date()),
                    "max_date": str(max_date.date()),
                    "unique_dates": unique_dates,
                    "span_days": total_span_days,
                }
        except Exception:
            pass

    # Missing values summary
    missing_summary = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct_missing = (n_missing / max(total_rows, 1)) * 100
        missing_summary.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Missing Count": n_missing,
            "Missing %": round(pct_missing, 2),
            "Unique Values": int(df[col].nunique(dropna=False)),
        })
    missing_df = pd.DataFrame(missing_summary).sort_values("Missing %", ascending=False)

    # Numerical statistics & IQR Outlier Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 4:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            n_outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
            pct_outliers = (n_outliers / len(series)) * 100

            outlier_summary.append({
                "Column": col,
                "Mean": round(float(series.mean()), 2),
                "Std": round(float(series.std()), 2),
                "Min": round(float(series.min()), 2),
                "Median": round(float(series.median()), 2),
                "Max": round(float(series.max()), 2),
                "IQR": round(iqr, 2),
                "Potential Outliers": n_outliers,
                "Outlier %": round(pct_outliers, 2),
            })
    outlier_df = pd.DataFrame(outlier_summary)

    # Categorical breakdown
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_summary = []
    for col in cat_cols:
        if col != date_col:
            cat_summary.append({
                "Column": col,
                "Unique Classes": int(df[col].nunique()),
                "Top Value": str(df[col].mode().iloc[0]) if not df[col].empty else "N/A",
                "Top Frequency": int(df[col].value_counts().iloc[0]) if not df[col].empty else 0,
            })
    cat_df = pd.DataFrame(cat_summary)

    # Excluded columns from audit report
    excluded_df = pd.DataFrame()
    if audit_df is not None and not audit_df.empty:
        excluded_df = audit_df[audit_df["Decision"] == "Excluded"].copy()

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "memory_mb": round(memory_mb, 2),
        "duplicate_rows": duplicate_rows,
        "date_info": date_info,
        "missing_df": missing_df,
        "outlier_df": outlier_df,
        "cat_df": cat_df,
        "excluded_df": excluded_df,
    }
