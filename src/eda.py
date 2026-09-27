"""
Exploratory Data Analysis (EDA) Module.
Prepares analytical aggregations for time-series trajectories, rolling trends,
calendar seasonalities, entity performance rankings, and exogenous correlation patterns.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def prepare_time_series_eda(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    frequency: str = "Daily",
) -> pd.DataFrame:
    """
    Aggregate sales strictly by date and compute frequency-adapted rolling mean, median, and trend.
    """
    if date_col not in df.columns or sales_col not in df.columns:
        return pd.DataFrame()

    df_ts = df.copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col])

    daily_agg = (
        df_ts.groupby(date_col)[sales_col]
        .agg(["sum", "mean", "median", "count"])
        .reset_index()
        .sort_values(date_col)
    )
    daily_agg.rename(columns={"sum": "Total_Sales", "mean": "Average_Sales", "median": "Median_Sales"}, inplace=True)

    # Frequency-adapted rolling windows
    freq_norm = str(frequency).strip().lower()
    if "weekly" in freq_norm:
        w1, w2 = 4, 12
        w1_label, w2_label = "4-Week", "12-Week"
    elif "monthly" in freq_norm:
        w1, w2 = 3, 6
        w1_label, w2_label = "3-Month", "6-Month"
    else:  # Daily or default
        w1, w2 = 7, 30
        w1_label, w2_label = "7-Day", "30-Day"

    n_records = len(daily_agg)
    if n_records < w2:
        w1 = max(2, min(3, n_records // 4)) if n_records >= 4 else 1
        w2 = max(w1 + 1, min(7, n_records // 2)) if n_records >= 4 else 2

    daily_agg[f"Rolling_Mean_Fast"] = daily_agg["Total_Sales"].rolling(window=w1, min_periods=1).mean()
    daily_agg[f"Rolling_Mean_Slow"] = daily_agg["Total_Sales"].rolling(window=w2, min_periods=1).mean()
    daily_agg["Rolling_Fast_Label"] = w1_label
    daily_agg["Rolling_Slow_Label"] = w2_label

    # Linear trend line values using numpy polyfit
    if len(daily_agg) > 2:
        x_vals = np.arange(len(daily_agg))
        slope, intercept = np.polyfit(x_vals, daily_agg["Total_Sales"], 1)
        daily_agg["Trend_Line"] = intercept + slope * x_vals

    return daily_agg


def filter_historical_by_range(
    df: pd.DataFrame,
    date_col: str = "Date",
    range_option: str = "OVERALL",
) -> pd.DataFrame:
    """
    Filter historical time series by actual date ranges (1 Month, 3 Months, 6 Months, or Overall)
    using pandas DateOffset.
    """
    if df.empty or date_col not in df.columns:
        return df

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    max_d = out[date_col].max()

    opt = str(range_option).strip().upper()
    if "1 MONTH" in opt or opt == "1M":
        cutoff = max_d - pd.DateOffset(months=1)
        filtered = out[out[date_col] >= cutoff]
        return filtered if len(filtered) >= 2 else out.tail(max(2, min(5, len(out))))
    elif "3 MONTH" in opt or opt == "3M":
        cutoff = max_d - pd.DateOffset(months=3)
        filtered = out[out[date_col] >= cutoff]
        return filtered if len(filtered) >= 2 else out.tail(max(2, min(15, len(out))))
    elif "6 MONTH" in opt or opt == "6M":
        cutoff = max_d - pd.DateOffset(months=6)
        filtered = out[out[date_col] >= cutoff]
        return filtered if len(filtered) >= 2 else out.tail(max(2, min(30, len(out))))
    else:  # OVERALL
        return out


def compute_historical_summary_kpis(
    df: pd.DataFrame,
    sales_col: str = "Sales",
    date_col: str = "Date",
    freq: str = "Daily",
) -> Dict[str, Any]:
    """
    Calculate executive KPI metrics for the entire historical dataset:
    Total, Average, Median, Min, Max Sales, Observations, Date Range, Frequency.
    """
    if df.empty or sales_col not in df.columns:
        return {
            "total_sales": 0.0, "avg_sales": 0.0, "median_sales": 0.0,
            "min_sales": 0.0, "max_sales": 0.0, "observations": 0,
            "date_range": "N/A", "frequency": freq,
        }

    s = pd.to_numeric(df[sales_col], errors="coerce").dropna()
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna() if date_col in df.columns else pd.Series()

    d_range = f"{str(dates.min())[:10]} → {str(dates.max())[:10]}" if not dates.empty else "N/A"

    return {
        "total_sales": float(s.sum()),
        "avg_sales": float(s.mean()),
        "median_sales": float(s.median()),
        "min_sales": float(s.min()) if not s.empty else 0.0,
        "max_sales": float(s.max()) if not s.empty else 0.0,
        "observations": len(s),
        "date_range": d_range,
        "frequency": freq,
    }


def compute_period_sales_summaries(
    df: pd.DataFrame,
    sales_col: str = "Sales",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Compute dynamic 1M, 3M, 6M, and Overall historical sales summaries directly from the uploaded dataset.
    """
    if df.empty or sales_col not in df.columns or date_col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, sales_col]).sort_values(date_col)
    max_d = work[date_col].max()

    periods = [
        ("LAST 1 MONTH", max_d - pd.DateOffset(months=1)),
        ("LAST 3 MONTHS", max_d - pd.DateOffset(months=3)),
        ("LAST 6 MONTHS", max_d - pd.DateOffset(months=6)),
        ("OVERALL", work[date_col].min()),
    ]

    records = []
    for label, cutoff in periods:
        sub = work[work[date_col] >= cutoff]
        if sub.empty:
            sub = work.tail(4 if "1" in label else (13 if "3" in label else 26))

        s = pd.to_numeric(sub[sales_col], errors="coerce").dropna()
        records.append({
            "Period": label,
            "Total Sales": float(s.sum()),
            "Average Sales": float(s.mean()) if not s.empty else 0.0,
            "Median Sales": float(s.median()) if not s.empty else 0.0,
            "Observations": len(s),
            "Start Date": str(sub[date_col].min())[:10],
            "End Date": str(sub[date_col].max())[:10],
        })

    return pd.DataFrame(records)


def prepare_seasonality_analysis(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    frequency: str = "Daily",
) -> Dict[str, pd.DataFrame]:
    """
    Extract calendar seasonality patterns (Day of Week, Month, Quarter)
    with average, median, and volume counts.
    """
    results: Dict[str, pd.DataFrame] = {}
    if date_col not in df.columns or sales_col not in df.columns:
        return results

    df_work = df.copy()
    df_work[date_col] = pd.to_datetime(df_work[date_col])

    # 1. Day of Week (applicable for Daily data)
    if frequency in ["Daily", "Irregular"]:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_work["Day_of_Week"] = df_work[date_col].dt.day_name()
        dow_agg = (
            df_work.groupby("Day_of_Week")[sales_col]
            .agg(Mean_Sales="mean", Median_Sales="median", Total_Sales="sum", Record_Count="count")
            .reindex(dow_order)
            .dropna(how="all")
            .reset_index()
        )
        results["dow"] = dow_agg

    # 2. Month of Year
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    df_work["Month_Name"] = df_work[date_col].dt.month_name()
    month_agg = (
        df_work.groupby("Month_Name")[sales_col]
        .agg(Mean_Sales="mean", Median_Sales="median", Total_Sales="sum", Record_Count="count")
        .reindex(month_order)
        .dropna(how="all")
        .reset_index()
    )
    results["month"] = month_agg

    # 3. Quarter
    df_work["Quarter"] = "Q" + df_work[date_col].dt.quarter.astype(str)
    quarter_agg = (
        df_work.groupby("Quarter")[sales_col]
        .agg(Mean_Sales="mean", Median_Sales="median", Total_Sales="sum", Record_Count="count")
        .reset_index()
        .sort_values("Quarter")
    )
    results["quarter"] = quarter_agg

    return results


def prepare_entity_analysis(
    df: pd.DataFrame,
    entity_col: str,
    sales_col: str = "Sales",
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Rank top and lowest performing entities (stores, products, categories, or departments).
    """
    if entity_col not in df.columns or sales_col not in df.columns:
        return {"top_entities": pd.DataFrame(), "bottom_entities": pd.DataFrame(), "total_entities": 0}

    agg = (
        df.groupby(entity_col)[sales_col]
        .agg(Total_Sales="sum", Mean_Sales="mean", Median_Sales="median", Records="count")
        .reset_index()
        .sort_values(by="Total_Sales", ascending=False)
    )

    total_sales = agg["Total_Sales"].sum()
    agg["Sales_Share_%"] = (agg["Total_Sales"] / max(total_sales, 1e-5)) * 100.0

    top_entities = agg.head(top_n).copy()
    bottom_entities = agg.tail(top_n).sort_values(by="Total_Sales", ascending=True).copy()

    return {
        "top_entities": top_entities,
        "bottom_entities": bottom_entities,
        "total_entities": len(agg),
        "total_sales": total_sales,
    }


def prepare_correlation_analysis(
    df: pd.DataFrame,
    sales_col: str = "Sales",
    order_col: str = "Order",
) -> Optional[Dict[str, Any]]:
    """
    Calculate statistical association metrics between volume (Orders/Units) and Revenue (Sales).
    """
    if sales_col not in df.columns or order_col not in df.columns:
        return None

    valid = df[[sales_col, order_col]].dropna()
    valid = valid[(valid[sales_col] >= 0) & (valid[order_col] >= 0)]
    if len(valid) < 5:
        return None

    s = valid[sales_col].astype(float)
    o = valid[order_col].astype(float)

    pearson_corr = float(s.corr(o, method="pearson"))
    spearman_corr = float(s.corr(o, method="spearman"))

    # Compute OLS slope and intercept for trend line
    slope, intercept = np.polyfit(o, s, 1) if len(valid) > 2 else (0.0, 0.0)

    # Subsample scatter points for UI responsiveness if very large
    sample_df = valid.sample(min(2000, len(valid)), random_state=42) if len(valid) > 2000 else valid

    return {
        "pearson_corr": round(pearson_corr, 4),
        "spearman_corr": round(spearman_corr, 4),
        "slope": round(float(slope), 2),
        "intercept": round(float(intercept), 2),
        "sample_df": sample_df,
        "total_paired_records": len(valid),
    }


def prepare_discount_analysis(
    df: pd.DataFrame,
    discount_col: str,
    sales_col: str = "Sales",
) -> Optional[Dict[str, Any]]:
    """
    Compare sales distributions during discounted vs non-discounted periods.
    """
    if discount_col not in df.columns or sales_col not in df.columns:
        return None

    s_str = df[discount_col].astype(str).str.strip().str.lower()
    is_promo = s_str.isin(["yes", "1", "true", "y", "promo", "discount"])

    promo_sales = df.loc[is_promo, sales_col].dropna()
    reg_sales = df.loc[~is_promo, sales_col].dropna()

    if len(promo_sales) < 3 or len(reg_sales) < 3:
        return None

    p_mean = float(promo_sales.mean())
    p_med = float(promo_sales.median())
    r_mean = float(reg_sales.mean())
    r_med = float(reg_sales.median())

    mean_diff_pct = ((p_mean - r_mean) / max(r_mean, 1e-5)) * 100.0
    med_diff_pct = ((p_med - r_med) / max(r_med, 1e-5)) * 100.0

    summary_df = pd.DataFrame([
        {"Condition": "Promotional / Discount", "Mean Sales": round(p_mean, 2), "Median Sales": round(p_med, 2), "Observations": len(promo_sales)},
        {"Condition": "Regular / Non-Discount", "Mean Sales": round(r_mean, 2), "Median Sales": round(r_med, 2), "Observations": len(reg_sales)},
    ])

    return {
        "promo_mean": p_mean,
        "regular_mean": r_mean,
        "promo_median": p_med,
        "regular_median": r_med,
        "mean_diff_pct": round(mean_diff_pct, 2),
        "median_diff_pct": round(med_diff_pct, 2),
        "summary_df": summary_df,
    }


def prepare_holiday_analysis(
    df: pd.DataFrame,
    holiday_col: str,
    sales_col: str = "Sales",
) -> Optional[Dict[str, Any]]:
    """
    Compare sales metrics during holiday vs non-holiday calendar periods.
    """
    if holiday_col not in df.columns or sales_col not in df.columns:
        return None

    h_str = df[holiday_col].astype(str).str.strip().str.lower()
    is_holiday = h_str.isin(["yes", "1", "true", "y", "holiday", "1.0"])

    hol_sales = df.loc[is_holiday, sales_col].dropna()
    reg_sales = df.loc[~is_holiday, sales_col].dropna()

    if len(hol_sales) < 3 or len(reg_sales) < 3:
        return None

    h_mean = float(hol_sales.mean())
    h_med = float(hol_sales.median())
    r_mean = float(reg_sales.mean())
    r_med = float(reg_sales.median())

    mean_diff_pct = ((h_mean - r_mean) / max(r_mean, 1e-5)) * 100.0
    med_diff_pct = ((h_med - r_med) / max(r_med, 1e-5)) * 100.0

    summary_df = pd.DataFrame([
        {"Condition": "Designated Holiday", "Mean Sales": round(h_mean, 2), "Median Sales": round(h_med, 2), "Observations": len(hol_sales)},
        {"Condition": "Regular Calendar Day", "Mean Sales": round(r_mean, 2), "Median Sales": round(r_med, 2), "Observations": len(reg_sales)},
    ])

    return {
        "holiday_mean": h_mean,
        "regular_mean": r_mean,
        "holiday_median": h_med,
        "regular_median": r_med,
        "mean_diff_pct": round(mean_diff_pct, 2),
        "median_diff_pct": round(med_diff_pct, 2),
        "summary_df": summary_df,
    }
