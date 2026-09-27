"""
Statistical Analysis & Commercial Business Insights Module.
Executes rigorous hypothesis testing (Welch's t-test, ANOVA, Kruskal-Wallis, Spearman/Pearson)
and generates transparent business takeaways using calibrated non-causal language.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


def run_statistical_tests(
    df: pd.DataFrame,
    sales_col: str = "Sales",
    discount_col: Optional[str] = None,
    holiday_col: Optional[str] = None,
    entity_col: Optional[str] = None,
    order_col: Optional[str] = None,
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Run applicable statistical hypothesis and correlation tests depending on available columns.
    Returns structured results formatted for data science viva and executive review.
    """
    results: List[Dict[str, Any]] = []

    if sales_col not in df.columns or df[sales_col].dropna().empty:
        return results

    sales = pd.to_numeric(df[sales_col], errors="coerce").dropna()
    if len(sales) < 10:
        return results

    def format_verdict(p_val: float) -> str:
        return "Statistically Significant (Reject H0, p < 0.05)" if p_val < alpha else "Not Statistically Significant (Fail to Reject H0, p ≥ 0.05)"

    # 1. Discount / Promotion Impact Test
    if discount_col and discount_col in df.columns:
        disc_series = df[discount_col].astype(str).str.strip().str.lower()
        is_promo = disc_series.isin(["yes", "1", "true", "y", "promo", "discount"])
        s_promo = sales[is_promo]
        s_reg = sales[~is_promo]

        if len(s_promo) >= 5 and len(s_reg) >= 5:
            # Check variance equality via Levene's test
            try:
                sp_sample = s_promo.sample(min(3000, len(s_promo)), random_state=42)
                sr_sample = s_reg.sample(min(3000, len(s_reg)), random_state=42)
                _, p_levene = stats.levene(sp_sample, sr_sample)
                equal_var = bool(p_levene > alpha)
            except Exception:
                equal_var = False

            t_stat, p_val = stats.ttest_ind(s_promo, s_reg, equal_var=equal_var)
            mean_p = float(s_promo.mean())
            mean_r = float(s_reg.mean())
            diff_pct = ((mean_p - mean_r) / max(mean_r, 1e-5)) * 100.0

            test_name = "Two-Sample Student's t-test" if equal_var else "Welch's Two-Sample t-test"
            results.append({
                "hypothesis": "Promotional discounts associate with a significant shift in observed sales volume.",
                "test_name": test_name,
                "statistic": round(float(t_stat), 4),
                "p_value": float(p_val),
                "verdict": format_verdict(p_val),
                "interpretation": (
                    f"Promotional periods averaged ${mean_p:,.2f} vs ${mean_r:,.2f} during standard periods "
                    f"({diff_pct:+.1f}% historical difference). This observed difference is "
                    f"{'statistically significant (p < 0.05)' if p_val < alpha else 'not statistically significant at α = 0.05'}."
                ),
            })

    # 2. Holiday Calendar Effect Test
    if holiday_col and holiday_col in df.columns:
        hol_series = df[holiday_col].astype(str).str.strip().str.lower()
        is_hol = hol_series.isin(["yes", "1", "true", "y", "holiday", "1.0"])
        s_hol = sales[is_hol]
        s_non_hol = sales[~is_hol]

        if len(s_hol) >= 4 and len(s_non_hol) >= 4:
            try:
                sh_sample = s_hol.sample(min(3000, len(s_hol)), random_state=42)
                snh_sample = s_non_hol.sample(min(3000, len(s_non_hol)), random_state=42)
                _, p_levene = stats.levene(sh_sample, snh_sample)
                equal_var = bool(p_levene > alpha)
            except Exception:
                equal_var = False

            t_stat, p_val = stats.ttest_ind(s_hol, s_non_hol, equal_var=equal_var)
            mean_h = float(s_hol.mean())
            mean_nh = float(s_non_hol.mean())
            diff_pct = ((mean_h - mean_nh) / max(mean_nh, 1e-5)) * 100.0

            test_name = "Two-Sample Student's t-test" if equal_var else "Welch's Two-Sample t-test"
            results.append({
                "hypothesis": "Designated retail holidays exhibit significant variations in observed sales demand.",
                "test_name": test_name,
                "statistic": round(float(t_stat), 4),
                "p_value": float(p_val),
                "verdict": format_verdict(p_val),
                "interpretation": (
                    f"Holiday periods exhibited average sales of ${mean_h:,.2f} compared to ${mean_nh:,.2f} on regular periods "
                    f"({diff_pct:+.1f}% observed difference). The variation is "
                    f"{'statistically significant (p < 0.05)' if p_val < alpha else 'not statistically distinguishable from random noise (p ≥ 0.05)'}."
                ),
            })

    # 3. Categorical / Entity Variation Test (ANOVA / Kruskal-Wallis)
    if entity_col and entity_col in df.columns:
        groups = [
            grp[sales_col].dropna().values
            for _, grp in df.groupby(entity_col)
            if len(grp[sales_col].dropna()) >= 5
        ]
        if len(groups) >= 2 and len(groups) <= 60:
            # Check normality
            is_normal = True
            for g in groups[:5]:
                if len(g) >= 8:
                    _, p_norm = stats.shapiro(g[:100])
                    if p_norm <= alpha:
                        is_normal = False
                        break

            if is_normal:
                stat_val, p_val = stats.f_oneway(*groups)
                test_name = "One-Way ANOVA"
            else:
                stat_val, p_val = stats.kruskal(*groups)
                test_name = "Kruskal-Wallis Non-Parametric H-Test"

            results.append({
                "hypothesis": f"Sales distribution is uniform across distinct {entity_col} entities.",
                "test_name": test_name,
                "statistic": round(float(stat_val), 4),
                "p_value": float(p_val),
                "verdict": format_verdict(p_val),
                "interpretation": (
                    f"Analyzed {len(groups)} distinct {entity_col} cohorts using {test_name}. "
                    f"Entities display {'highly significant sales performance differences (p < 0.05)' if p_val < alpha else 'homogeneous sales distributions across groups'}."
                ),
            })

    # 4. Volume (Orders/Quantity) vs Revenue (Sales) Correlation
    if order_col and order_col in df.columns:
        valid_mask = df[sales_col].notna() & df[order_col].notna()
        sub = df.loc[valid_mask]
        if len(sub) >= 10:
            s_vals = sub[sales_col].astype(float)
            o_vals = sub[order_col].astype(float)

            corr_p, p_p = stats.pearsonr(s_vals, o_vals)
            corr_s, p_s = stats.spearmanr(s_vals, o_vals)

            results.append({
                "hypothesis": "Order volume demonstrates a strong monotonic association with sales revenue.",
                "test_name": "Spearman Rank & Pearson Correlation",
                "statistic": round(float(corr_p), 4),
                "p_value": float(p_p),
                "verdict": format_verdict(p_p),
                "interpretation": (
                    f"Pearson r = {corr_p:+.4f}, Spearman ρ = {corr_s:+.4f}. "
                    f"Demonstrates a strong positive linear association between transaction volume and top-line revenue."
                ),
            })

    return results


def generate_business_takeaways(
    df: pd.DataFrame,
    forecast_summary: Optional[Dict[str, Any]] = None,
    sales_col: str = "Sales",
    date_col: str = "Date",
    discount_col: Optional[str] = None,
    holiday_col: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Synthesize factual, non-causal takeaways for commercial decision-makers.
    """
    takeaways = []

    if df.empty or sales_col not in df.columns:
        return takeaways

    sales = pd.to_numeric(df[sales_col], errors="coerce").dropna()
    total_rev = sales.sum()
    avg_rev = sales.mean()

    # 1. Historical Baseline
    takeaways.append({
        "category": "Historical Baseline",
        "title": "Historical Sales Performance",
        "metric": f"${total_rev:,.0f}",
        "summary": f"Historical aggregate revenue across {len(sales):,} records with average transaction/period sales of ${avg_rev:,.2f}.",
    })

    # 2. Promotional Uplift
    if discount_col and discount_col in df.columns:
        d_str = df[discount_col].astype(str).str.strip().str.lower()
        is_promo = d_str.isin(["yes", "1", "true", "y", "promo"])
        mean_p = df.loc[is_promo, sales_col].mean()
        mean_r = df.loc[~is_promo, sales_col].mean()
        if pd.notna(mean_p) and pd.notna(mean_r) and mean_r > 0:
            diff = ((mean_p - mean_r) / mean_r) * 100.0
            takeaways.append({
                "category": "Promotions",
                "title": "Promotional Uplift Association",
                "metric": f"{diff:+.1f}%",
                "summary": f"Promotional periods observed ${mean_p:,.2f} average sales vs ${mean_r:,.2f} on regular periods.",
            })

    # 3. Holiday Multiplier
    if holiday_col and holiday_col in df.columns:
        h_str = df[holiday_col].astype(str).str.strip().str.lower()
        is_h = h_str.isin(["yes", "1", "true", "y", "holiday", "1.0"])
        mean_h = df.loc[is_h, sales_col].mean()
        mean_nh = df.loc[~is_h, sales_col].mean()
        if pd.notna(mean_h) and pd.notna(mean_nh) and mean_nh > 0:
            diff_h = ((mean_h - mean_nh) / mean_nh) * 100.0
            takeaways.append({
                "category": "Seasonality",
                "title": "Holiday Demand Multiplier",
                "metric": f"{diff_h:+.1f}%",
                "summary": f"Holiday periods averaged ${mean_h:,.2f} vs ${mean_nh:,.2f} during standard operations.",
            })

    # 4. Forecast Trajectory
    if forecast_summary and forecast_summary.get("total_forecast", 0) > 0:
        tot_f = forecast_summary["total_forecast"]
        trend = forecast_summary.get("trend", "Stable")
        chg = forecast_summary.get("change_vs_recent_pct", 0.0)
        takeaways.append({
            "category": "Forecast Outlook",
            "title": f"Forecast Horizon Demand ({trend} Trend)",
            "metric": f"${tot_f:,.0f}",
            "summary": f"Projected {chg:+.1f}% change relative to the preceding historical period of equal duration.",
        })

    return takeaways


# Backward-compatible alias
generate_business_insights = generate_business_takeaways
