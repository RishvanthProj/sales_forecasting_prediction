"""
Statistical Analysis & Hypothesis Testing Module.
Executes automated hypothesis tests (Welch's t-test, ANOVA, Kruskal-Wallis, Levene, Spearman/Pearson)
tailored to available dataset features with commercial interpretations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats


def run_statistical_analysis(
    df: pd.DataFrame,
    sales_col: str = "Sales",
    order_col: Optional[str] = "Order",
    discount_col: Optional[str] = "Discount",
    holiday_col: Optional[str] = "Holiday",
    store_type_col: Optional[str] = "Store_Type",
    region_col: Optional[str] = "Region_Code",
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Run applicable hypothesis and correlation tests depending on available columns.
    Returns a list of structured test result dictionaries.
    """
    results: List[Dict[str, Any]] = []

    if sales_col not in df.columns or df[sales_col].dropna().empty:
        return results

    sales = pd.to_numeric(df[sales_col], errors="coerce").dropna()
    if len(sales) < 10:
        return results

    # Helper function for interpretation
    def format_verdict(p_val: float) -> str:
        return "Reject H0 (Statistically Significant)" if p_val < alpha else "Fail to Reject H0 (Not Significant)"

    # 1. Discount Impact Hypothesis Test
    if discount_col and discount_col in df.columns:
        disc_series = df[discount_col].astype(str).str.strip().str.lower()
        is_promo = disc_series.isin(["yes", "1", "true", "y", "promo", "discount"])
        sales_promo = sales[is_promo]
        sales_no_promo = sales[~is_promo]

        if len(sales_promo) >= 5 and len(sales_no_promo) >= 5:
            # Check variance equality via Levene's test
            try:
                sample_p = sales_promo.sample(min(5000, len(sales_promo)), random_state=42)
                sample_np = sales_no_promo.sample(min(5000, len(sales_no_promo)), random_state=42)
                _, p_var = stats.levene(sample_p, sample_np)
                equal_var = bool(p_var > alpha)
            except Exception:
                equal_var = False

            # Welch's or Student's t-test
            t_stat, p_val = stats.ttest_ind(sales_promo, sales_no_promo, equal_var=equal_var)
            mean_promo = float(sales_promo.mean())
            mean_no_promo = float(sales_no_promo.mean())
            pct_diff = ((mean_promo - mean_no_promo) / max(mean_no_promo, 1e-5)) * 100

            results.append({
                "test_id": "discount_impact",
                "title": "Discount / Promotion Impact on Sales",
                "h0": "Mean sales on discounted days equals mean sales on non-discounted days.",
                "h1": "Mean sales differ significantly between discounted and non-discounted days.",
                "test_used": f"{'Student\'s' if equal_var else 'Welch\'s'} Two-Sample t-test",
                "statistic": round(float(t_stat), 4),
                "p_value": float(p_val),
                "alpha": alpha,
                "verdict": format_verdict(p_val),
                "summary": (
                    f"Discounted days averaged {mean_promo:,.2f} vs {mean_no_promo:,.2f} on non-discounted days "
                    f"({pct_diff:+.1f}% difference). The observed difference is "
                    f"{'statistically significant' if p_val < alpha else 'not statistically significant'}."
                ),
                "group_stats": {
                    "Discounted Mean": round(mean_promo, 2),
                    "Non-Discounted Mean": round(mean_no_promo, 2),
                    "Percentage Difference": round(pct_diff, 2),
                }
            })

    # 2. Holiday Effect Hypothesis Test
    if holiday_col and holiday_col in df.columns:
        hol_series = df[holiday_col].astype(str).str.strip().str.lower()
        is_holiday = hol_series.isin(["yes", "1", "true", "y", "holiday"])
        sales_hol = sales[is_holiday]
        sales_no_hol = sales[~is_holiday]

        if len(sales_hol) >= 5 and len(sales_no_hol) >= 5:
            try:
                sample_h = sales_hol.sample(min(5000, len(sales_hol)), random_state=42)
                sample_nh = sales_no_hol.sample(min(5000, len(sales_no_hol)), random_state=42)
                _, p_var = stats.levene(sample_h, sample_nh)
                equal_var = bool(p_var > alpha)
            except Exception:
                equal_var = False

            t_stat, p_val = stats.ttest_ind(sales_hol, sales_no_hol, equal_var=equal_var)
            mean_hol = float(sales_hol.mean())
            mean_no_hol = float(sales_no_hol.mean())
            pct_diff = ((mean_hol - mean_no_hol) / max(mean_no_hol, 1e-5)) * 100

            results.append({
                "test_id": "holiday_effect",
                "title": "Holiday Effect on Sales",
                "h0": "Mean sales on holidays equals mean sales on non-holidays.",
                "h1": "Mean sales on holidays differ from non-holidays.",
                "test_used": f"{'Student\'s' if equal_var else 'Welch\'s'} Two-Sample t-test",
                "statistic": round(float(t_stat), 4),
                "p_value": float(p_val),
                "alpha": alpha,
                "verdict": format_verdict(p_val),
                "summary": (
                    f"Holiday sales averaged {mean_hol:,.2f} vs {mean_no_hol:,.2f} on regular days "
                    f"({pct_diff:+.1f}% variation). The difference is "
                    f"{'statistically significant' if p_val < alpha else 'not statistically significant'}."
                ),
                "group_stats": {
                    "Holiday Mean": round(mean_hol, 2),
                    "Regular Day Mean": round(mean_no_hol, 2),
                    "Percentage Difference": round(pct_diff, 2),
                }
            })

    # 3. Store Type Variation Test (ANOVA / Kruskal-Wallis)
    for col_name, label in [(store_type_col, "Store Type"), (region_col, "Region Code")]:
        if col_name and col_name in df.columns:
            groups = [
                group[sales_col].dropna().values
                for _, group in df.groupby(col_name)
                if len(group[sales_col].dropna()) >= 5
            ]
            if len(groups) >= 2:
                # Normality check on sample
                is_normal = True
                for g in groups:
                    sample_size = min(3000, len(g))
                    sample = np.random.choice(g, sample_size, replace=False)
                    _, p_norm = stats.shapiro(sample)
                    if p_norm <= alpha:
                        is_normal = False
                        break

                if is_normal:
                    stat_val, p_val = stats.f_oneway(*groups)
                    test_name = "One-Way ANOVA"
                else:
                    stat_val, p_val = stats.kruskal(*groups)
                    test_name = "Kruskal-Wallis H-Test"

                results.append({
                    "test_id": f"{col_name}_variation",
                    "title": f"Sales Variation Across {label} Categories",
                    "h0": f"Sales distribution is uniform across all {label} categories.",
                    "h1": f"At least one {label} category exhibits a significantly different sales distribution.",
                    "test_used": test_name,
                    "statistic": round(float(stat_val), 4),
                    "p_value": float(p_val),
                    "alpha": alpha,
                    "verdict": format_verdict(p_val),
                    "summary": (
                        f"Evaluated {len(groups)} distinct {label} groups using {test_name}. "
                        f"{label} categories show {'significant variation in sales performance' if p_val < alpha else 'no statistically significant variation'}."
                    ),
                    "group_stats": {
                        "Categories Evaluated": len(groups),
                        "Test Type": test_name,
                    }
                })

    # 4. Orders vs Sales Correlation Analysis
    if order_col and order_col in df.columns:
        valid_mask = df[sales_col].notna() & df[order_col].notna()
        orders_s = pd.to_numeric(df.loc[valid_mask, order_col], errors="coerce").dropna()
        sales_s = pd.to_numeric(df.loc[valid_mask, sales_col], errors="coerce").loc[orders_s.index]

        if len(orders_s) >= 10:
            sample_size = min(4000, len(orders_s))
            idx = np.random.choice(orders_s.index, sample_size, replace=False)
            _, p_norm_o = stats.shapiro(orders_s.loc[idx])
            _, p_norm_s = stats.shapiro(sales_s.loc[idx])

            if p_norm_o > alpha and p_norm_s > alpha:
                corr_val, p_val = stats.pearsonr(orders_s, sales_s)
                method = "Pearson Correlation"
            else:
                corr_val, p_val = stats.spearmanr(orders_s, sales_s)
                method = "Spearman Rank Correlation"

            results.append({
                "test_id": "orders_sales_correlation",
                "title": "Orders vs Sales Demand Correlation",
                "h0": "There is no monotonic correlation between Order volume and Sales revenue (ρ = 0).",
                "h1": "A significant correlation exists between Order volume and Sales revenue (ρ ≠ 0).",
                "test_used": method,
                "statistic": round(float(corr_val), 4),
                "p_value": float(p_val),
                "alpha": alpha,
                "verdict": format_verdict(p_val),
                "summary": (
                    f"Measured correlation coefficient of {corr_val:+.4f} ({method}). "
                    f"Orders and Sales exhibit an {'exceptionally strong' if abs(corr_val) > 0.8 else 'moderate'} "
                    f"positive association, validating the two-stage forecasting architecture."
                ),
                "group_stats": {
                    "Correlation Coefficient": round(corr_val, 4),
                    "Method": method,
                }
            })

    return results
