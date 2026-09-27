"""
Data Cleaner Module.
Executes non-destructive data cleaning, chronological sequencing, missing value handling,
and transparent Before vs. After audit statistics.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def clean_dataset(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    order_col: Optional[str] = "Order",
    store_col: Optional[str] = None,
    product_col: Optional[str] = None,
    negative_handling: str = "Clip to Zero",  # 'Clip to Zero' or 'Drop'
    outlier_treatment: str = "Keep",  # 'Keep', 'Cap', or 'Remove'
    impute_missing: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Safely clean dataset and generate comprehensive before/after audit statistics.
    Never deletes legitimate sales spikes without explicit user configuration.
    """
    initial_rows = len(df)
    initial_cols = df.shape[1]
    initial_missing = int(df.isna().sum().sum())
    initial_duplicates = int(df.duplicated().sum())

    clean_df = df.copy()

    # 1. Standardize string whitespace
    for col in clean_df.select_dtypes(include=["object"]).columns:
        clean_df[col] = clean_df[col].astype(str).str.strip()

    # 2. Remove exact duplicate rows
    clean_df = clean_df.drop_duplicates()
    duplicates_removed = initial_duplicates

    # 3. Convert Date and Chronological Sequencing
    if date_col in clean_df.columns:
        clean_df[date_col] = pd.to_datetime(clean_df[date_col], errors="coerce")
        clean_df = clean_df.dropna(subset=[date_col])

        # Entity-aware chronological sorting
        sort_keys = []
        if store_col and store_col in clean_df.columns:
            sort_keys.append(store_col)
        if product_col and product_col in clean_df.columns:
            sort_keys.append(product_col)
        sort_keys.append(date_col)

        clean_df = clean_df.sort_values(sort_keys).reset_index(drop=True)

    # 4. Target Variable Hygiene (Sales & Orders)
    if sales_col in clean_df.columns:
        clean_df[sales_col] = pd.to_numeric(clean_df[sales_col], errors="coerce")
        clean_df = clean_df.dropna(subset=[sales_col])

        if negative_handling == "Drop":
            clean_df = clean_df[clean_df[sales_col] >= 0]
        else:
            clean_df[sales_col] = clean_df[sales_col].clip(lower=0.0)

    if order_col and order_col in clean_df.columns:
        clean_df[order_col] = pd.to_numeric(clean_df[order_col], errors="coerce")
        if negative_handling == "Drop":
            clean_df = clean_df[clean_df[order_col].isna() | (clean_df[order_col] >= 0)]
        else:
            clean_df[order_col] = clean_df[order_col].clip(lower=0.0)

    # 5. Outlier Statistics & Optional Robust Treatment
    outliers_detected = 0
    outliers_modified = 0
    if sales_col in clean_df.columns and len(clean_df) > 4:
        s_vals = clean_df[sales_col]
        q1 = float(s_vals.quantile(0.25))
        q3 = float(s_vals.quantile(0.75))
        iqr = q3 - q1
        lower_bound = max(0.0, q1 - 1.5 * iqr)
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (clean_df[sales_col] < lower_bound) | (clean_df[sales_col] > upper_bound)
        outliers_detected = int(outlier_mask.sum())

        if outlier_treatment == "Cap":
            clean_df[sales_col] = clean_df[sales_col].clip(lower=lower_bound, upper=upper_bound)
            outliers_modified = outliers_detected
        elif outlier_treatment == "Remove":
            # Only remove extreme statistical anomalies (> 3.0 * IQR)
            extreme_upper = q3 + 3.0 * iqr
            extreme_lower = max(0.0, q1 - 3.0 * iqr)
            extreme_mask = (clean_df[sales_col] < extreme_lower) | (clean_df[sales_col] > extreme_upper)
            outliers_modified = int(extreme_mask.sum())
            clean_df = clean_df[~extreme_mask].reset_index(drop=True)

    # 6. Missing Feature Imputation
    if impute_missing:
        numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()
        group_keys = [k for k in [store_col, product_col] if k and k in clean_df.columns]

        for col in numeric_cols:
            if clean_df[col].isna().sum() > 0:
                if group_keys:
                    clean_df[col] = clean_df.groupby(group_keys)[col].transform(lambda s: s.ffill().bfill())
                clean_df[col] = clean_df[col].fillna(clean_df[col].median()).fillna(0.0)

        cat_cols = clean_df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if col != date_col and clean_df[col].isna().sum() > 0:
                mode_val = clean_df[col].mode().iloc[0] if not clean_df[col].mode().empty else "Unknown"
                clean_df[col] = clean_df[col].fillna(mode_val)

    final_rows = len(clean_df)
    final_cols = clean_df.shape[1]
    final_missing = int(clean_df.isna().sum().sum())

    before_after = pd.DataFrame([
        {"Metric": "Total Rows", "Before Cleaning": f"{initial_rows:,}", "After Cleaning": f"{final_rows:,}", "Change": f"{final_rows - initial_rows:+d}"},
        {"Metric": "Duplicate Rows", "Before Cleaning": f"{initial_duplicates:,}", "After Cleaning": "0", "Change": f"{-initial_duplicates:+d}"},
        {"Metric": "Missing Cells", "Before Cleaning": f"{initial_missing:,}", "After Cleaning": f"{final_missing:,}", "Change": f"{final_missing - initial_missing:+d}"},
        {"Metric": "Total Columns", "Before Cleaning": f"{initial_cols}", "After Cleaning": f"{final_cols}", "Change": "0"},
        {"Metric": "Potential Outliers", "Before Cleaning": f"{outliers_detected:,}", "After Cleaning": f"{outliers_detected - outliers_modified:,}", "Change": f"{-outliers_modified:+d}"},
    ])

    stats = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_removed": initial_rows - final_rows,
        "duplicates_removed": duplicates_removed,
        "initial_missing": initial_missing,
        "final_missing": final_missing,
        "outliers_detected": outliers_detected,
        "outliers_modified": outliers_modified,
        "negative_handling": negative_handling,
        "outlier_treatment": outlier_treatment,
        "before_after_df": before_after,
    }

    return clean_df, stats
