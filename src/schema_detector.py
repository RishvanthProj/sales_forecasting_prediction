"""
Schema Detector & Column Role Understanding Module.
Detects semantic roles, matches aliases, categorizes required vs. unnecessary columns,
and supports manual overrides with zero data leakage.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
import pandas as pd
import numpy as np


# Semantic Candidate Patterns (normalized lowercase alphanumeric)
CANDIDATE_PATTERNS = {
    "date": [
        "date", "orderdate", "salesdate", "transactiondate", "transdate",
        "timestamp", "datetime", "day", "invoicedate", "perioddate", "recorddate",
        "week", "month", "purchasedate"
    ],
    "sales": [
        "sales", "weeklysales", "monthlysales", "dailysales", "totalsales",
        "revenue", "totalrevenue", "netsales", "grosssales", "revenueamount",
        "amount", "turnover", "totalamount", "saleprice"
    ],
    "order": [
        "order", "orders", "ordercount", "quantity", "qty",
        "unitssold", "units", "demand", "volume", "totalorders", "itemcount"
    ],
    "store": [
        "store", "storeid", "storecode", "branch", "branchid", "shop",
        "shopid", "outlet", "outletid", "locationid", "facility"
    ],
    "product": [
        "product", "productid", "item", "itemid", "sku", "skuid",
        "category", "productcategory", "family", "productname", "itemname"
    ],
    "discount": [
        "discount", "promotion", "promo", "discountflag", "isdiscount",
        "discountavailable", "discountpct", "promoflag", "onpromotion"
    ],
    "holiday": [
        "holiday", "isholiday", "holidayflag", "publicholiday", "ispublicholiday",
        "vacation", "nationalholiday", "stateholiday"
    ],
    "department": [
        "dept", "department", "deptid", "departmentid", "section"
    ],
    "region": [
        "region", "regioncode", "zone", "territory", "area", "state"
    ],
}


def normalize_col_name(col: str) -> str:
    """Normalize column name to lowercase alphanumeric characters only."""
    return re.sub(r"[^a-z0-9]", "", str(col).strip().lower())


def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze dataset columns, detect semantic roles, and classify each column.
    Returns detected candidates, chosen mappings, and a column audit report.
    """
    columns = list(df.columns)
    normalized = {col: normalize_col_name(col) for col in columns}

    detected_roles: Dict[str, Optional[str]] = {
        "date_col": None,
        "sales_col": None,
        "order_col": None,
        "store_col": None,
        "product_col": None,
        "discount_col": None,
        "holiday_col": None,
        "dept_col": None,
        "region_col": None,
    }

    candidates: Dict[str, List[str]] = {
        "date": [],
        "sales": [],
        "order": [],
        "store": [],
        "product": [],
        "discount": [],
        "holiday": [],
        "department": [],
        "region": [],
    }

    # 1. Match candidates based on normalized string heuristics
    for col, norm in normalized.items():
        for role, patterns in CANDIDATE_PATTERNS.items():
            if norm in patterns or any(norm.endswith(p) or p in norm for p in patterns):
                candidates[role].append(col)

    # 2. Refine candidates using datatypes and heuristics
    # Date candidate check
    for col in candidates["date"]:
        try:
            converted = pd.to_datetime(df[col].dropna().head(40), errors="coerce")
            if converted.notna().mean() >= 0.75:
                detected_roles["date_col"] = col
                break
        except Exception:
            continue

    if not detected_roles["date_col"]:
        # Fallback: scan columns for datetime parseability
        for col in columns:
            if df[col].dtype == "object" or "date" in normalized[col] or "time" in normalized[col]:
                try:
                    converted = pd.to_datetime(df[col].dropna().head(30), errors="coerce")
                    if converted.notna().mean() >= 0.8:
                        detected_roles["date_col"] = col
                        break
                except Exception:
                    continue

    # Sales candidate check (must be numeric)
    sales_numeric_cands = [c for c in candidates["sales"] if pd.api.types.is_numeric_dtype(df[c])]
    if sales_numeric_cands:
        # Prioritize exact names like "weekly_sales", "sales", "revenue"
        exact = [c for c in sales_numeric_cands if normalized[c] in ["weeklysales", "sales", "revenue", "totalsales", "totalrevenue"]]
        detected_roles["sales_col"] = exact[0] if exact else sales_numeric_cands[0]
    else:
        # Fallback: look for any numeric column with 'sales' or 'revenue' in name
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]) and ("sale" in col.lower() or "rev" in col.lower()):
                detected_roles["sales_col"] = col
                break

    # Order candidate check (must be numeric and distinct from sales)
    order_numeric_cands = [
        c for c in candidates["order"]
        if pd.api.types.is_numeric_dtype(df[c]) and c != detected_roles["sales_col"]
    ]
    if order_numeric_cands:
        exact = [c for c in order_numeric_cands if normalized[c] in ["order", "orders", "quantity", "unitssold", "demand"]]
        detected_roles["order_col"] = exact[0] if exact else order_numeric_cands[0]

    # Entity candidates (Store / Branch / Facility)
    if candidates["store"]:
        detected_roles["store_col"] = candidates["store"][0]

    # Product / SKU / Category
    if candidates["product"]:
        detected_roles["product_col"] = candidates["product"][0]

    # Discount / Promotion
    if candidates["discount"]:
        detected_roles["discount_col"] = candidates["discount"][0]

    # Holiday
    if candidates["holiday"]:
        detected_roles["holiday_col"] = candidates["holiday"][0]

    # Department
    if candidates["department"]:
        detected_roles["dept_col"] = candidates["department"][0]

    # Region
    if candidates["region"]:
        detected_roles["region_col"] = candidates["region"][0]

    # 3. Comprehensive Column Classification & Transparency Audit
    audit_records = []
    chosen_role_cols = {v: k for k, v in detected_roles.items() if v is not None}

    total_rows = len(df)
    for col in columns:
        col_type = str(df[col].dtype)
        n_unique = int(df[col].nunique(dropna=False))
        uniqueness_ratio = n_unique / max(total_rows, 1)
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / max(total_rows, 1)) * 100, 2)
        norm = normalized[col]

        category = "USEFUL"
        status = "KEEP"
        reason = "Predictive numeric or categorical feature for ML modeling."
        role_label = chosen_role_cols.get(col, "Predictor Feature")

        if col == detected_roles["date_col"]:
            category = "REQUIRED / PRIMARY"
            status = "KEEP"
            role_label = "Index: Date"
            reason = "Primary temporal index for chronological ordering and lag generation."
        elif col == detected_roles["sales_col"]:
            category = "REQUIRED / PRIMARY"
            status = "KEEP"
            role_label = "Target: Sales"
            reason = "Primary continuous target variable to be forecasted."
        elif col == detected_roles["order_col"]:
            category = "USEFUL"
            status = "KEEP"
            role_label = "Predictor: Order/Demand"
            reason = "Correlated volume indicator used as supporting predictive feature."
        elif col in [detected_roles["store_col"], detected_roles["product_col"], detected_roles["dept_col"]]:
            category = "USEFUL"
            status = "KEEP"
            role_label = f"Entity: {col}"
            reason = "Entity grouping key for entity-specific historical lags and rolling averages."
        elif col in [detected_roles["discount_col"], detected_roles["holiday_col"]]:
            category = "OPTIONAL"
            status = "KEEP"
            role_label = "Business Driver"
            reason = "Calendar promotional or holiday shock indicator."
        elif n_unique <= 1:
            category = "UNUSED / IRRELEVANT"
            status = "EXCLUDED"
            role_label = "Zero-Variance Constant"
            reason = "Single constant value across all rows; contains zero predictive variance."
        elif norm in ["id", "rowid", "uuid", "transactionid", "transid", "index", "unnamed0"] or "uuid" in norm or (norm.endswith("id") and norm not in ["storeid", "productid", "itemid", "branchid", "deptid", "locationid"]):
            category = "UNUSED / IRRELEVANT"
            status = "EXCLUDED"
            role_label = "Record Identifier"
            reason = "Arbitrary transactional row ID; causes spurious memorization."
        elif df[col].dtype == "object" and uniqueness_ratio > 0.7 and n_unique > 50:
            category = "UNUSED / IRRELEVANT"
            status = "EXCLUDED"
            role_label = "High Cardinality Text"
            reason = f"High-cardinality string field ({n_unique} unique values) with low predictive signal."
        elif any(term in norm for term in ["url", "link", "comment", "note", "desc", "hash", "timestamp"]):
            category = "UNUSED / IRRELEVANT"
            status = "EXCLUDED"
            role_label = "Text/Metadata"
            reason = "Free text or web metadata unsuitable for time-series regression."
        elif pd.api.types.is_numeric_dtype(df[col]):
            category = "USEFUL"
            status = "KEEP"
            role_label = "Numeric Predictor"
            reason = "Continuous numerical exogenous feature (e.g., economic or environmental factor)."
        elif df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            category = "OPTIONAL"
            status = "KEEP"
            role_label = "Categorical Feature"
            reason = f"Categorical attribute with {n_unique} distinct classes."

        audit_records.append({
            "Column Name": col,
            "Type": col_type,
            "Unique Values": n_unique,
            "Missing %": missing_pct,
            "Detected Role": role_label,
            "Status": status,
            "Category": category,
            "Reason": reason,
        })

    audit_df = pd.DataFrame(audit_records)

    return {
        "detected_roles": detected_roles,
        "candidates": candidates,
        "audit_df": audit_df,
    }


def standardize_dataset(
    df: pd.DataFrame,
    detected_roles: Dict[str, Optional[str]],
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Create a clean standardized modeling dataframe from the raw dataset.
    Renames target to 'Sales', date to 'Date', etc., while retaining useful predictor columns.
    Original dataframe is never mutated.
    """
    std_df = df.copy()
    mapping_applied = {}

    date_col = detected_roles.get("date_col")
    sales_col = detected_roles.get("sales_col")

    if date_col and date_col in std_df.columns:
        if date_col != "Date":
            std_df.rename(columns={date_col: "Date"}, inplace=True)
            mapping_applied[date_col] = "Date"

    if sales_col and sales_col in std_df.columns:
        if sales_col != "Sales":
            std_df.rename(columns={sales_col: "Sales"}, inplace=True)
            mapping_applied[sales_col] = "Sales"

    order_col = detected_roles.get("order_col")
    if order_col and order_col in std_df.columns and order_col != "Order":
        std_df.rename(columns={order_col: "Order"}, inplace=True)
        mapping_applied[order_col] = "Order"

    # Drop explicit excluded columns if requested
    if exclude_columns:
        cols_to_drop = [c for c in exclude_columns if c in std_df.columns]
        std_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    return std_df, mapping_applied


def prepare_dataset_scope_timeseries(
    df: pd.DataFrame,
    date_col: str = "Date",
    sales_col: str = "Sales",
    order_col: Optional[str] = "Order",
) -> pd.DataFrame:
    """
    Construct a unified, dataset-scope time series aggregated strictly by Date.
    When an uploaded dataset contains multiple rows per date (multi-store, multi-product,
    or transactional), this function sums Sales and Orders and averages continuous exogenous
    features, guaranteeing numerical scale consistency between training, validation, and multi-step forecasting.
    """
    if date_col not in df.columns or sales_col not in df.columns:
        return df.copy()

    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
    work_df = work_df.dropna(subset=[date_col]).sort_values(date_col)

    # Check if multiple records exist per date
    total_dates = work_df[date_col].nunique()
    total_rows = len(work_df)

    # Determine aggregation rules for all available columns
    agg_rules = {sales_col: "sum"}
    if order_col and order_col in work_df.columns:
        agg_rules[order_col] = "sum"

    # Identify other columns
    numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in [sales_col, order_col, "Store", "Product", "Store_ID", "Product_ID", "Order_ID"]:
            continue
        norm_c = normalize_col_name(col)
        if any(h in norm_c for h in ["holiday", "discount", "promo", "weekend", "flag"]):
            agg_rules[col] = "max"
        else:
            agg_rules[col] = "mean"

    # Group by date
    grouped = work_df.groupby(date_col).agg(agg_rules).reset_index()
    grouped = grouped.sort_values(date_col).reset_index(drop=True)
    return grouped
