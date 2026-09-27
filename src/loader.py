"""
Data Loader Module.
Handles safe CSV ingestion, encoding resilience, frequency detection,
and sample dataset loading.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os
import io
import pandas as pd
import numpy as np


def load_csv(file_input: Union[str, io.BytesIO, io.StringIO, Any]) -> pd.DataFrame:
    """
    Safely load a CSV dataset from a file path or file-like object.
    Automatically handles common encodings and separators.
    """
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    separators = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in separators:
            try:
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                df = pd.read_csv(file_input, sep=sep, encoding=enc)
                if df.shape[1] > 1 and len(df) > 0:
                    return df
            except Exception:
                continue

    # Fallback to standard pandas reader
    if hasattr(file_input, "seek"):
        file_input.seek(0)
    df = pd.read_csv(file_input)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_frequency(df: pd.DataFrame, date_col: str) -> Tuple[str, Dict[str, List[int]]]:
    """
    Detect dataset temporal frequency (Daily, Weekly, Monthly, or Irregular)
    and provide appropriate default lag & rolling window configurations.
    """
    if date_col not in df.columns or df.empty:
        return "Daily", {"lags": [1, 7, 14, 28], "rolling": [7, 14, 30]}

    try:
        sample_dates = pd.to_datetime(df[date_col].dropna().drop_duplicates()).sort_values()
        if len(sample_dates) < 3:
            return "Daily", {"lags": [1, 7], "rolling": [7]}

        diffs = sample_dates.diff().dt.total_seconds() / 86400.0
        median_diff = float(diffs.median())

        if 0.6 <= median_diff <= 1.5:
            freq = "Daily"
            defaults = {"lags": [1, 7, 14, 28], "rolling": [7, 14, 30]}
        elif 5.5 <= median_diff <= 8.5:
            freq = "Weekly"
            defaults = {"lags": [1, 2, 4, 8], "rolling": [4, 8, 12]}
        elif 25.0 <= median_diff <= 33.0:
            freq = "Monthly"
            defaults = {"lags": [1, 2, 3, 6], "rolling": [3, 6, 12]}
        else:
            freq = "Irregular"
            defaults = {"lags": [1, 2, 3], "rolling": [3, 7]}

        return freq, defaults
    except Exception:
        return "Daily", {"lags": [1, 7, 14, 28], "rolling": [7, 14, 30]}


def get_available_samples() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of available built-in sample datasets."""
    samples = {}
    walmart_path = "data/sample/walmart_sales.csv"
    if os.path.exists(walmart_path):
        samples["walmart"] = {
            "name": "Walmart Sales (Store & Macroeconomic Data)",
            "path": walmart_path,
            "description": "Weekly store-level sales across 45 stores with CPI, Fuel Price, Temperature, and Holiday indicators.",
            "target": "Weekly_Sales",
            "frequency": "Weekly",
        }

    retail_path = "data/sample/sample_sales.csv"
    if os.path.exists(retail_path):
        samples["retail"] = {
            "name": "Multi-Store Retail Sales & Orders",
            "path": retail_path,
            "description": "Daily store transactions with Order demand volumes, promotional discounts, and regional attributes.",
            "target": "Sales",
            "frequency": "Daily",
        }

    desktop_path = "/Users/rishvantha/Desktop/product_sales_dataset_final 2.csv"
    if os.path.exists(desktop_path):
        samples["desktop"] = {
            "name": "Desktop Product Sales (200k Transactions)",
            "path": desktop_path,
            "description": "200,000 transaction records with Order_Date, Quantity, Revenue, Profit, Category, and Region.",
            "target": "Revenue",
            "frequency": "Daily",
        }

    return samples
