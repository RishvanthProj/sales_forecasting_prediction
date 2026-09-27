"""
Data Loader & Schema Compatibility Module.
Re-exports functions from src.loader and src.schema_detector for backward compatibility.
"""

from src.loader import load_csv, detect_frequency, get_available_samples
from src.schema_detector import detect_schema, standardize_dataset, normalize_col_name, CANDIDATE_PATTERNS

__all__ = [
    "load_csv",
    "detect_frequency",
    "get_available_samples",
    "detect_schema",
    "standardize_dataset",
    "normalize_col_name",
    "CANDIDATE_PATTERNS",
]
