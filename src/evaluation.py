"""
Evaluation Compatibility Module.
Re-exports calculate_metrics, build_comparison_table, and diagnostics from src.validation.
"""

from src.validation import (
    calculate_metrics,
    build_comparison_table,
    generate_diagnostics,
    extract_feature_importance,
    get_human_metric_summary,
)

__all__ = [
    "calculate_metrics",
    "build_comparison_table",
    "generate_diagnostics",
    "extract_feature_importance",
    "get_human_metric_summary",
]
