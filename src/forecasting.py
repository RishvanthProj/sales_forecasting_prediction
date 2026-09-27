"""
Forecasting Compatibility Module.
Re-exports generate_recursive_forecast and compute_forecast_summary from src.forecast.
"""

from src.forecast import generate_recursive_forecast, compute_forecast_summary

__all__ = ["generate_recursive_forecast", "compute_forecast_summary"]
