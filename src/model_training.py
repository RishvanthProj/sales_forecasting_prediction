"""
Model Training Compatibility Module.
Re-exports chronological_split, train_models, and helpers from src.models.
"""

from src.models import (
    chronological_split,
    train_models,
    build_regressor,
    NaiveForecaster,
    ModelWrapper,
    XGBOOST_AVAILABLE,
)

__all__ = [
    "chronological_split",
    "train_models",
    "build_regressor",
    "NaiveForecaster",
    "ModelWrapper",
    "XGBOOST_AVAILABLE",
]
