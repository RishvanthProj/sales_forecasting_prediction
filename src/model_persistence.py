"""
Model Persistence Module.
Handles serialization and deserialization of trained ML pipelines, feature configurations,
and evaluation metrics to the local artifacts/ directory using joblib.
"""

from typing import Dict, Any, Optional, List
import os
import json
import time
import joblib
import pandas as pd
import numpy as np


ARTIFACTS_DIR = "artifacts"
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
PREPROCESSORS_DIR = os.path.join(ARTIFACTS_DIR, "preprocessors")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")


def _ensure_dirs():
    """Ensure all artifact subdirectories exist."""
    for d in [MODELS_DIR, PREPROCESSORS_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)


def save_training_artifacts(
    training_results: Dict[str, Any],
    feature_pipeline,
    dataset_name: str = "session",
) -> Dict[str, str]:
    """
    Persist trained model pipelines, feature pipeline, and metrics to disk.

    Returns a dict of saved file paths.
    """
    _ensure_dirs()
    timestamp = int(time.time())
    safe_name = dataset_name.replace(" ", "_").replace("/", "_").replace(".", "_")[:40]
    run_id = f"{safe_name}_{timestamp}"

    saved_paths = {}

    # 1. Save Feature Engineering Pipeline (FeaturePipeline object)
    pipeline_path = os.path.join(PREPROCESSORS_DIR, f"feature_pipeline_{run_id}.joblib")
    joblib.dump(feature_pipeline, pipeline_path)
    saved_paths["feature_pipeline"] = pipeline_path

    # 2. Save each trained model pipeline
    for model_name, model_obj in training_results.get("pipelines", {}).items():
        safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        model_path = os.path.join(MODELS_DIR, f"{safe_model_name}_{run_id}.joblib")
        joblib.dump(model_obj, model_path)
        saved_paths[f"model_{model_name}"] = model_path

    # 3. Save evaluation metrics as JSON
    metrics_record = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "timestamp": timestamp,
        "has_orders": training_results.get("has_orders", False),
        "feature_count": len(feature_pipeline.feature_columns) if feature_pipeline else 0,
        "training_times": training_results.get("training_times", {}),
    }
    metrics_path = os.path.join(METRICS_DIR, f"metrics_{run_id}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_record, f, indent=2, default=str)
    saved_paths["metrics"] = metrics_path

    return saved_paths


def load_latest_artifacts(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the most recently saved model artifacts from disk.
    Optionally filter by dataset name prefix.

    Returns a dict with 'feature_pipeline', 'pipelines', and 'metadata' keys.
    If nothing is found, returns an empty dict.
    """
    _ensure_dirs()

    # Find latest metrics file
    metric_files = [
        f for f in os.listdir(METRICS_DIR)
        if f.endswith(".json") and f.startswith("metrics_")
    ]
    if not metric_files:
        return {}

    # Filter by dataset name if provided
    if dataset_name:
        safe_name = dataset_name.replace(" ", "_").replace("/", "_").replace(".", "_")[:40]
        metric_files = [f for f in metric_files if safe_name in f]

    if not metric_files:
        return {}

    # Sort by timestamp (embedded in filename)
    metric_files.sort(reverse=True)
    latest_metric_file = metric_files[0]

    with open(os.path.join(METRICS_DIR, latest_metric_file), "r") as f:
        metadata = json.load(f)

    run_id = metadata["run_id"]

    # Load feature pipeline
    pipeline_path = os.path.join(PREPROCESSORS_DIR, f"feature_pipeline_{run_id}.joblib")
    if not os.path.exists(pipeline_path):
        return {}

    feature_pipeline = joblib.load(pipeline_path)

    # Load all model pipelines for this run
    pipelines = {}
    model_files = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".joblib") and run_id in f
    ]
    for mf in model_files:
        # Reconstruct model name from filename
        # e.g. XGBoost_session_1234567890.joblib → XGBoost
        model_name_raw = mf.replace(f"_{run_id}.joblib", "").replace("_", " ").strip()
        # Handle known model names
        for known in ["XGBoost", "Random Forest", "Ridge Baseline", "Naive Last-Value",
                      "Seasonal Naive Lag 7", "LightGBM"]:
            if known.replace(" ", "_").replace("(", "").replace(")", "") in mf:
                model_name_raw = known
                break
        model_obj = joblib.load(os.path.join(MODELS_DIR, mf))
        pipelines[model_name_raw] = model_obj

    return {
        "feature_pipeline": feature_pipeline,
        "pipelines": pipelines,
        "metadata": metadata,
        "run_id": run_id,
    }


def list_saved_runs() -> pd.DataFrame:
    """
    List all saved model runs with their metadata as a DataFrame.
    """
    _ensure_dirs()
    records = []
    for fname in sorted(os.listdir(METRICS_DIR), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(METRICS_DIR, fname), "r") as f:
                meta = json.load(f)
            records.append({
                "Run ID": meta.get("run_id", fname),
                "Dataset": meta.get("dataset_name", "Unknown"),
                "Features": meta.get("feature_count", 0),
                "Has Orders": meta.get("has_orders", False),
                "Saved At": pd.to_datetime(meta.get("timestamp", 0), unit="s").strftime("%Y-%m-%d %H:%M"),
            })
        except Exception:
            continue
    return pd.DataFrame(records)
