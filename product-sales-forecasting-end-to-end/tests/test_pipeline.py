"""
Automated End-to-End Pipeline Integration Test.
Validates data loading, schema detection, cleaning, feature engineering,
model training, recursive forecasting, evaluation, and insights.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_csv, detect_schema, standardize_dataset
from src.data_profiler import generate_profile
from src.data_cleaner import clean_dataset
from src.statistical_analysis import run_statistical_analysis
from src.feature_engineering import FeaturePipeline
from src.model_training import chronological_split, train_models
from src.forecasting import generate_recursive_forecast
from src.evaluation import calculate_metrics, build_comparison_table
from src.insights import generate_business_insights


def test_sample_dataset_pipeline():
    print("\n--- TEST 1: Sample Sales Dataset Pipeline ---")
    sample_file = "data/sample/sample_sales.csv"
    assert os.path.exists(sample_file), f"File {sample_file} not found"

    df = load_csv(sample_file)
    print(f"Loaded sample dataset shape: {df.shape}")
    assert len(df) > 100

    # 1. Schema Detection
    meta = detect_schema(df)
    roles = meta["detected_roles"]
    print("Detected roles:", roles)
    assert roles["date_col"] == "Date"
    assert roles["sales_col"] == "Sales"
    assert roles["order_col"] == "Order"
    assert roles["store_col"] == "Store_id"

    # 2. Standardization
    std_df, _ = standardize_dataset(df, roles)
    assert "Date" in std_df.columns
    assert "Sales" in std_df.columns
    assert "Order" in std_df.columns

    # 3. Profiling
    profile = generate_profile(std_df, audit_df=meta["audit_df"])
    assert profile["total_rows"] == len(std_df)
    print("Profile generated. Memory:", profile["memory_mb"], "MB")

    # 4. Cleaning
    clean_df, stats = clean_dataset(std_df, outlier_treatment="Keep")
    assert stats["final_rows"] > 0
    print("Cleaning stats:", stats)

    # 5. Statistical Analysis
    stats_results = run_statistical_analysis(clean_df)
    print(f"Executed {len(stats_results)} statistical hypothesis tests.")
    assert len(stats_results) >= 2

    # 6. Feature Engineering & Leakage Prevention
    pipeline = FeaturePipeline(
        date_col="Date",
        sales_col="Sales",
        order_col="Order",
        store_col="Store_id",
        discount_col="Discount",
        holiday_col="Holiday",
        lags=[1, 7],
        rolling_windows=[7],
        categorical_cols=["Store_Type", "Location_Type", "Region_Code"],
    )
    feat_df = pipeline.fit_transform(clean_df)
    print(f"Engineered features: {len(pipeline.feature_columns)}")

    # Audit Leakage
    audit_report = pipeline.get_leakage_audit_report()
    assert not audit_report.empty
    assert (audit_report["Pipeline Status"] == "Included").all()
    print("Feature leakage audit passed!")

    # Verify shift(1) for rolling
    assert "rolling_Sales_7_mean" in feat_df.columns
    assert "lag_1_Sales" in feat_df.columns

    # 7. Chronological Split
    modeling_df = feat_df.dropna(subset=pipeline.feature_columns + ["Sales"]).reset_index(drop=True)
    train_df, val_df, test_df, split_info = chronological_split(modeling_df, train_pct=0.7, val_pct=0.15)
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0
    assert train_df["Date"].max() <= val_df["Date"].min()
    assert val_df["Date"].max() <= test_df["Date"].min()
    print("Chronological split verified:", split_info)

    # 8. Model Training
    models_to_train = ["Naive Last-Value", "Ridge Baseline", "Random Forest", "XGBoost"]
    train_out = train_models(
        train_df, val_df, test_df,
        feature_cols=pipeline.feature_columns,
        sales_col="Sales",
        order_col="Order",
        models_to_train=models_to_train,
    )
    assert train_out["has_orders"] is True
    print("Trained models:", list(train_out["pipelines"].keys()))

    # 9. Evaluation
    comp_df = build_comparison_table(train_out, split="test")
    print("\nModel Leaderboard:")
    print(comp_df[["Model", "Target", "MAE", "RMSE", "WAPE (%)"]])
    assert not comp_df.empty

    # 10. Recursive Multi-step Forecasting
    xgb_pipe = train_out["pipelines"]["XGBoost"]
    forecast_df = generate_recursive_forecast(
        historical_df=clean_df,
        pipeline=pipeline,
        model=xgb_pipe,
        horizon=7,
        entity_col="Store_id",
        entity_val=1,
    )
    print("\nForecasted 7-Day Horizon for Store 1:")
    print(forecast_df)
    assert len(forecast_df) == 7
    assert "Forecasted_Sales" in forecast_df.columns
    assert "Forecasted_Orders" in forecast_df.columns

    # 11. Commercial Business Insights
    insights = generate_business_insights(clean_df, forecast_df=forecast_df)
    assert len(insights) >= 3
    print(f"\nGenerated {len(insights)} commercial insights.")
    for ins in insights:
        print(f"[{ins['category']}] {ins['title']}: {ins['metric']}")

    print("\n--- TEST 1 PASSED SUCCESSFULLY! ---")


def test_sales_only_dataset():
    print("\n--- TEST 2: Arbitrary Sales-Only Minimal Dataset ---")
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sales = 1000 + 50 * np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 10, 100)
    df_minimal = pd.DataFrame({"Transaction_Date": dates, "Total_Revenue": sales})

    meta = detect_schema(df_minimal)
    assert meta["detected_roles"]["date_col"] == "Transaction_Date"
    assert meta["detected_roles"]["sales_col"] == "Total_Revenue"
    assert meta["detected_roles"]["order_col"] is None

    std_df, _ = standardize_dataset(df_minimal, meta["detected_roles"])
    clean_df, _ = clean_dataset(std_df)

    pipeline = FeaturePipeline(date_col="Date", sales_col="Sales", lags=[1, 7], rolling_windows=[7])
    feat_df = pipeline.fit_transform(clean_df)

    modeling_df = feat_df.dropna(subset=pipeline.feature_columns + ["Sales"]).reset_index(drop=True)
    train_df, val_df, test_df, _ = chronological_split(modeling_df, train_pct=0.7, val_pct=0.15)

    train_out = train_models(
        train_df, val_df, test_df,
        feature_cols=pipeline.feature_columns,
        sales_col="Sales",
        order_col=None,
        models_to_train=["Ridge Baseline", "XGBoost"],
    )
    assert train_out["has_orders"] is False
    comp_df = build_comparison_table(train_out, split="test")
    assert not comp_df.empty
    print(comp_df[["Model", "Target", "MAE", "WAPE (%)"]])

    forecast_df = generate_recursive_forecast(
        historical_df=clean_df,
        pipeline=pipeline,
        model=train_out["pipelines"]["XGBoost"],
        horizon=5,
        entity_col=None,
        entity_val=None,
    )
    assert len(forecast_df) == 5
    assert "Forecasted_Sales" in forecast_df.columns
    print("\nSales-Only Forecast:")
    print(forecast_df)
    print("\n--- TEST 2 PASSED SUCCESSFULLY! ---")


def test_optuna_tuning():
    print("\n--- TEST 3: Optuna Hyperparameter Optimization ---")
    from src.model_tuning import tune_model
    df = load_csv("data/sample/sample_sales.csv")
    meta = detect_schema(df)
    std_df, _ = standardize_dataset(df, meta["detected_roles"])
    pipeline = FeaturePipeline(date_col="Date", sales_col="Sales", order_col="Order", store_col="Store_id", lags=[1, 7], rolling_windows=[7])
    feat_df = pipeline.fit_transform(std_df).dropna(subset=pipeline.feature_columns + ["Sales"]).reset_index(drop=True)

    tune_res = tune_model(feat_df, pipeline.feature_columns, target_col="Sales", model_name="XGBoost", n_trials=3, n_splits=2)
    assert tune_res["status"] == "success"
    assert len(tune_res["best_params"]) > 0
    print(f"Optuna Best Score (CV WAPE): {tune_res['best_score']}")
    print(f"Discovered Best Params: {tune_res['best_params']}")
    print("\n--- TEST 3 PASSED SUCCESSFULLY! ---")


if __name__ == "__main__":
    test_sample_dataset_pipeline()
    test_sales_only_dataset()
    test_optuna_tuning()
    print("\n==========================================")
    print("ALL 3 AUTOMATED TEST SUITES PASSED!")
    print("==========================================")

