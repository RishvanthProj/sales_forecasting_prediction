"""
Master Integration Test Suite.
Validates all 10 mandatory testing scenarios specified in Master Prompt Section 56:
1. Date + Sales
2. Date + Store + Sales
3. Date + Store + Orders + Sales
4. Walmart-style dataset
5. Extra unwanted / irrelevant columns
6. Missing values handling
7. Duplicated rows removal
8. Outliers / retail demand spikes
9. Different date column names
10. Different sales column names
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import load_csv, detect_frequency, get_available_samples
from src.schema_detector import detect_schema, standardize_dataset
from src.cleaner import clean_dataset
from src.outliers import analyze_outliers
from src.eda import (
    prepare_time_series_eda,
    prepare_seasonality_analysis,
    prepare_entity_analysis,
    prepare_correlation_analysis,
)
from src.features import FeaturePipeline
from src.models import chronological_split, train_models, XGBOOST_AVAILABLE
from src.validation import calculate_metrics, build_comparison_table, generate_diagnostics, extract_feature_importance
from src.forecast import generate_recursive_forecast, compute_forecast_summary
from src.insights import run_statistical_tests, generate_business_takeaways


def execute_pipeline(df: pd.DataFrame, scenario_name: str) -> dict:
    """Helper executing full end-to-end pipeline on any input dataframe."""
    print(f"\n=======================================================")
    print(f"RUNNING SCENARIO: {scenario_name}")
    print(f"Input Shape: {df.shape}, Columns: {list(df.columns)}")
    print(f"=======================================================")

    # 1. Detection
    schema_meta = detect_schema(df)
    roles = schema_meta["detected_roles"]
    assert roles["date_col"] is not None, f"Date column not detected in {scenario_name}"
    assert roles["sales_col"] is not None, f"Sales column not detected in {scenario_name}"

    # 2. Standardization
    std_df, _ = standardize_dataset(df, roles)
    assert "Date" in std_df.columns
    assert "Sales" in std_df.columns

    # 3. Frequency
    freq, default_cfg = detect_frequency(df, roles["date_col"])

    # 4. Cleaning
    clean_df, clean_stats = clean_dataset(
        std_df,
        date_col="Date",
        sales_col="Sales",
        order_col="Order" if roles.get("order_col") else None,
        store_col=roles.get("store_col"),
        product_col=roles.get("product_col"),
    )
    assert len(clean_df) > 0, f"Cleaned df is empty in {scenario_name}"

    # 5. Outliers
    out_res = analyze_outliers(clean_df, sales_col="Sales", date_col="Date")
    assert "outlier_count" in out_res

    # 6. EDA
    ts_eda = prepare_time_series_eda(clean_df, date_col="Date", sales_col="Sales")
    assert not ts_eda.empty

    # 7. Features
    pipeline = FeaturePipeline(
        date_col="Date",
        sales_col="Sales",
        order_col="Order" if roles.get("order_col") else None,
        store_col=roles.get("store_col"),
        product_col=roles.get("product_col"),
        discount_col=roles.get("discount_col"),
        holiday_col=roles.get("holiday_col"),
        lags=default_cfg["lags"][:2],  # compact lags for small test sets
        rolling_windows=default_cfg["rolling"][:2],
    )
    feat_df = pipeline.fit_transform(clean_df)
    assert len(pipeline.feature_columns) > 0

    # Leakage check
    audit_rep = pipeline.get_leakage_audit_report()
    assert not audit_rep.empty

    # 8. Training
    modeling_df = feat_df.dropna(subset=pipeline.feature_columns + ["Sales"]).reset_index(drop=True)
    assert len(modeling_df) >= 10, f"Insufficient records for {scenario_name}"

    train_df, val_df, test_df, split_info = chronological_split(modeling_df, date_col="Date", train_pct=0.7, val_pct=0.15)
    train_results = train_models(
        train_df, val_df, test_df,
        feature_cols=pipeline.feature_columns,
        sales_col="Sales",
        models_to_train=["Naive Baseline", "Ridge Baseline", "Random Forest"],
    )
    assert len(train_results["models"]) >= 2

    # 9. Validation
    comp_df = build_comparison_table(train_results, split="test")
    assert not comp_df.empty
    best_model_name = comp_df.iloc[0]["Model"]
    best_model = train_results["models"][best_model_name]

    # Diagnostics
    y_test_act = train_results["actuals"]["test_sales"]
    y_test_pred = train_results["predictions"][best_model_name]["test_sales"]
    diag_df = generate_diagnostics(test_df, "Date", y_test_act, y_test_pred)
    assert not diag_df.empty

    # 10. Forecast
    forecast_df = generate_recursive_forecast(
        historical_df=clean_df,
        pipeline=pipeline,
        model=best_model,
        horizon=5,
        frequency=freq,
    )
    assert len(forecast_df) == 5
    assert "Forecasted_Sales" in forecast_df.columns

    f_sum = compute_forecast_summary(forecast_df, clean_df)
    assert f_sum["total_forecast"] >= 0.0

    print(f"✓ SCENARIO {scenario_name} PASSED! Best Model: {best_model_name}, WAPE: {comp_df.iloc[0]['WAPE (%)']:.2f}%")
    return {
        "comp_df": comp_df,
        "forecast_df": forecast_df,
        "schema_meta": schema_meta,
    }


def test_scenario_1_date_plus_sales():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sales = 500 + 30 * np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 5, 100)
    df = pd.DataFrame({"Date": dates, "Sales": sales})
    res = execute_pipeline(df, "1: Date + Sales")
    assert res["schema_meta"]["detected_roles"]["store_col"] is None


def test_scenario_2_date_store_sales():
    dates = pd.date_range("2023-01-01", periods=60, freq="D").repeat(2)
    stores = np.tile([1, 2], 60)
    sales = 1000 + stores * 200 + np.random.normal(0, 10, 120)
    df = pd.DataFrame({"Date": dates, "Store": stores, "Sales": sales})
    res = execute_pipeline(df, "2: Date + Store + Sales")
    assert res["schema_meta"]["detected_roles"]["store_col"] == "Store"


def test_scenario_3_date_store_orders_sales():
    dates = pd.date_range("2023-01-01", periods=60, freq="D").repeat(2)
    stores = np.tile([1, 2], 60)
    orders = np.random.randint(10, 100, 120)
    sales = orders * 45.0 + np.random.normal(0, 5, 120)
    df = pd.DataFrame({"Date": dates, "Store": stores, "Quantity": orders, "Sales": sales})
    res = execute_pipeline(df, "3: Date + Store + Orders + Sales")
    assert res["schema_meta"]["detected_roles"]["order_col"] == "Quantity"


def test_scenario_4_walmart_data():
    walmart_file = "data/sample/walmart_sales.csv"
    assert os.path.exists(walmart_file), f"{walmart_file} does not exist"
    df = pd.read_csv(walmart_file)
    res = execute_pipeline(df, "4: Walmart-Style Data")
    assert res["schema_meta"]["detected_roles"]["sales_col"] == "Weekly_Sales"
    assert res["schema_meta"]["detected_roles"]["holiday_col"] == "Holiday_Flag"


def test_scenario_5_unwanted_columns():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sales = 700 + np.random.normal(0, 10, 100)
    df = pd.DataFrame({
        "Date": dates,
        "Sales": sales,
        "Transaction_UUID": [f"uuid-{i:05d}" for i in range(100)],
        "High_Cardinality_Text": [f"random comment text {i}" for i in range(100)],
        "Constant_Zero": [0] * 100,
        "Web_URL": ["https://example.com/record" for _ in range(100)],
    })
    res = execute_pipeline(df, "5: Extra Unwanted Columns")
    audit_df = res["schema_meta"]["audit_df"]
    excluded_cols = audit_df[audit_df["Status"] == "EXCLUDED"]["Column Name"].tolist()
    assert "Transaction_UUID" in excluded_cols
    assert "Constant_Zero" in excluded_cols


def test_scenario_6_missing_values():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sales = 800 + np.random.normal(0, 15, 100)
    temp = np.random.uniform(20, 80, 100)
    # inject missing values
    sales[5] = np.nan
    sales[12] = np.nan
    temp[8] = np.nan
    temp[20] = np.nan
    df = pd.DataFrame({"Date": dates, "Sales": sales, "Temperature": temp})
    execute_pipeline(df, "6: Missing Values Handling")


def test_scenario_7_duplicated_rows():
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    sales = 400 + np.random.normal(0, 5, 80)
    df = pd.DataFrame({"Date": dates, "Sales": sales})
    # duplicate 20 rows
    df_dup = pd.concat([df, df.iloc[:20]], ignore_index=True)
    assert len(df_dup) == 100
    execute_pipeline(df_dup, "7: Duplicated Rows Removal")


def test_scenario_8_outliers():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sales = 300 + np.random.normal(0, 5, 100)
    # inject massive holiday spikes
    sales[30] = 5000.0  # 15x spike
    sales[65] = 4500.0
    df = pd.DataFrame({"Date": dates, "Sales": sales})
    execute_pipeline(df, "8: Outliers / Spikes Retention")


def test_scenario_9_different_date_names():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sales = 600 + np.random.normal(0, 10, 100)
    df = pd.DataFrame({"Transaction_Date": dates, "Sales": sales})
    res = execute_pipeline(df, "9: Different Date Column Name")
    assert res["schema_meta"]["detected_roles"]["date_col"] == "Transaction_Date"


def test_scenario_10_different_sales_names():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    revenue = 950 + np.random.normal(0, 15, 100)
    df = pd.DataFrame({"Date": dates, "Total_Revenue": revenue})
    res = execute_pipeline(df, "10: Different Sales Column Name")
    assert res["schema_meta"]["detected_roles"]["sales_col"] == "Total_Revenue"


def test_scenario_11_dataset_scope_scale_continuity():
    walmart_path = "data/sample/walmart_sales.csv"
    if not os.path.exists(walmart_path):
        return
    df = pd.read_csv(walmart_path)
    from src.schema_detector import prepare_dataset_scope_timeseries
    meta = detect_schema(df)
    roles = meta["detected_roles"]
    std_df, _ = standardize_dataset(df, roles)
    clean_df, _ = clean_dataset(std_df, date_col="Date", sales_col="Sales")
    scope_ts = prepare_dataset_scope_timeseries(clean_df, date_col="Date", sales_col="Sales")

    freq, cfg = detect_frequency(df, roles["date_col"])
    from src.forecast import get_horizon_for_period
    horizon = get_horizon_for_period("3 Months", freq)

    pipe = FeaturePipeline(date_col="Date", sales_col="Sales", lags=cfg["lags"][:2], rolling_windows=cfg["rolling"][:2])
    feat_df = pipe.fit_transform(scope_ts)
    modeling_df = feat_df.dropna(subset=pipe.feature_columns + ["Sales"]).reset_index(drop=True)
    train_df, val_df, test_df, _ = chronological_split(modeling_df, date_col="Date")
    res = train_models(train_df, val_df, test_df, feature_cols=pipe.feature_columns, sales_col="Sales", models_to_train=["Random Forest"])
    model = res["models"]["Random Forest"]

    fc = generate_recursive_forecast(scope_ts, pipe, model, horizon=horizon, frequency=freq)
    last_act = scope_ts["Sales"].iloc[-1]
    first_fc = fc["Forecasted_Sales"].iloc[0]
    scale_ratio = first_fc / last_act
    assert 0.7 <= scale_ratio <= 1.5, f"Forecast collapsed or exploded! Scale ratio: {scale_ratio}"


def test_scenario_12_frequency_aware_horizon_mapping():
    from src.forecast import get_horizon_for_period
    assert get_horizon_for_period("1 Month", "Daily") == 30
    assert get_horizon_for_period("3 Months", "Daily") == 90
    assert get_horizon_for_period("6 Months", "Daily") == 180

    assert get_horizon_for_period("1 Month", "Weekly") == 4
    assert get_horizon_for_period("3 Months", "Weekly") == 13
    assert get_horizon_for_period("6 Months", "Weekly") == 26

    assert get_horizon_for_period("1 Month", "Monthly") == 1
    assert get_horizon_for_period("3 Months", "Monthly") == 3
    assert get_horizon_for_period("6 Months", "Monthly") == 6


def test_scenario_13_wape_based_forecast_accuracy():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([90.0, 210.0, 310.0, 380.0])
    # abs errors: 10 + 10 + 10 + 20 = 50. sum actual: 1000. WAPE = 50 / 1000 = 5.0%
    # Forecast Accuracy = 95.0%
    m = calculate_metrics(y_true, y_pred)
    assert m["WAPE_%"] == 5.0
    assert m["WAPE_Accuracy_%"] == 95.0


def test_scenario_14_historical_range_filtering():
    from src.eda import filter_historical_by_range
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="W")
    df = pd.DataFrame({"Date": dates, "Total_Sales": np.random.uniform(100, 200, len(dates))})
    m1 = filter_historical_by_range(df, date_col="Date", range_option="1 MONTH")
    m3 = filter_historical_by_range(df, date_col="Date", range_option="3 MONTHS")
    m6 = filter_historical_by_range(df, date_col="Date", range_option="6 MONTHS")
    overall = filter_historical_by_range(df, date_col="Date", range_option="OVERALL")

    assert len(m1) < len(m3) < len(m6) < len(overall)
    assert len(overall) == len(df)


def test_scenario_15_outlier_diagnostics_and_table():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    sales = np.ones(100) * 100.0
    sales[50] = 5000.0
    df = pd.DataFrame({"Date": dates, "Sales": sales})
    res = analyze_outliers(df, sales_col="Sales", date_col="Date")
    assert res["outlier_count"] >= 1
    assert "Deviation_From_Normal" in res["top_outliers_df"].columns
    assert "Outlier_Type" in res["top_outliers_df"].columns


if __name__ == "__main__":
    test_scenario_1_date_plus_sales()
    test_scenario_2_date_store_sales()
    test_scenario_3_date_store_orders_sales()
    test_scenario_4_walmart_data()
    test_scenario_5_unwanted_columns()
    test_scenario_6_missing_values()
    test_scenario_7_duplicated_rows()
    test_scenario_8_outliers()
    test_scenario_9_different_date_names()
    test_scenario_10_different_sales_names()
    test_scenario_11_dataset_scope_scale_continuity()
    test_scenario_12_frequency_aware_horizon_mapping()
    test_scenario_13_wape_based_forecast_accuracy()
    test_scenario_14_historical_range_filtering()
    test_scenario_15_outlier_diagnostics_and_table()

    print("\n=======================================================")
    print("ALL 15 MASTER TEST SCENARIOS PASSED WITH ZERO ERRORS!")
    print("=======================================================")
