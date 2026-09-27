"""
Product Sales Forecasting ML Dashboard.
Pure Machine Learning Standalone Application.
Provides leakage-free time-series forecasting, automated schema understanding,
robust outlier diagnostics, model validation, and commercial business takeaways.
Designed with a dark quantitative visual aesthetic.
"""

import os
import io
import time
import pandas as pd
import numpy as np
import textwrap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Backend Modules
from src.loader import load_csv, detect_frequency, get_available_samples
from src.schema_detector import detect_schema, standardize_dataset, prepare_dataset_scope_timeseries
from src.cleaner import clean_dataset
from src.outliers import analyze_outliers
from src.eda import (
    prepare_time_series_eda,
    filter_historical_by_range,
    compute_historical_summary_kpis,
    compute_period_sales_summaries,
    prepare_seasonality_analysis,
    prepare_entity_analysis,
    prepare_correlation_analysis,
    prepare_discount_analysis,
    prepare_holiday_analysis,
)
from src.features import FeaturePipeline
from src.models import chronological_split, train_models, XGBOOST_AVAILABLE
from src.validation import (
    calculate_metrics,
    build_comparison_table,
    generate_diagnostics,
    extract_feature_importance,
    get_human_metric_summary,
    get_accuracy_calculation_explanation,
)
from src.forecast import generate_recursive_forecast, compute_forecast_summary, get_horizon_for_period
from src.insights import run_statistical_tests, generate_business_takeaways
from src.ui_theme import (
    DARK_THEME_CSS,
    apply_dark_theme,
    COLOR_PRIMARY,
    COLOR_CYAN,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_WARNING,
    COLOR_TEXT,
    COLOR_MUTED,
    COLOR_BORDER,
    COLOR_PANEL,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Product Sales AI — ML Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global Formatting Helpers for Clean Executive Presentation
# -----------------------------------------------------------------------------
def fmt_currency(val: float, curr: str = "$") -> str:
    """Format large currency values cleanly: e.g. $709.8M, $45.5M, $1.32B."""
    if val is None or pd.isna(val):
        return f"{curr}0.00"
    try:
        abs_v = abs(float(val))
    except Exception:
        return f"{curr}0.00"
    sign = "-" if float(val) < 0 else ""
    if abs_v >= 1e9:
        return f"{sign}{curr}{abs_v / 1e9:.2f}B"
    elif abs_v >= 1e6:
        return f"{sign}{curr}{abs_v / 1e6:.1f}M"
    elif abs_v >= 1e3:
        return f"{sign}{curr}{abs_v / 1e3:.0f}K"
    return f"{sign}{curr}{abs_v:,.2f}"


def fmt_date_readable(val) -> str:
    """Format date strings into readable human dates: e.g. Dec 23, 2012."""
    if not val or val == "N/A":
        return "N/A"
    try:
        dt = pd.to_datetime(val)
        return dt.strftime("%b %d, %Y")
    except Exception:
        return str(val)[:10]


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session():
    defaults = {
        "raw_df": None,
        "clean_df": None,
        "std_df": None,
        "scope_ts_df": None,
        "dataset_name": "Walmart Sales (Store & Macroeconomic Data)",
        "dataset_source": "walmart_sample",
        "schema_meta": None,
        "frequency": "Weekly",
        "default_lags": [1, 2, 4, 8],
        "default_rolling": [4, 8, 12],
        "pipeline": None,
        "feature_df": None,
        "split_info": None,
        "training_results": None,
        "selected_model": "XGBoost" if XGBOOST_AVAILABLE else "Random Forest",
        "forecast_period_choice": "3 Months",
        "forecast_horizon": 13,
        "forecast_df": None,
        "forecast_summary": None,
        "cleaning_stats": None,
        "outlier_analysis": None,
        "manual_overrides": {},
        "active_currency": "$",
        "historical_range_overview": "OVERALL",
        "historical_range_forecast": "OVERALL",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# -----------------------------------------------------------------------------
# End-to-End Pipeline Execution
# -----------------------------------------------------------------------------
def run_full_pipeline(df: pd.DataFrame, dataset_name: str, overrides: dict = None):
    """Execute end-to-end ML pipeline with caching in session state."""
    with st.spinner(f"Analyzing {dataset_name} and training forecasting models..."):
        # Strip column whitespace
        df.columns = [str(c).strip() for c in df.columns]
        st.session_state["raw_df"] = df
        st.session_state["dataset_name"] = dataset_name

        # 1. Schema Detection
        schema_meta = detect_schema(df)
        roles = schema_meta["detected_roles"].copy()
        if overrides:
            for r_key, r_col in overrides.items():
                if r_col != "None" and r_col in df.columns:
                    roles[r_key] = r_col
                elif r_col == "None":
                    roles[r_key] = None

        st.session_state["schema_meta"] = schema_meta
        st.session_state["detected_roles"] = roles

        date_col_raw = roles.get("date_col")
        sales_col_raw = roles.get("sales_col")

        if not date_col_raw or not sales_col_raw:
            st.error("Error: Could not detect a valid Date column or Sales target column in dataset.")
            return False

        # Detect Frequency
        freq, default_cfg = detect_frequency(df, date_col_raw)
        st.session_state["frequency"] = freq
        st.session_state["default_lags"] = default_cfg["lags"]
        st.session_state["default_rolling"] = default_cfg["rolling"]

        # 2. Standardization
        std_df, _ = standardize_dataset(df, roles)
        st.session_state["std_df"] = std_df

        # 3. Safe Cleaning (Preserves original outliers)
        store_col = roles.get("store_col")
        product_col = roles.get("product_col")
        clean_df, clean_stats = clean_dataset(
            std_df,
            date_col="Date",
            sales_col="Sales",
            order_col="Order" if roles.get("order_col") else None,
            store_col=store_col,
            product_col=product_col,
            negative_handling="Clip to Zero",
            outlier_treatment="Keep",  # Strictly preserve real sales spikes
            impute_missing=True,
        )
        st.session_state["clean_df"] = clean_df
        st.session_state["cleaning_stats"] = clean_stats

        # 4. Outlier Diagnostics
        outlier_res = analyze_outliers(
            clean_df,
            sales_col="Sales",
            date_col="Date",
            store_col=store_col,
            product_col=product_col,
        )
        st.session_state["outlier_analysis"] = outlier_res

        # 5. Dataset-Scope Aggregated Time Series
        # This guarantees mathematical scale continuity between historical sales and multi-step forecast!
        scope_ts_df = prepare_dataset_scope_timeseries(
            clean_df,
            date_col="Date",
            sales_col="Sales",
            order_col="Order" if "Order" in clean_df.columns else None,
        )
        st.session_state["scope_ts_df"] = scope_ts_df

        # 6. Leakage-Free Feature Engineering at Dataset Scope
        pipeline = FeaturePipeline(
            date_col="Date",
            sales_col="Sales",
            order_col="Order" if "Order" in scope_ts_df.columns else None,
            lags=default_cfg["lags"],
            rolling_windows=default_cfg["rolling"],
        )

        feat_df = pipeline.fit_transform(scope_ts_df)
        st.session_state["pipeline"] = pipeline
        st.session_state["feature_df"] = feat_df

        # 7. Chronological Split & ML Model Training
        modeling_df = feat_df.dropna(subset=pipeline.feature_columns + ["Sales"]).reset_index(drop=True)
        if len(modeling_df) < 10:
            st.error("Error: Insufficient historical records remain after lag feature generation.")
            return False

        train_df, val_df, test_df, split_info = chronological_split(
            modeling_df, date_col="Date", train_pct=0.70, val_pct=0.15
        )
        st.session_state["split_info"] = split_info

        # Train ML models and baselines for leaderboard comparison
        models_to_train = ["Naive Baseline", "Ridge Baseline", "Random Forest"]
        if XGBOOST_AVAILABLE:
            models_to_train.append("XGBoost")

        train_results = train_models(
            train_df, val_df, test_df,
            feature_cols=pipeline.feature_columns,
            sales_col="Sales",
            models_to_train=models_to_train,
        )
        st.session_state["training_results"] = train_results

        # Champion model selection (Only genuine ML models: XGBoost preferred, Random Forest fallback)
        if XGBOOST_AVAILABLE:
            st.session_state["selected_model"] = "XGBoost"
        else:
            st.session_state["selected_model"] = "Random Forest"

        # 8. Generate Default Multi-Step Forecast (3 Months by default)
        default_period = "3 Months"
        horizon = get_horizon_for_period(default_period, freq)
        st.session_state["forecast_period_choice"] = default_period
        st.session_state["forecast_horizon"] = horizon

        chosen_pipe = train_results["models"].get(st.session_state["selected_model"])
        if chosen_pipe:
            forecast_df = generate_recursive_forecast(
                historical_df=scope_ts_df,
                pipeline=pipeline,
                model=chosen_pipe,
                horizon=horizon,
                frequency=freq,
            )
            st.session_state["forecast_df"] = forecast_df
            st.session_state["forecast_summary"] = compute_forecast_summary(forecast_df, scope_ts_df)

        return True


# Auto-load default dataset on first start
if st.session_state["raw_df"] is None:
    sample_path = "data/sample/walmart_sales.csv"
    if os.path.exists(sample_path):
        df_walmart = pd.read_csv(sample_path)
        run_full_pipeline(df_walmart, "Walmart Sales (Store & Macroeconomic Data)")
    else:
        alt_path = "data/sample/sample_sales.csv"
        if os.path.exists(alt_path):
            df_retail = pd.read_csv(alt_path)
            run_full_pipeline(df_retail, "Multi-Store Retail Sales & Orders")


# -----------------------------------------------------------------------------
# TOP APPLICATION HEADER
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([7, 4])

raw_df = st.session_state["raw_df"]
clean_df = st.session_state["clean_df"]
scope_ts_df = st.session_state.get("scope_ts_df")
roles = st.session_state.get("detected_roles", {})
currency = st.session_state["active_currency"]

date_span = "N/A"
if clean_df is not None and "Date" in clean_df.columns:
    d_min = str(clean_df["Date"].min())[:10]
    d_max = str(clean_df["Date"].max())[:10]
    date_span = f"{d_min} → {d_max}"

row_count = f"{len(raw_df):,}" if raw_df is not None else "0"
model_tag = st.session_state.get("selected_model", "XGBoost").upper()
dataset_tag = st.session_state.get("dataset_name", "CUSTOM CSV").upper()
freq_tag = st.session_state.get("frequency", "DAILY").upper()

with header_col1:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-left">
                <span class="app-header-title">PRODUCT SALES FORECASTING</span>
                <span class="badge badge-blue">MODEL: {model_tag}</span>
                <span class="badge badge-gray">SCOPE: ENTIRE UPLOADED DATASET</span>
                <span class="badge badge-gray">FREQ: {freq_tag}</span>
            </div>
            <div class="app-header-meta">
                DATASET: <b>{dataset_tag}</b> &nbsp;|&nbsp; SPAN: <b>{date_span}</b> &nbsp;|&nbsp; ROWS: <b>{row_count}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    desktop_exists = os.path.exists("/Users/rishvantha/Desktop/product_sales_dataset_final 2.csv")
    btn_cols = st.columns(4 if desktop_exists else 3)

    with btn_cols[0]:
        if st.button("Walmart", use_container_width=True):
            w_path = "data/sample/walmart_sales.csv"
            if os.path.exists(w_path):
                df_w = pd.read_csv(w_path)
                run_full_pipeline(df_w, "Walmart Sales (Store & Macroeconomic Data)")
                st.rerun()

    with btn_cols[1]:
        if st.button("Retail", use_container_width=True):
            r_path = "data/sample/sample_sales.csv"
            if os.path.exists(r_path):
                df_r = pd.read_csv(r_path)
                run_full_pipeline(df_r, "Multi-Store Retail Sales & Orders")
                st.rerun()

    btn_idx = 2
    if desktop_exists:
        with btn_cols[btn_idx]:
            if st.button("Desktop CSV", use_container_width=True):
                df_desk = pd.read_csv("/Users/rishvantha/Desktop/product_sales_dataset_final 2.csv")
                run_full_pipeline(df_desk, "Desktop Product Sales (200k Records)")
                st.rerun()
        btn_idx += 1

    with btn_cols[btn_idx]:
        with st.popover("Upload CSV"):
            uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])
            if uploaded_file is not None:
                if st.session_state.get("dataset_name") != uploaded_file.name:
                    try:
                        df_up = load_csv(uploaded_file)
                        if df_up.shape[1] > 1:
                            success = run_full_pipeline(df_up, uploaded_file.name)
                            if success:
                                st.success(f"Loaded {uploaded_file.name} successfully!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load uploaded file: {str(e)}")


# -----------------------------------------------------------------------------
# TOP KPI STRIP
# -----------------------------------------------------------------------------
if clean_df is not None and "Sales" in clean_df.columns:
    s_series = clean_df["Sales"].dropna()
    total_sales = s_series.sum()
    avg_sales = s_series.mean()

    # Calculate recent 1M historical sales
    max_d = clean_df["Date"].max()
    m1_cutoff = max_d - pd.DateOffset(months=1)
    recent_1m_sales = clean_df[clean_df["Date"] >= m1_cutoff]["Sales"].sum()
    if recent_1m_sales == 0 and len(clean_df) > 0:
        recent_1m_sales = clean_df.tail(max(1, len(clean_df) // 10))["Sales"].sum()

    # Forecast values
    f_summary = st.session_state.get("forecast_summary", {})
    total_forecast = f_summary.get("total_forecast", 0.0)
    trend_str = f_summary.get("trend", "Stable")
    trend_color = "subtext-green" if trend_str == "Increasing" else ("subtext-red" if trend_str == "Decreasing" else "subtext-blue")

    # Model metrics on test split
    test_metrics = {"MAE": 0.0, "R2": 0.0, "WAPE_%": 0.0, "WAPE_Accuracy_%": 0.0}
    train_results = st.session_state.get("training_results")
    if train_results:
        test_comp = build_comparison_table(train_results, split="test")
        if not test_comp.empty:
            sel = test_comp[test_comp["Model"] == st.session_state["selected_model"]]
            if not sel.empty:
                test_metrics["MAE"] = float(sel.iloc[0]["MAE"])
                test_metrics["R2"] = float(sel.iloc[0]["R² Score"])
                test_metrics["WAPE_%"] = float(sel.iloc[0]["WAPE (%)"])
                test_metrics["WAPE_Accuracy_%"] = float(sel.iloc[0]["WAPE-based Forecast Accuracy (%)"])

    def fmt_curr(val):
        if val >= 1e9:
            return f"{currency}{val / 1e9:.2f}B"
        elif val >= 1e6:
            return f"{currency}{val / 1e6:.2f}M"
        elif val >= 1e3:
            return f"{currency}{val / 1e3:.1f}K"
        return f"{currency}{val:,.2f}"

    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">TOTAL SALES</div>
            <div class="kpi-value">{fmt_curr(total_sales)}</div>
            <div class="kpi-subtext subtext-muted">{len(s_series):,} dataset observations</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">AVERAGE SALES</div>
            <div class="kpi-value">{fmt_curr(avg_sales)}</div>
            <div class="kpi-subtext subtext-muted">Mean observed target</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">RECENT 1M SALES</div>
            <div class="kpi-value">{fmt_curr(recent_1m_sales)}</div>
            <div class="kpi-subtext subtext-muted">Trailing 30-day realized volume</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">FORECAST SALES</div>
            <div class="kpi-value">{fmt_curr(total_forecast)}</div>
            <div class="kpi-subtext {trend_color}">Horizon: {st.session_state.get('forecast_period_choice', '3 Months')}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">WAPE-BASED FORECAST ACCURACY</div>
            <div class="kpi-value">{test_metrics['WAPE_Accuracy_%']:.1f}%</div>
            <div class="kpi-subtext subtext-green">Holdout Test (1 - WAPE)</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HORIZONTAL NAVIGATION TABS
# -----------------------------------------------------------------------------
(
    tab_overview,
    tab_quality,
    tab_eda,
    tab_forecast,
    tab_validation,
    tab_features,
    tab_data,
) = st.tabs([
    "1. OVERVIEW",
    "2. DATA QUALITY",
    "3. SALES ANALYSIS & OUTLIERS",
    "4. FORECASTING",
    "5. MODEL VALIDATION",
    "6. FEATURE INSIGHTS",
    "7. HISTORICAL DATA",
])


# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab_overview:
    if clean_df is not None and "Sales" in clean_df.columns:
        # Full historical time series aggregate (dataset scope)
        full_ts_agg = prepare_time_series_eda(
            clean_df, date_col="Date", sales_col="Sales", frequency=st.session_state.get("frequency", "Weekly")
        )
        forecast_df = st.session_state.get("forecast_df")

        # Range Control Bar for Historical Sales Graph
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([5, 4, 3])
        with ctrl_c1:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #8B949E; margin-bottom: 4px;'>HISTORICAL VIEW RANGE</div>", unsafe_allow_html=True)
            hist_range_sel = st.segmented_control(
                "Historical Range",
                ["1 MONTH", "3 MONTHS", "6 MONTHS", "OVERALL"],
                default="OVERALL",
                label_visibility="collapsed",
                key="seg_hist_range_overview",
            )
            if not hist_range_sel:
                hist_range_sel = "OVERALL"

        with ctrl_c2:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #8B949E; margin-bottom: 4px;'>CHART OVERLAYS</div>", unsafe_allow_html=True)
            ov_cols = st.columns(2)
            with ov_cols[0]:
                show_rolling = st.checkbox("Rolling Average", value=True)
            with ov_cols[1]:
                show_trend = st.checkbox("Trend Line", value=False)

        with ctrl_c3:
            st.markdown(
                f"""
                <div style='text-align: right; padding-top: 18px;'>
                    <span class="badge badge-blue">MODEL: {st.session_state['selected_model']}</span>
                    <span class="badge badge-gray">{st.session_state.get('forecast_period_choice', '3 Months').upper()}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Filter historical data for visualization based on user selection
        filtered_ts_agg = filter_historical_by_range(full_ts_agg, date_col="Date", range_option=hist_range_sel)

        main_col, right_col = st.columns([7, 3])

        with main_col:
            fig_main = go.Figure()

            # 1. Historical Actual line
            if not filtered_ts_agg.empty:
                hist_dates = pd.to_datetime(filtered_ts_agg["Date"]).dt.strftime("%Y-%m-%d").tolist()
                fig_main.add_trace(go.Scatter(
                    x=hist_dates,
                    y=filtered_ts_agg["Total_Sales"],
                    mode="lines",
                    name="Actual Sales",
                    line=dict(color="#E2E8F0", width=1.8),
                ))

                # Rolling Average overlay
                if show_rolling and "Rolling_Mean_Fast" in filtered_ts_agg.columns:
                    fast_label = filtered_ts_agg["Rolling_Fast_Label"].iloc[0] if "Rolling_Fast_Label" in filtered_ts_agg.columns else "Rolling Mean"
                    fig_main.add_trace(go.Scatter(
                        x=hist_dates,
                        y=filtered_ts_agg["Rolling_Mean_Fast"],
                        mode="lines",
                        name=f"Rolling Mean ({fast_label})",
                        line=dict(color="#38BDF8", width=1.6, dash="dot"),
                    ))

                # Trend overlay
                if show_trend and "Trend_Line" in filtered_ts_agg.columns:
                    fig_main.add_trace(go.Scatter(
                        x=hist_dates,
                        y=filtered_ts_agg["Trend_Line"],
                        mode="lines",
                        name="Linear Trend",
                        line=dict(color="#94A3B8", width=1.3, dash="dash"),
                    ))

            # 2. Seamless Forecast Line & Boundary Connection
            if forecast_df is not None and not forecast_df.empty and not full_ts_agg.empty:
                last_hist_date_str = str(full_ts_agg["Date"].iloc[-1])[:10]
                last_hist_sales = float(full_ts_agg["Total_Sales"].iloc[-1])

                f_date_strings = [str(d)[:10] for d in forecast_df["Date"]]
                comb_dates = [last_hist_date_str] + f_date_strings
                comb_sales = [last_hist_sales] + list(forecast_df["Forecasted_Sales"])

                fig_main.add_trace(go.Scatter(
                    x=comb_dates,
                    y=comb_sales,
                    mode="lines+markers",
                    name=f"Forecast Sales ({st.session_state['selected_model']})",
                    line=dict(color=COLOR_PRIMARY, width=3.0),
                    marker=dict(size=5, color=COLOR_PRIMARY),
                ))

                # Subtle blue shaded forecast region background
                fig_main.add_vrect(
                    x0=last_hist_date_str,
                    x1=f_date_strings[-1],
                    fillcolor="rgba(59, 130, 246, 0.08)",
                    layer="below",
                    line_width=0,
                )

                # Vertical Forecast Start marker line & annotation
                fig_main.add_shape(
                    type="line",
                    x0=last_hist_date_str,
                    x1=last_hist_date_str,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color=COLOR_WARNING, width=2.0, dash="dash"),
                )
                fig_main.add_annotation(
                    x=last_hist_date_str,
                    y=1.02,
                    yref="paper",
                    text=f"Forecast Start ({last_hist_date_str})",
                    showarrow=False,
                    font=dict(color=COLOR_WARNING, size=11, family="Inter"),
                    xanchor="right",
                    yanchor="bottom",
                )

            apply_dark_theme(fig_main, title=f"Historical Actual Sales vs. Forecast ({hist_range_sel} View)", height=450)
            st.plotly_chart(fig_main, use_container_width=True)

        with right_col:
            # Forecast Summary Panel - Beginner & Executive Friendly
            f_sum = st.session_state.get("forecast_summary", {})
            last_actual_s = float(full_ts_agg["Total_Sales"].iloc[-1]) if not full_ts_agg.empty else 0.0
            last_actual_d = full_ts_agg["Date"].iloc[-1] if not full_ts_agg.empty else "N/A"
            first_pred_s = float(forecast_df["Forecasted_Sales"].iloc[0]) if (forecast_df is not None and not forecast_df.empty) else 0.0
            first_pred_d = forecast_df["Date"].iloc[0] if (forecast_df is not None and not forecast_df.empty) else "N/A"

            total_fc = float(f_sum.get("total_forecast", 0.0))
            avg_fc = float(f_sum.get("avg_forecast", 0.0))
            peak_val = float(f_sum.get("peak_val", 0.0))
            peak_date = f_sum.get("peak_period", "N/A")
            low_val = float(f_sum.get("low_val", 0.0))
            low_date = f_sum.get("low_period", "N/A")
            change_pct = float(f_sum.get("change_vs_recent_pct", 0.0))

            # Step continuation from actual data into prediction
            step_diff = first_pred_s - last_actual_s
            step_pct = (step_diff / max(last_actual_s, 1.0)) * 100.0
            step_sign = "+" if step_pct >= 0 else ""

            # Frequency & Horizon labels
            freq_name = st.session_state.get("frequency", "Weekly").lower()
            horizon_txt = st.session_state.get("forecast_period_choice", "3 Months")
            num_steps = len(forecast_df) if forecast_df is not None else 0

            # Dynamic plain-English trend badge & narrative
            if change_pct >= 2.0:
                trend_title = f"📈 EXPECTED GROWTH (+{change_pct:.1f}%)"
                trend_badge = "badge badge-green"
                trend_expl = f"Sales are projected to be <b>{abs(change_pct):.1f}% higher</b> compared to the previous {num_steps} {freq_name} periods."
                takeaway_msg = f"The <b>{st.session_state['selected_model']}</b> model predicts an <b>upward demand trend (+{change_pct:.1f}%)</b> across the next {horizon_txt}. Highest projected demand occurs on <b>{fmt_date_readable(peak_date)}</b> ({fmt_currency(peak_val, currency)})."
            elif change_pct <= -2.0:
                trend_title = f"📉 EXPECTED SLOWDOWN ({change_pct:.1f}%)"
                trend_badge = "badge badge-red"
                trend_expl = f"Sales are projected to be <b>{abs(change_pct):.1f}% lower</b> compared to the previous {num_steps} {freq_name} periods."
                takeaway_msg = f"The <b>{st.session_state['selected_model']}</b> model projects a <b>softening in demand ({change_pct:.1f}%)</b> over the next {horizon_txt}. Lowest projected volume is on <b>{fmt_date_readable(low_date)}</b> ({fmt_currency(low_val, currency)})."
            else:
                trend_title = f"➡️ STEADY DEMAND ({change_pct:+.1f}%)"
                trend_badge = "badge badge-blue"
                trend_expl = f"Sales are projected to remain steady, matching recent historical volume levels."
                takeaway_msg = f"The <b>{st.session_state['selected_model']}</b> model projects <b>stable ongoing demand</b> across the next {horizon_txt}, averaging approx <b>{fmt_currency(avg_fc, currency)}</b> per {freq_name} period."

            summary_html = textwrap.dedent(f"""
            <div class="analytics-panel">
                <div class="panel-title">
                    <span>FORECAST SUMMARY</span>
                    <span class="badge badge-blue">{st.session_state['selected_model']}</span>
                </div>

                <div class="summary-section">
                    <div class="summary-section-label">💰 Expected Future Sales ({horizon_txt})</div>
                    <div style="display: flex; align-items: baseline; gap: 8px; margin-top: 2px;">
                        <span style="font-size: 1.45rem; font-weight: 700; color: #FFFFFF;">{fmt_currency(total_fc, currency)}</span>
                        <span style="font-size: 0.78rem; color: #8B949E;">({currency}{total_fc:,.0f} exact)</span>
                    </div>
                    <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 3px;">
                        Projected total across the next <b>{num_steps} {freq_name} periods</b>
                    </div>
                </div>

                <div class="summary-section">
                    <div class="summary-section-label">📊 Demand Direction vs. Recent Sales</div>
                    <div style="margin: 5px 0;">
                        <span class="{trend_badge}" style="font-size: 0.76rem; padding: 3px 8px;">{trend_title}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #C9D1D9; line-height: 1.35;">
                        {trend_expl}
                    </div>
                </div>

                <div class="summary-dual-grid">
                    <div class="summary-mini-card">
                        <div class="summary-section-label" style="color: #22C55E;">🌟 Highest Peak</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #22C55E; margin-top: 2px;">{fmt_currency(peak_val, currency)}</div>
                        <div style="font-size: 0.76rem; color: #E2E8F0; font-weight: 600;">{fmt_date_readable(peak_date)}</div>
                        <div style="font-size: 0.70rem; color: #8B949E; margin-top: 2px;">Expected sales peak</div>
                    </div>
                    <div class="summary-mini-card">
                        <div class="summary-section-label" style="color: #EF4444;">📉 Lowest Dip</div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #EF4444; margin-top: 2px;">{fmt_currency(low_val, currency)}</div>
                        <div style="font-size: 0.76rem; color: #E2E8F0; font-weight: 600;">{fmt_date_readable(low_date)}</div>
                        <div style="font-size: 0.70rem; color: #8B949E; margin-top: 2px;">Expected sales valley</div>
                    </div>
                </div>

                <div class="summary-section">
                    <div class="summary-section-label">🎯 Seamless Starting Point</div>
                    <div style="font-size: 0.82rem; color: #E2E8F0; display: flex; justify-content: space-between; align-items: center; margin-top: 2px;">
                        <span>Last Actual: <b>{fmt_currency(last_actual_s, currency)}</b></span>
                        <span style="color: #38BDF8;">➔ First Forecast: <b>{fmt_currency(first_pred_s, currency)}</b></span>
                    </div>
                    <div style="font-size: 0.72rem; color: #8B949E; margin-top: 3px;">
                        Smooth continuation ({step_sign}{step_pct:.1f}% first step) from actual data into prediction
                    </div>
                </div>

                <div class="summary-takeaway">
                    <div style="font-weight: 700; color: #38BDF8; margin-bottom: 2px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">
                        💡 Plain English Takeaway
                    </div>
                    {takeaway_msg}
                </div>
            </div>
            """).strip()

            st.html(summary_html)

        # 1 / 3 / 6 MONTH & OVERALL HISTORICAL SALES SUMMARY TABLE
        st.markdown("#### 📅 Historical Sales Summary (1M / 3M / 6M / Overall)")
        period_summaries = compute_period_sales_summaries(full_ts_agg, sales_col="Total_Sales", date_col="Date")
        if not period_summaries.empty:
            summary_cols = st.columns(4)
            for idx, row in period_summaries.iterrows():
                with summary_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="analytics-panel" style="padding: 12px 16px;">
                            <div class="kpi-label">{row['Period']}</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{fmt_currency(row['Total Sales'], currency)}</div>
                            <div style="font-size: 0.82rem; color: #94A3B8;">Average: {fmt_currency(row['Average Sales'], currency)}</div>
                            <div style="font-size: 0.82rem; color: #94A3B8;">Median: {fmt_currency(row['Median Sales'], currency)}</div>
                            <div class="kpi-subtext subtext-muted">{row['Observations']} periods ({fmt_date_readable(row['Start Date'])} → {fmt_date_readable(row['End Date'])})</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Bottom Overview Panels
        bot_c1, bot_c2, bot_c3, bot_c4 = st.columns(4)
        if not full_ts_agg.empty:
            peak_hist_idx = full_ts_agg["Total_Sales"].idxmax()
            low_hist_idx = full_ts_agg["Total_Sales"].idxmin()

            with bot_c1:
                st.markdown(
                    f"""
                    <div class="analytics-panel" style="padding: 12px 16px;">
                        <div class="kpi-label">HISTORICAL PEAK RECORD</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #22C55E;">{fmt_currency(full_ts_agg['Total_Sales'].max(), currency)}</div>
                        <div class="kpi-subtext subtext-muted">{fmt_date_readable(full_ts_agg['Date'].iloc[peak_hist_idx])} (All-time high)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with bot_c2:
                st.markdown(
                    f"""
                    <div class="analytics-panel" style="padding: 12px 16px;">
                        <div class="kpi-label">HISTORICAL LOWEST RECORD</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #EF4444;">{fmt_currency(full_ts_agg['Total_Sales'].min(), currency)}</div>
                        <div class="kpi-subtext subtext-muted">{fmt_date_readable(full_ts_agg['Date'].iloc[low_hist_idx])} (All-time low)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with bot_c3:
                out_cnt = st.session_state.get("outlier_analysis", {}).get("outlier_count", 0)
                out_pct = st.session_state.get("outlier_analysis", {}).get("outlier_pct", 0.0)
                st.markdown(
                    f"""
                    <div class="analytics-panel" style="padding: 12px 16px;">
                        <div class="kpi-label">DATASET OUTLIERS / SPIKES</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #F59E0B;">{out_cnt:,} records ({out_pct:.1f}%)</div>
                        <div class="kpi-subtext subtext-muted">Preserved for model training</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with bot_c4:
                st.markdown(
                    f"""
                    <div class="analytics-panel" style="padding: 12px 16px;">
                        <div class="kpi-label">MODEL VALIDATION SCORE</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #38BDF8;">R² = {test_metrics['R2']:.4f}</div>
                        <div class="kpi-subtext subtext-green">Accuracy: {test_metrics['WAPE_Accuracy_%']:.1f}% (Holdout)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =============================================================================
# TAB 2: DATA QUALITY & CLEANING
# =============================================================================
with tab_quality:
    clean_stats = st.session_state.get("cleaning_stats", {})
    schema_meta = st.session_state.get("schema_meta", {})

    # Top Data Quality KPI Cards
    dq1, dq2, dq3, dq4, dq5, dq6 = st.columns(6)
    with dq1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">TOTAL ROWS</div><div class="kpi-value">{clean_stats.get("final_rows", len(raw_df)):,}</div><div class="kpi-subtext subtext-muted">Initial: {clean_stats.get("initial_rows", len(raw_df)):,}</div></div>',
            unsafe_allow_html=True,
        )
    with dq2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">TOTAL COLUMNS</div><div class="kpi-value">{raw_df.shape[1]}</div><div class="kpi-subtext subtext-muted">Detected schema</div></div>',
            unsafe_allow_html=True,
        )
    with dq3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">MISSING CELLS</div><div class="kpi-value">{clean_stats.get("final_missing", 0):,}</div><div class="kpi-subtext subtext-muted">Imputed via forward/median</div></div>',
            unsafe_allow_html=True,
        )
    with dq4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">DUPLICATES REMOVED</div><div class="kpi-value">{clean_stats.get("duplicates_removed", 0):,}</div><div class="kpi-subtext subtext-muted">Exact duplicates</div></div>',
            unsafe_allow_html=True,
        )
    with dq5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">DETECTED FREQUENCY</div><div class="kpi-value" style="font-size: 1.25rem;">{st.session_state.get("frequency", "Weekly").upper()}</div><div class="kpi-subtext subtext-muted">Cadence</div></div>',
            unsafe_allow_html=True,
        )
    with dq6:
        out_cnt = st.session_state.get("outlier_analysis", {}).get("outlier_count", 0)
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">POTENTIAL OUTLIERS</div><div class="kpi-value">{out_cnt:,}</div><div class="kpi-subtext subtext-muted">Retained for ML training</div></div>',
            unsafe_allow_html=True,
        )

    # Column Classification Panel
    st.markdown("#### 📋 Data Columns Classification Panel")
    if schema_meta and "audit_df" in schema_meta:
        st.dataframe(schema_meta["audit_df"], use_container_width=True, hide_index=True)

    # Manual Column Overrides
    with st.expander("⚙️ Manual Role Override & Re-run Pipeline"):
        st.caption("Override automatically detected column mappings if your schema uses custom nomenclature.")
        c_cols = list(raw_df.columns)
        ov_col1, ov_col2, ov_col3, ov_col4, ov_col5 = st.columns(5)

        roles = st.session_state.get("detected_roles", {})
        with ov_col1:
            date_idx = c_cols.index(roles["date_col"]) if roles.get("date_col") in c_cols else 0
            new_date = st.selectbox("Date Column", c_cols, index=date_idx)
        with ov_col2:
            sales_idx = c_cols.index(roles["sales_col"]) if roles.get("sales_col") in c_cols else 0
            new_sales = st.selectbox("Sales Target Column", c_cols, index=sales_idx)
        with ov_col3:
            store_opts = ["None"] + c_cols
            store_idx = store_opts.index(roles["store_col"]) if roles.get("store_col") in store_opts else 0
            new_store = st.selectbox("Store / Entity Column", store_opts, index=store_idx)
        with ov_col4:
            prod_opts = ["None"] + c_cols
            prod_idx = prod_opts.index(roles["product_col"]) if roles.get("product_col") in prod_opts else 0
            new_prod = st.selectbox("Product / SKU Column", prod_opts, index=prod_idx)
        with ov_col5:
            ord_opts = ["None"] + c_cols
            ord_idx = ord_opts.index(roles["order_col"]) if roles.get("order_col") in ord_opts else 0
            new_order = st.selectbox("Order / Volume Column", ord_opts, index=ord_idx)

        if st.button("Apply Column Overrides & Re-run Pipeline"):
            overrides = {
                "date_col": new_date,
                "sales_col": new_sales,
                "store_col": None if new_store == "None" else new_store,
                "product_col": None if new_prod == "None" else new_prod,
                "order_col": None if new_order == "None" else new_order,
            }
            run_full_pipeline(raw_df, st.session_state["dataset_name"], overrides=overrides)
            st.rerun()

    # Before vs After Cleaning Summary
    st.markdown("#### 🔄 Before vs. After Cleaning Audit")
    if clean_stats and "before_after_df" in clean_stats:
        st.dataframe(clean_stats["before_after_df"], use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3: SALES ANALYSIS & OUTLIERS
# =============================================================================
with tab_eda:
    if clean_df is not None and "Sales" in clean_df.columns:
        st.markdown("### 🚨 Dedicated Outlier Analysis & Diagnostics")
        st.caption(
            "Retail sales spikes frequently represent genuine seasonal surges, holiday promotions, or bulk demand. "
            "Our system keeps these real observations in the model while rigorously identifying and highlighting them."
        )

        out_res = st.session_state.get("outlier_analysis", {})

        # Outlier KPI Strip
        ok1, ok2, ok3, ok4, ok5, ok6 = st.columns(6)
        with ok1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Q1 (25TH PERCENTILE)</div><div class="kpi-value">{currency}{out_res.get("q1", 0):,.2f}</div><div class="kpi-subtext subtext-muted">Lower Quartile</div></div>',
                unsafe_allow_html=True,
            )
        with ok2:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">MEDIAN (50TH PERCENTILE)</div><div class="kpi-value">{currency}{out_res.get("median", 0):,.2f}</div><div class="kpi-subtext subtext-muted">Central Value</div></div>',
                unsafe_allow_html=True,
            )
        with ok3:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Q3 (75TH PERCENTILE)</div><div class="kpi-value">{currency}{out_res.get("q3", 0):,.2f}</div><div class="kpi-subtext subtext-muted">Upper Quartile</div></div>',
                unsafe_allow_html=True,
            )
        with ok4:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">IQR (SPREAD)</div><div class="kpi-value">{currency}{out_res.get("iqr", 0):,.2f}</div><div class="kpi-subtext subtext-muted">Q3 - Q1</div></div>',
                unsafe_allow_html=True,
            )
        with ok5:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">UPPER BOUND (Q3 + 1.5 IQR)</div><div class="kpi-value">{currency}{out_res.get("upper_bound", 0):,.2f}</div><div class="kpi-subtext subtext-muted">Outlier Threshold</div></div>',
                unsafe_allow_html=True,
            )
        with ok6:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">OUTLIER COUNT & SHARE</div><div class="kpi-value">{out_res.get("outlier_count", 0):,}</div><div class="kpi-subtext subtext-muted">{out_res.get("outlier_pct", 0.0):.2f}% of total records</div></div>',
                unsafe_allow_html=True,
            )

        # 1. Sales-over-time chart with outliers clearly highlighted!
        st.markdown("#### 📈 Sales Over Time with Highlighted Outliers")
        ts_agg = prepare_time_series_eda(clean_df, date_col="Date", sales_col="Sales", frequency=st.session_state.get("frequency", "Weekly"))
        fig_out_ts = go.Figure()

        if not ts_agg.empty:
            ts_dates = pd.to_datetime(ts_agg["Date"]).dt.strftime("%Y-%m-%d").tolist()
            # Normal line
            fig_out_ts.add_trace(go.Scatter(
                x=ts_dates,
                y=ts_agg["Total_Sales"],
                mode="lines",
                name="Sales Time Series",
                line=dict(color="#94A3B8", width=1.8),
            ))

            # Threshold line for aggregated time series
            ts_s = ts_agg["Total_Sales"]
            ts_q3 = float(ts_s.quantile(0.75))
            ts_iqr = ts_q3 - float(ts_s.quantile(0.25))
            ts_upper = ts_q3 + 1.5 * ts_iqr

            # Outlier points
            out_mask = ts_agg["Total_Sales"] > ts_upper
            out_dates = [ts_dates[i] for i, m in enumerate(out_mask) if m]
            out_vals = ts_agg.loc[out_mask, "Total_Sales"].tolist()

            if out_dates:
                fig_out_ts.add_trace(go.Scatter(
                    x=out_dates,
                    y=out_vals,
                    mode="markers",
                    name="Identified Outlier Spikes",
                    marker=dict(color=COLOR_NEGATIVE, size=8, symbol="circle", line=dict(color="#FFFFFF", width=1)),
                ))

            fig_out_ts.add_hline(
                y=ts_upper,
                line_dash="dash",
                line_color=COLOR_WARNING,
                annotation_text=f"Normal Range Upper Bound ({currency}{ts_upper:,.0f})",
                annotation_position="top left",
            )

        apply_dark_theme(fig_out_ts, title="Sales-Over-Time Trajectory (Normal Sales Line vs. Highlighted Outliers)", height=380)
        st.plotly_chart(fig_out_ts, use_container_width=True)

        # 2. Box Plot & Distribution Histogram
        box_c1, box_c2 = st.columns(2)
        with box_c1:
            fig_dist = px.histogram(
                clean_df,
                x="Sales",
                nbins=45,
                title="Sales Distribution Histogram & IQR Bounds",
                color_discrete_sequence=[COLOR_PRIMARY],
            )
            if out_res.get("upper_bound", 0) > 0:
                fig_dist.add_vline(
                    x=out_res["upper_bound"],
                    line_dash="dash",
                    line_color=COLOR_NEGATIVE,
                    annotation_text=f"Upper Bound ({currency}{out_res['upper_bound']:,.0f})",
                    annotation_position="top right",
                )
            apply_dark_theme(fig_dist, height=340)
            st.plotly_chart(fig_dist, use_container_width=True)

        with box_c2:
            fig_box = px.box(
                clean_df,
                y="Sales",
                title="Sales Target Box Plot (Dispersion & Extreme Values)",
                color_discrete_sequence=["#38BDF8"],
            )
            apply_dark_theme(fig_box, height=340)
            st.plotly_chart(fig_box, use_container_width=True)

        # 3. Top 10 Largest Outliers Table
        st.markdown("#### 📌 Top 10 Largest Outliers (Ranked by Deviation from Normal Range)")
        if out_res.get("has_outliers") and not out_res["top_outliers_df"].empty:
            st.dataframe(out_res["top_outliers_df"], use_container_width=True, hide_index=True)
        else:
            st.info("No extreme statistical outliers detected in the sales series.")

        st.markdown("---")

        # 4. Seasonality Analytics
        st.markdown("#### 📅 Calendar Seasonality Analysis")
        seas_data = prepare_seasonality_analysis(
            clean_df, date_col="Date", sales_col="Sales", frequency=st.session_state.get("frequency", "Daily")
        )

        seas_c1, seas_c2 = st.columns(2)
        with seas_c1:
            if "month" in seas_data and not seas_data["month"].empty:
                fig_m = px.bar(
                    seas_data["month"],
                    x="Month_Name",
                    y="Mean_Sales",
                    title="Average Sales Volume by Month",
                    color="Mean_Sales",
                    color_continuous_scale="Blues",
                )
                apply_dark_theme(fig_m, height=340, show_legend=False)
                st.plotly_chart(fig_m, use_container_width=True)
            elif "quarter" in seas_data:
                fig_q = px.bar(
                    seas_data["quarter"],
                    x="Quarter",
                    y="Mean_Sales",
                    title="Average Sales Volume by Quarter",
                    color="Mean_Sales",
                    color_continuous_scale="Blues",
                )
                apply_dark_theme(fig_q, height=340, show_legend=False)
                st.plotly_chart(fig_q, use_container_width=True)

        with seas_c2:
            if "dow" in seas_data and not seas_data["dow"].empty:
                fig_dow = px.bar(
                    seas_data["dow"],
                    x="Day_of_Week",
                    y="Mean_Sales",
                    title="Average Sales by Day of Week",
                    color_discrete_sequence=[COLOR_PRIMARY],
                )
                apply_dark_theme(fig_dow, height=340)
                st.plotly_chart(fig_dow, use_container_width=True)
            elif "quarter" in seas_data:
                fig_q = px.pie(
                    seas_data["quarter"],
                    names="Quarter",
                    values="Total_Sales",
                    title="Quarterly Sales Share",
                    color_discrete_sequence=["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"],
                )
                apply_dark_theme(fig_q, height=340)
                st.plotly_chart(fig_q, use_container_width=True)


# =============================================================================
# TAB 4: FORECASTING
# =============================================================================
with tab_forecast:
    if clean_df is not None and "Sales" in clean_df.columns:
        st.markdown("### 🔮 MULTI-STEP SALES FORECASTING")

        pipeline = st.session_state.get("pipeline")
        train_results = st.session_state.get("training_results")
        scope_ts_df = st.session_state.get("scope_ts_df")

        if pipeline and train_results and scope_ts_df is not None:
            # Controls Bar strictly per Section 31
            fc_ctrl1, fc_ctrl2, fc_ctrl3, fc_ctrl4 = st.columns([3, 4, 3, 2])

            # 1. Model Selector: ONLY Genuine ML Models (No baselines!)
            valid_ml_models = []
            if XGBOOST_AVAILABLE and "XGBoost" in train_results["models"]:
                valid_ml_models.append("XGBoost")
            if "Random Forest" in train_results["models"]:
                valid_ml_models.append("Random Forest")
            if not valid_ml_models:
                valid_ml_models = ["XGBoost"]

            with fc_ctrl1:
                cur_sel = st.session_state.get("selected_model", valid_ml_models[0])
                m_idx = valid_ml_models.index(cur_sel) if cur_sel in valid_ml_models else 0
                chosen_model_name = st.selectbox("MODEL", valid_ml_models, index=m_idx)

            # 2. Forecast Period: [ 1 MONTH ] [ 3 MONTHS ] [ 6 MONTHS ]
            with fc_ctrl2:
                cur_period = st.session_state.get("forecast_period_choice", "3 Months")
                chosen_period = st.segmented_control(
                    "FORECAST PERIOD",
                    ["1 MONTH", "3 MONTHS", "6 MONTHS"],
                    default=cur_period.upper(),
                )
                if not chosen_period:
                    chosen_period = "3 MONTHS"

            # 3. Scope: Entire Uploaded Dataset (Read-only)
            with fc_ctrl3:
                st.text_input("FORECAST SCOPE", value="Entire Uploaded Dataset", disabled=True)

            with fc_ctrl4:
                st.write("")
                st.write("")
                if st.button("Generate Forecast", use_container_width=True):
                    st.session_state["selected_model"] = chosen_model_name
                    st.session_state["forecast_period_choice"] = chosen_period

                    # Convert selected period to frequency-aware horizon
                    freq = st.session_state.get("frequency", "Weekly")
                    horizon = get_horizon_for_period(chosen_period, freq)
                    st.session_state["forecast_horizon"] = horizon

                    chosen_m = train_results["models"][chosen_model_name]
                    new_forecast = generate_recursive_forecast(
                        historical_df=scope_ts_df,
                        pipeline=pipeline,
                        model=chosen_m,
                        horizon=horizon,
                        frequency=freq,
                    )
                    st.session_state["forecast_df"] = new_forecast
                    st.session_state["forecast_summary"] = compute_forecast_summary(new_forecast, scope_ts_df)
                    st.rerun()

            # Display Forecast Metric Cards per Section 31
            forecast_df = st.session_state.get("forecast_df")
            f_sum = st.session_state.get("forecast_summary", {})

            full_ts_agg = prepare_time_series_eda(
                clean_df, date_col="Date", sales_col="Sales", frequency=st.session_state.get("frequency", "Weekly")
            )
            last_actual_s = float(full_ts_agg["Total_Sales"].iloc[-1]) if not full_ts_agg.empty else 0.0

            # Test Accuracy
            test_metrics = {"WAPE_Accuracy_%": 0.0}
            if train_results:
                test_comp = build_comparison_table(train_results, split="test")
                if not test_comp.empty:
                    sel = test_comp[test_comp["Model"] == st.session_state["selected_model"]]
                    if not sel.empty:
                        test_metrics["WAPE_Accuracy_%"] = float(sel.iloc[0]["WAPE-based Forecast Accuracy (%)"])

            f_kpi1, f_kpi2, f_kpi3, f_kpi4 = st.columns(4)
            with f_kpi1:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">LAST ACTUAL SALES</div><div class="kpi-value">{fmt_currency(last_actual_s, currency)}</div><div class="kpi-subtext subtext-muted">{fmt_date_readable(full_ts_agg["Date"].iloc[-1])}</div></div>',
                    unsafe_allow_html=True,
                )
            with f_kpi2:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">FORECAST TOTAL</div><div class="kpi-value">{fmt_currency(f_sum.get("total_forecast", 0), currency)}</div><div class="kpi-subtext subtext-muted">{len(forecast_df) if forecast_df is not None else 0} future periods</div></div>',
                    unsafe_allow_html=True,
                )
            with f_kpi3:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">EXPECTED AVERAGE</div><div class="kpi-value">{fmt_currency(f_sum.get("avg_forecast", 0), currency)}</div><div class="kpi-subtext subtext-muted">Mean forecast value</div></div>',
                    unsafe_allow_html=True,
                )
            with f_kpi4:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">MODEL ACCURACY (WAPE)</div><div class="kpi-value">{test_metrics["WAPE_Accuracy_%"]:.1f}%</div><div class="kpi-subtext subtext-green">Holdout Test Set</div></div>',
                    unsafe_allow_html=True,
                )

            # Historical View Range Controls for Forecast Tab Chart
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            f_rng_c1, f_rng_c2 = st.columns([5, 7])
            with f_rng_c1:
                hist_range_fc = st.segmented_control(
                    "Historical View Range (Chart Zoom)",
                    ["1 MONTH", "3 MONTHS", "6 MONTHS", "OVERALL"],
                    default="OVERALL",
                    key="seg_hist_range_fc",
                )
                if not hist_range_fc:
                    hist_range_fc = "OVERALL"

            filtered_fc_ts = filter_historical_by_range(full_ts_agg, date_col="Date", range_option=hist_range_fc)

            # Large Dedicated Forecast Chart
            fig_fc = go.Figure()

            # Historical Trace
            if not filtered_fc_ts.empty:
                fc_hist_dates = pd.to_datetime(filtered_fc_ts["Date"]).dt.strftime("%Y-%m-%d").tolist()
                fig_fc.add_trace(go.Scatter(
                    x=fc_hist_dates,
                    y=filtered_fc_ts["Total_Sales"],
                    mode="lines",
                    name="Actual Sales",
                    line=dict(color="#E2E8F0", width=2.0),
                ))

            # Seamless Forecast Connection
            if forecast_df is not None and not forecast_df.empty and not full_ts_agg.empty:
                last_d_str = str(full_ts_agg["Date"].iloc[-1])[:10]
                last_s = float(full_ts_agg["Total_Sales"].iloc[-1])

                f_date_strings = [str(d)[:10] for d in forecast_df["Date"]]
                comb_dates = [last_d_str] + f_date_strings
                comb_sales = [last_s] + list(forecast_df["Forecasted_Sales"])

                fig_fc.add_trace(go.Scatter(
                    x=comb_dates,
                    y=comb_sales,
                    mode="lines+markers",
                    name=f"Forecast Sales ({st.session_state['selected_model']})",
                    line=dict(color=COLOR_PRIMARY, width=3.2),
                    marker=dict(size=6, color=COLOR_PRIMARY),
                ))

                # Shaded forecast area
                fig_fc.add_vrect(
                    x0=last_d_str,
                    x1=f_date_strings[-1],
                    fillcolor="rgba(59, 130, 246, 0.08)",
                    layer="below",
                    line_width=0,
                )

                # Vertical Forecast Start Marker
                fig_fc.add_shape(
                    type="line",
                    x0=last_d_str,
                    x1=last_d_str,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color=COLOR_WARNING, width=2.0, dash="dash"),
                )
                fig_fc.add_annotation(
                    x=last_d_str,
                    y=1.02,
                    yref="paper",
                    text=f"FORECAST START: {last_d_str}",
                    showarrow=False,
                    font=dict(color=COLOR_WARNING, size=11, family="Inter"),
                    xanchor="right",
                    yanchor="bottom",
                )

            apply_dark_theme(fig_fc, title="Multi-Step Recursive Sales Forecast Curve", height=460)
            st.plotly_chart(fig_fc, use_container_width=True)

            # FORECAST SUMMARY & TABLE
            st.markdown("#### 📊 FORECAST SUMMARY")
            fs1, fs2, fs3, fs4 = st.columns(4)
            with fs1:
                st.markdown(
                    f'<div class="analytics-panel"><div class="kpi-label">PEAK FORECAST</div><div style="font-size: 1.25rem; font-weight: 700; color: #22C55E;">{fmt_currency(f_sum.get("peak_val", 0), currency)}</div><div class="kpi-subtext subtext-muted">{fmt_date_readable(f_sum.get("peak_period", "N/A"))}</div></div>',
                    unsafe_allow_html=True,
                )
            with fs2:
                st.markdown(
                    f'<div class="analytics-panel"><div class="kpi-label">LOWEST FORECAST</div><div style="font-size: 1.25rem; font-weight: 700; color: #EF4444;">{fmt_currency(f_sum.get("low_val", 0), currency)}</div><div class="kpi-subtext subtext-muted">{fmt_date_readable(f_sum.get("low_period", "N/A"))}</div></div>',
                    unsafe_allow_html=True,
                )
            with fs3:
                st.markdown(
                    f'<div class="analytics-panel"><div class="kpi-label">AVERAGE FORECAST</div><div style="font-size: 1.25rem; font-weight: 700; color: #38BDF8;">{fmt_currency(f_sum.get("avg_forecast", 0), currency)}</div><div class="kpi-subtext subtext-muted">Per step expectation</div></div>',
                    unsafe_allow_html=True,
                )
            with fs4:
                st.markdown(
                    f'<div class="analytics-panel"><div class="kpi-label">TOTAL FORECAST</div><div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">{fmt_currency(f_sum.get("total_forecast", 0), currency)}</div><div class="kpi-subtext subtext-muted">Cumulative Demand</div></div>',
                    unsafe_allow_html=True,
                )

            # FORECAST TABLE
            st.markdown("#### 📄 FORECAST TABLE")
            fc_t1, fc_t2 = st.columns([7, 3])
            with fc_t1:
                if forecast_df is not None:
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            with fc_t2:
                if forecast_df is not None:
                    csv_fc = forecast_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv_fc,
                        file_name=f"sales_forecast_{st.session_state['selected_model']}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.markdown(
                        f"""
                        <div class="analytics-panel" style="margin-top: 10px; font-size: 0.82rem; color: #94A3B8;">
                            <strong>Multi-Step Engine Rules:</strong><br>
                            Each step predicts period <i>t</i>, writes the result to the lag buffer, recomputes rolling statistics, and iterates forward cleanly.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# =============================================================================
# TAB 5: MODEL VALIDATION
# =============================================================================
with tab_validation:
    train_results = st.session_state.get("training_results")
    split_info = st.session_state.get("split_info")

    if train_results and split_info:
        st.markdown("### 🏆 MODEL PERFORMANCE & VALIDATION")

        # Partitioning Cards (Section 22)
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">TRAIN PERIOD (70%)</div><div class="kpi-value">{split_info["train_rows"]:,}</div><div class="kpi-subtext subtext-muted">{split_info["train_start"]} → {split_info["train_end"]}</div></div>',
                unsafe_allow_html=True,
            )
        with p_c2:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">VAL PERIOD (15%)</div><div class="kpi-value">{split_info["val_rows"]:,}</div><div class="kpi-subtext subtext-muted">{split_info["val_start"]} → {split_info["val_end"]}</div></div>',
                unsafe_allow_html=True,
            )
        with p_c3:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">TEST PERIOD (15%)</div><div class="kpi-value">{split_info["test_rows"]:,}</div><div class="kpi-subtext subtext-muted">{split_info["test_start"]} → {split_info["test_end"]}</div></div>',
                unsafe_allow_html=True,
            )
        with p_c4:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">CHAMPION MODEL</div><div class="kpi-value" style="font-size: 1.25rem;">{st.session_state["selected_model"].upper()}</div><div class="kpi-subtext subtext-green">Evaluated on unseen test data</div></div>',
                unsafe_allow_html=True,
            )

        # Leaderboard Table
        st.markdown("#### 🥇 Model Comparison Leaderboard (Holdout Test Split)")
        test_comp = build_comparison_table(train_results, split="test")
        st.dataframe(
            test_comp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model Candidate"),
                "WAPE-based Forecast Accuracy (%)": st.column_config.NumberColumn("WAPE-based Forecast Accuracy", format="%.2f%%"),
                "MAE": st.column_config.NumberColumn(f"MAE ({currency})", format=f"{currency}%.2f"),
                "RMSE": st.column_config.NumberColumn(f"RMSE ({currency})", format=f"{currency}%.2f"),
                "WAPE (%)": st.column_config.NumberColumn("WAPE (%)", format="%.2f%%"),
                "R² Score": st.column_config.NumberColumn("R² Score", format="%.4f"),
                "Train Time (s)": st.column_config.NumberColumn("Train Time (s)", format="%.3f s"),
            },
        )

        # Collapsible Explanation Section (Section 21)
        with st.expander("ℹ️ How is Forecast Accuracy Calculated? (Click to Expand)", expanded=True):
            st.markdown(get_accuracy_calculation_explanation(), unsafe_allow_html=True)

        # Selected Model Performance Cards
        sel_row = test_comp[test_comp["Model"] == st.session_state["selected_model"]].iloc[0]
        sel_metrics = {
            "MAE": float(sel_row["MAE"]),
            "RMSE": float(sel_row["RMSE"]),
            "WAPE_%": float(sel_row["WAPE (%)"]),
            "WAPE_Accuracy_%": float(sel_row["WAPE-based Forecast Accuracy (%)"]),
            "R2": float(sel_row["R² Score"]),
        }
        human_summaries = get_human_metric_summary(sel_metrics, currency_symbol=currency)

        h_cols = st.columns(4)
        for idx, item in enumerate(human_summaries):
            with h_cols[idx]:
                st.markdown(
                    f"""
                    <div class="analytics-panel" style="padding: 12px 16px;">
                        <div class="kpi-label">{item['metric']}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{item['value']}</div>
                        <div class="kpi-subtext subtext-muted">{item['interpretation']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Actual vs Predicted Graph (Section 23)
        st.markdown(f"#### 📈 Actual Sales vs. Predicted Sales (Unseen Test Period: {split_info['test_start']} → {split_info['test_end']})")
        sel_model_name = st.session_state["selected_model"]
        y_test_act = train_results["actuals"]["test_sales"]
        y_test_pred = train_results["predictions"][sel_model_name]["test_sales"]

        feat_df = st.session_state.get("feature_df")
        diag_df = generate_diagnostics(feat_df.tail(len(y_test_act)), date_col="Date", y_true=y_test_act, y_pred=y_test_pred)

        val_chart1, val_chart2 = st.columns([7, 3])

        with val_chart1:
            fig_act_pred = go.Figure()
            fig_act_pred.add_trace(go.Scatter(
                x=diag_df["Date"],
                y=diag_df["Actual_Sales"],
                mode="lines",
                name="Actual Sales",
                line=dict(color="#CBD5E1", width=2.0),
            ))
            fig_act_pred.add_trace(go.Scatter(
                x=diag_df["Date"],
                y=diag_df["Predicted_Sales"],
                mode="lines+markers",
                name=f"Predicted ({sel_model_name})",
                line=dict(color=COLOR_PRIMARY, width=2.4),
                marker=dict(size=4),
            ))
            apply_dark_theme(fig_act_pred, title="Actual Sales vs. Predicted Sales (Holdout Test Split)", height=380)
            st.plotly_chart(fig_act_pred, use_container_width=True)

        with val_chart2:
            fig_45 = go.Figure()
            fig_45.add_trace(go.Scatter(
                x=diag_df["Actual_Sales"],
                y=diag_df["Predicted_Sales"],
                mode="markers",
                name="Test Observations",
                marker=dict(color=COLOR_PRIMARY, size=5, opacity=0.7),
            ))
            min_v = min(diag_df["Actual_Sales"].min(), diag_df["Predicted_Sales"].min())
            max_v = max(diag_df["Actual_Sales"].max(), diag_df["Predicted_Sales"].max())
            fig_45.add_trace(go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="1:1 Parity Line",
                line=dict(color=COLOR_POSITIVE, width=1.5, dash="dash"),
            ))
            apply_dark_theme(fig_45, title="Predicted vs. Actual Parity Plot", height=380)
            st.plotly_chart(fig_45, use_container_width=True)

        # Residuals Over Time & Error Distribution (Section 24)
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            fig_res_time = px.line(
                diag_df,
                x="Date",
                y="Residual",
                title="Prediction Error Over Time (Actual - Predicted)",
                color_discrete_sequence=["#F59E0B"],
            )
            fig_res_time.add_hline(y=0.0, line_dash="dash", line_color="#94A3B8")
            apply_dark_theme(fig_res_time, height=320)
            st.plotly_chart(fig_res_time, use_container_width=True)

        with res_col2:
            fig_res_dist = px.histogram(
                diag_df,
                x="Residual",
                nbins=25,
                title="Error Distribution Histogram (Zero-Centered Normality)",
                color_discrete_sequence=[COLOR_PRIMARY],
            )
            fig_res_dist.add_vline(x=0.0, line_dash="dash", line_color=COLOR_POSITIVE)
            apply_dark_theme(fig_res_dist, height=320)
            st.plotly_chart(fig_res_dist, use_container_width=True)


# =============================================================================
# TAB 6: FEATURE INSIGHTS
# =============================================================================
with tab_features:
    pipeline = st.session_state.get("pipeline")
    train_results = st.session_state.get("training_results")

    if pipeline and train_results:
        st.markdown("### 🧬 FEATURE IMPORTANCE & LEAKAGE AUDIT")

        fi_col1, fi_col2 = st.columns([5, 5])

        with fi_col1:
            st.markdown("#### 📊 TOP FEATURES INFLUENCING FORECAST")
            sel_model_obj = train_results["models"].get(st.session_state["selected_model"])
            fi_df = extract_feature_importance(sel_model_obj, pipeline.feature_columns, top_n=15)

            if not fi_df.empty:
                fig_fi = px.bar(
                    fi_df.sort_values(by="Importance_%", ascending=True),
                    x="Importance_%",
                    y="Feature",
                    orientation="h",
                    title=f"Top 15 Predictive Features ({st.session_state['selected_model']})",
                    color="Importance_%",
                    color_continuous_scale="Blues",
                )
                apply_dark_theme(fig_fi, height=420, show_legend=False)
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance is available for tree-based regressors (XGBoost & Random Forest).")

        with fi_col2:
            st.markdown("#### 🛡️ FEATURE LEAKAGE PREVENTION AUDIT")
            audit_report = pipeline.get_leakage_audit_report()
            st.dataframe(audit_report, use_container_width=True, hide_index=True)

        # Commercial Takeaways
        st.markdown("#### 💡 Commercial Business Observations")
        takeaways = generate_business_takeaways(
            clean_df,
            forecast_summary=st.session_state.get("forecast_summary"),
            sales_col="Sales",
            date_col="Date",
            discount_col=roles.get("discount_col"),
            holiday_col=roles.get("holiday_col"),
        )
        if takeaways:
            t_cols = st.columns(len(takeaways))
            for i, tk in enumerate(takeaways):
                with t_cols[i]:
                    st.markdown(
                        f"""
                        <div class="analytics-panel" style="padding: 14px 16px;">
                            <div class="kpi-label">{tk['category']}</div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{tk['title']}</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">{tk['metric']}</div>
                            <div class="kpi-subtext subtext-muted">{tk['summary']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# =============================================================================
# TAB 7: HISTORICAL DATA & EXPORT
# =============================================================================
with tab_data:
    st.markdown("### 🗄️ Standardized Historical Dataset & Export Hub")

    raw_df = st.session_state.get("raw_df")
    clean_df = st.session_state.get("clean_df")
    feat_df = st.session_state.get("feature_df")

    if clean_df is not None:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([4, 3, 3])

        with ctrl_col1:
            search_query = st.text_input("🔍 Search Historical Records", "")
        with ctrl_col2:
            view_mode = st.radio("Column View Mode", ["Cleaned Dataset", "Engineered Feature Matrix", "Original Raw Columns"], horizontal=True)
        with ctrl_col3:
            row_limit = st.selectbox("Display Limit", [50, 100, 500, 1000, "All Records"], index=0)

        if view_mode == "Original Raw Columns":
            display_data = raw_df.copy()
        elif view_mode == "Engineered Feature Matrix" and feat_df is not None:
            display_data = feat_df.copy()
        else:
            display_data = clean_df.copy()

        if search_query:
            mask = display_data.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            display_data = display_data[mask]

        if row_limit != "All Records":
            display_data_limited = display_data.head(int(row_limit))
        else:
            display_data_limited = display_data

        st.caption(f"Showing {len(display_data_limited):,} of {len(display_data):,} matching records.")
        st.dataframe(display_data_limited, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📥 Project Export Center")
        exp1, exp2, exp3, exp4 = st.columns(4)

        with exp1:
            csv_clean = clean_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Cleaned Dataset CSV",
                data=csv_clean,
                file_name=f"cleaned_dataset_{st.session_state['dataset_name'][:20]}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with exp2:
            if feat_df is not None:
                csv_feat = feat_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Feature Matrix CSV",
                    data=csv_feat,
                    file_name=f"features_matrix_{st.session_state['dataset_name'][:20]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with exp3:
            f_df = st.session_state.get("forecast_df")
            if f_df is not None:
                csv_fc = f_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Forecast Predictions CSV",
                    data=csv_fc,
                    file_name=f"forecast_predictions_{st.session_state['dataset_name'][:20]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with exp4:
            tr_res = st.session_state.get("training_results")
            if tr_res:
                comp_csv = build_comparison_table(tr_res, split="test").to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Model Leaderboard CSV",
                    data=comp_csv,
                    file_name=f"model_leaderboard_{st.session_state['dataset_name'][:20]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
