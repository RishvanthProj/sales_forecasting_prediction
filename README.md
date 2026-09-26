# 📈 Product Sales Forecasting — ML System & Analytics Dashboard

A professional, standalone Machine Learning application for retail sales and demand forecasting. Built with adaptive schema detection, non-destructive data cleaning, leakage-free temporal feature engineering, chronological model validation, multi-step recursive forecasting, and an executive dark quantitative dashboard inspired by modern quantitative analytics terminals.

---

## 🚀 Business Overview & Core User Flow

In retail operations, accurate demand forecasting directly drives inventory optimization, supply chain efficiency, and margin preservation:
- **Prevents Stockouts & Lost Revenue**: Accurately anticipates seasonal demand surges and promotional uplifts.
- **Minimizes Excess Holding Costs**: Avoids over-ordering and working capital lockup.
- **Maintains Operational Transparency**: Clarifies which variables drive predictive signal and which are safely excluded.

### 🔄 End-to-End Pipeline

```
UPLOAD CSV / LOAD SAMPLE (WALMART)
          ↓
DATA AUTO-DETECTION & ALIAS RESOLUTION
          ↓
DATA QUALITY & NON-DESTRUCTIVE CLEANING
          ↓
OUTLIER & SPIKES DIAGNOSTICS (IQR BOUNDS)
          ↓
EXPLORATORY SALES & SEASONALITY ANALYSIS
          ↓
LEAKAGE-FREE FEATURE ENGINEERING
          ↓
CHRONOLOGICAL TIME-AWARE SPLIT (70 / 15 / 15)
          ↓
MODEL TRAINING (NAIVE, RIDGE, RF, XGBOOST)
          ↓
REGRESSION VALIDATION LEADERBOARD (MAE, RMSE, WAPE, R²)
          ↓
RECURSIVE MULTI-STEP SALES FORECASTING
          ↓
COMMERCIAL BUSINESS TAKEAWAYS & TRANSPARENCY
          ↓
STANDARDIZED HISTORICAL & FORECAST EXPORT
```

---

## ⚡ Quick Start & Primary Run Command

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
Execute the primary standalone command:
```bash
streamlit run app.py
```

---

## 🎨 Visual Design & Dashboard Architecture

The dashboard implements a quantitative terminal aesthetic:
- **Color Palette**: Dark charcoal background (`#0D1117`), dense analytical panels (`#11161D`), secondary panels (`#151B23`), thin borders (`#29313C`), primary blue interaction accents (`#3B82F6`), restrained green for positive indicators (`#22C55E`), and restrained red for anomalies (`#EF4444`).
- **Typography & Structure**: High-density quantitative hierarchy, compact uppercase metric labels, tabular figures, and interactive Plotly dark charts with custom gridlines (`#1F2633`).
- **Zero Hype / Serious Analytics**: No generic chatbot interfaces, no glowing marketing cards, and no fabricated confidence intervals.

---

## 📑 7 Core Application Sections

The horizontal navigation bar organizes the analytics workflow into 7 sections:

| Section | Focus Area | Key Features & Visuals |
| :--- | :--- | :--- |
| **1. OVERVIEW** | Executive Command Center | Top KPI strip, main actuals vs. recursive forecast trajectory chart with rolling average and trend lines, forecast summary panel, peak/lowest period diagnostics. |
| **2. DATA QUALITY** | Data Hygiene & Schema | Column classification table (Required, Useful, Optional, Excluded), duplicate/missingness counters, manual role override form, Before vs. After cleaning audit. |
| **3. SALES ANALYSIS** | Anomaly & Demand Drivers | IQR-based outlier box plot & histogram, unusual observations table, day-of-week & monthly seasonality charts, entity performance breakdown, promotional & holiday comparisons, statistical hypothesis testing (Welch t-test, ANOVA/Kruskal-Wallis). |
| **4. FORECASTING** | Multi-Step Future Trajectory | Recursive multi-step forecasting engine, frequency-adapted horizons (7/14/30 days, 4/8/12 weeks, 3/6/12 months), entity scope filtering, test-residual prediction bands, forecast data table with CSV export. |
| **5. MODEL VALIDATION** | Holdout Evaluation | Chronological split summary (70/15/15), comparative model leaderboard ranked by WAPE, plain-language metric cards (MAE, RMSE, WAPE, $R^2$), Actual vs. Predicted time chart, 45° parity scatter plot, residual error distribution. |
| **6. FEATURE INSIGHTS** | Feature Importance & Audit | Top-15 feature importance bar chart for tree models, feature leakage prevention audit matrix verifying `shift(1)` isolation, Data Science Transparency box (what was used vs. excluded and why), commercial takeaways. |
| **7. HISTORICAL DATA** | Data Inspection & Export | Searchable records table with limit controls, toggle between Standardized/Cleaned, Feature Matrix, and Raw views, and export center for all pipeline artifacts. |

---

## 🧠 Key Data Science & ML Methodologies

### 1. Automatic Schema Detection & Column Role Classification
The system never hardcodes column names. It utilizes normalized candidate pattern matching, datetime parsers, and numeric type validations to map:
- **Sales Target**: `Sales`, `Weekly_Sales`, `Monthly_Sales`, `Revenue`, `Total_Sales`, `Amount`, etc.
- **Date Index**: `Date`, `Order_Date`, `Transaction_Date`, `Sales_Date`, `Timestamp`, `Week`, `Month`.
- **Entity Scope**: `Store`, `Store_ID`, `Branch`, `Outlet`, `Product`, `SKU`, `Item`.
- **Business Factors**: `Holiday_Flag`, `Holiday`, `Discount`, `Promotion`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`.
- **Excluded Non-Predictive Columns**: Automatically flags IDs, row indices, UUIDs, single-value constant fields, and high-cardinality text strings.

### 2. Non-Destructive Data Cleaning & Outlier Diagnostics
- Converts and validates calendar dates, sorting chronologically by entity and timestamp.
- Safely handles missing features using group-aware forward/backward filling and median imputation.
- Detects anomalies using Interquartile Range ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$).
- **Crucial Rule**: Retail sales spikes are treated as genuine seasonal surges rather than automatically discarded. Outliers are retained by default, with optional capping available.

### 3. Leakage-Safe Feature Engineering
For any target observation at timestamp $T$:
- **Calendar Signals**: Year, quarter, month, day, weekday, day-of-year, week-of-year, is_weekend, month start/end.
- **Historical Lags**: Strictly shifted by $k \ge 1$: $\text{lag}_k(S_T) = S_{T-k}$.
- **Rolling Windows**: Strictly shifted by 1 step before rolling window computation:
  $$\text{rolling\_mean}(S_T) = \frac{1}{W} \sum_{i=1}^{W} S_{T-i}$$
  *Never includes contemporaneous target $S_T$.*
- **Entity-Aware Partitioning**: If store or product entities exist, lags and rolling aggregations are calculated independently per entity.

### 4. Machine Learning Models & Chronological Split
- **Chronological Split**: 70% Train, 15% Validation, 15% Test strictly preserving temporal sequence without random shuffling.
- **Model Candidates**:
  1. `Naive Baseline`: Historical lag-1 benchmark.
  2. `Ridge Regression`: Linear baseline with $L_2$ regularization.
  3. `Random Forest Regressor`: Ensemble tree model capturing non-linear interactions.
  4. `XGBoost Regressor`: Gradient boosted decision trees for state-of-the-art predictive performance.
- **Regression Metric Clarity**: Emphasizes MAE, RMSE, WAPE, and $R^2$ Score. Clarifies that $R^2$ represents explained variance rather than classification "accuracy".

### 5. Recursive Multi-Step Forecasting Engine
Iteratively generates forecasts across horizon $H$:
$$\hat{y}_{T+1} = f(\text{History}_{1 \dots T})$$
$$\hat{y}_{T+2} = f(\text{History}_{1 \dots T} \cup \{\hat{y}_{T+1}\})$$
Dynamically updates historical lag buffers step by step without mutating original datasets.

---

## 📂 Repository File Structure

```
product-sales-forecasting/
├── app.py                      # Standalone Streamlit Quantitative Dashboard
├── requirements.txt            # Python dependencies (Streamlit, Scikit-learn, XGBoost, Plotly)
├── README.md                   # Project documentation
│
├── data/
│   ├── sample/
│   │   ├── walmart_sales.csv   # Primary Walmart Store & Macroeconomic Dataset (6,435 rows)
│   │   └── sample_sales.csv    # Multi-Store Daily Retail Dataset with Orders (1,548 rows)
│   └── uploads/                # Temporary directory for custom user CSV uploads
│
├── src/                        # Clean Modular Machine Learning Engine
│   ├── __init__.py
│   ├── loader.py               # Robust CSV ingestion & frequency detection
│   ├── schema_detector.py      # Candidate pattern matching & column role classification
│   ├── cleaner.py              # Chronological sorting, deduplication, missing imputation
│   ├── eda.py                  # Time-series trend, seasonality, and entity analysis
│   ├── outliers.py             # IQR anomaly detection & unusual observations extraction
│   ├── features.py             # Leakage-safe lag and rolling feature engineering pipeline
│   ├── models.py               # Chronological train/val/test split & regression models
│   ├── validation.py           # Model leaderboard, diagnostics, and feature importance
│   ├── forecast.py             # Multi-step recursive forecasting engine
│   ├── insights.py             # Statistical hypothesis testing & commercial takeaways
│   └── ui_theme.py             # Quantitative dark CSS & Plotly layout styling
│
├── artifacts/
│   ├── models/                 # Saved model serialization files
│   ├── metrics/                # Benchmark metrics JSON files
│   ├── preprocessors/          # Fitted feature pipelines
│   └── exports/                # Exported CSVs
│
└── tests/
    ├── test_master_suite.py    # Automated test suite covering all 10 mandatory scenarios
    └── test_pipeline.py        # Legacy integration test suite
```

---

## 🧪 Automated Testing & Scenarios Validated

All 15 master scenarios are rigorously tested with automated test suites:

```bash
pytest tests/test_master_suite.py
```

| # | Test Scenario | Verified Behavior | Status |
| :---: | :--- | :--- | :---: |
| 1 | **Date + Sales** | Gracefully trains univariate time-series model with calendar & lag features. | ✅ Passed |
| 2 | **Date + Store + Sales** | Groups historical lags and rolling windows independently per store entity. | ✅ Passed |
| 3 | **Date + Store + Orders + Sales** | Uses order volume as supporting predictive feature without leakage. | ✅ Passed |
| 4 | **Walmart-Style Dataset** | Full real-world Walmart dataset (Store, Weekly_Sales, Holiday_Flag, Macroeconomics). | ✅ Passed |
| 5 | **Extra Unwanted Columns** | Identifies and excludes arbitrary IDs, UUIDs, URLs, and constant columns. | ✅ Passed |
| 6 | **Missing Values** | Safely imputes missing values in features and targets without pipeline failure. | ✅ Passed |
| 7 | **Duplicated Rows** | Deduplicates records and reports before/after cleaning metrics. | ✅ Passed |
| 8 | **Outliers / Retail Spikes** | Detects retail demand spikes via IQR bounds without destructive deletion. | ✅ Passed |
| 9 | **Different Date Names** | Automatically maps `Transaction_Date`, `Order_Date`, etc. | ✅ Passed |
| 10 | **Different Sales Names** | Automatically maps `Weekly_Sales`, `Total_Revenue`, `Amount`, etc. | ✅ Passed |
| 11 | **Dataset Scope Scale Continuity** | Forecast curve seamlessly connects to historical actual sales without scale collapse. | ✅ Passed |
| 12 | **Frequency-Aware Horizons** | Converts 1M, 3M, 6M into exact frequency periods (Daily: 30/90/180; Weekly: 4/13/26; Monthly: 1/3/6). | ✅ Passed |
| 13 | **WAPE-Based Forecast Accuracy** | Rigorously calculates `(1 - WAPE) * 100` on unseen holdout test split. | ✅ Passed |
| 14 | **Historical Range Filtering** | Dynamically filters historical view by 1M, 3M, 6M, and Overall using pandas DateOffset. | ✅ Passed |
| 15 | **Outlier Table & Deviation** | Computes absolute deviation from normal range and assigns categorical outlier types. | ✅ Passed |

---

## ⚖️ Real Limitations & Scope Boundaries

1. **Univariate vs. External Macroeconomic Projections**: In recursive multi-step forecasting without exogenous future forecasts (e.g. future CPI or Fuel Price values), the pipeline propagates the last observed macroeconomic conditions into the future buffer.
2. **Extreme Retail Discontinuities**: While the model accommodates regular historical seasonality and designated holidays, unprecedented external events (e.g. natural disasters or unexpected supply chain collapses) outside the historical training distribution cannot be predicted.
3. **Data History Requirements**: To compute meaningful lag features (e.g., lag-7 or lag-14), the dataset must contain at least 15 to 20 chronological time intervals.
# sales_forecasting_prediction
