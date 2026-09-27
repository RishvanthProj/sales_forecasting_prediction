"""
UI Theme & Dark Dashboard Styling Module.
Provides dark quantitative aesthetics matching the Stock Trend AI visual design:
#0D1117 background, #11161D panels, #29313C thin borders, #3B82F6 primary blue,
#22C55E positive green, #EF4444 negative red, #E5E7EB text, and #8B949E muted labels.
"""

from typing import Optional, Dict, Any
import plotly.graph_objects as go


# Theme Colors
COLOR_BG = "#0D1117"
COLOR_PANEL = "#11161D"
COLOR_SECONDARY = "#151B23"
COLOR_BORDER = "#29313C"
COLOR_PRIMARY = "#3B82F6"
COLOR_CYAN = "#06B6D4"
COLOR_POSITIVE = "#22C55E"
COLOR_NEGATIVE = "#EF4444"
COLOR_WARNING = "#F59E0B"
COLOR_TEXT = "#E5E7EB"
COLOR_MUTED = "#8B949E"
COLOR_GRID = "#1F2633"


DARK_THEME_CSS = """
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0D1117 !important;
        color: #E5E7EB !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Top Application Header */
    .app-header {
        background-color: #11161D;
        border: 1px solid #29313C;
        border-radius: 6px;
        padding: 12px 20px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .app-header-left {
        display: flex;
        align-items: baseline;
        gap: 16px;
    }
    .app-header-title {
        color: #FFFFFF;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0;
    }
    .app-header-meta {
        color: #8B949E;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background-color: #11161D;
        border: 1px solid #29313C;
        border-radius: 6px;
        padding: 12px 16px;
        transition: border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #3B82F6;
    }
    .kpi-label {
        color: #8B949E;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #F3F4F6;
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    .kpi-subtext {
        font-size: 0.75rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .subtext-green { color: #22C55E; }
    .subtext-red { color: #EF4444; }
    .subtext-blue { color: #3B82F6; }
    .subtext-muted { color: #8B949E; }

    /* Analytical Panel */
    .analytics-panel {
        background-color: #11161D;
        border: 1px solid #29313C;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .panel-title {
        color: #E5E7EB;
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1F2633;
        padding-bottom: 8px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .badge-blue { background-color: rgba(59, 130, 246, 0.15); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-green { background-color: rgba(34, 197, 94, 0.15); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-red { background-color: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-gray { background-color: rgba(139, 148, 158, 0.15); color: #8B949E; border: 1px solid rgba(139, 148, 158, 0.3); }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #11161D !important;
        border: 1px solid #29313C !important;
        border-radius: 6px !important;
        padding: 4px 6px !important;
        gap: 4px !important;
        margin-bottom: 16px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #8B949E !important;
        border-radius: 4px !important;
        padding: 6px 14px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F2633 !important;
        color: #3B82F6 !important;
        border-bottom: 2px solid #3B82F6 !important;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #11161D !important;
        border-right: 1px solid #29313C !important;
    }

    /* Buttons, Popovers & Base Buttons: Simple Grey, Absolutely No Hover Effects */
    button:not([data-baseweb="tab"]),
    button[data-testid="stPopoverButton"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"],
    button[kind="secondary"],
    button[kind="primary"],
    div[data-testid="stButton"] button,
    div[data-testid="stPopover"] button,
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stFileUploader"] button,
    .stButton > button,
    .stPopover > button,
    .stDownloadButton > button {
        background-color: #1F2633 !important;
        background: #1F2633 !important;
        color: #E5E7EB !important;
        border: 1px solid #29313C !important;
        border-radius: 4px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        padding: 6px 14px !important;
        transition: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Completely eliminate any hover, focus, active background/color/border change */
    button:not([data-baseweb="tab"]):hover,
    button:not([data-baseweb="tab"]):focus,
    button:not([data-baseweb="tab"]):active,
    button[data-testid="stPopoverButton"]:hover,
    button[data-testid="stPopoverButton"]:focus,
    button[data-testid="stPopoverButton"]:active,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:focus,
    button[data-testid="stBaseButton-secondary"]:active,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:focus,
    button[data-testid="stBaseButton-primary"]:active,
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-secondary"]:focus,
    button[data-testid="baseButton-secondary"]:active,
    div[data-testid="stButton"] button:hover,
    div[data-testid="stButton"] button:focus,
    div[data-testid="stButton"] button:active,
    div[data-testid="stPopover"] button:hover,
    div[data-testid="stPopover"] button:focus,
    div[data-testid="stPopover"] button:active,
    div[data-testid="stDownloadButton"] button:hover,
    div[data-testid="stDownloadButton"] button:focus,
    div[data-testid="stDownloadButton"] button:active,
    div[data-testid="stFileUploader"] button:hover,
    div[data-testid="stFileUploader"] button:focus,
    div[data-testid="stFileUploader"] button:active,
    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active,
    .stPopover > button:hover,
    .stPopover > button:focus,
    .stPopover > button:active,
    .stDownloadButton > button:hover,
    .stDownloadButton > button:focus,
    .stDownloadButton > button:active {
        background-color: #1F2633 !important;
        background: #1F2633 !important;
        color: #E5E7EB !important;
        border: 1px solid #29313C !important;
        border-color: #29313C !important;
        box-shadow: none !important;
        outline: none !important;
        transition: none !important;
    }

    /* Button Text and Child Elements */
    button:not([data-baseweb="tab"]) p,
    button:not([data-baseweb="tab"]) span,
    button[data-testid="stPopoverButton"] p,
    button[data-testid="stPopoverButton"] span,
    button[data-testid="stBaseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] span,
    div[data-testid="stButton"] button p,
    div[data-testid="stButton"] button span,
    div[data-testid="stPopover"] button p,
    div[data-testid="stPopover"] button span,
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stDownloadButton"] button span {
        color: #E5E7EB !important;
    }

    /* Segmented Control Buttons: Simple Grey, Absolutely No Hover Effects */
    div[data-testid="stSegmentedControl"] {
        background-color: transparent !important;
    }
    div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
        background-color: #11161D !important;
        border: 1px solid #29313C !important;
        border-radius: 6px !important;
        padding: 3px !important;
        gap: 3px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background-color: #1F2633 !important;
        background: #1F2633 !important;
        color: #94A3B8 !important;
        border: 1px solid #29313C !important;
        border-radius: 4px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        padding: 5px 12px !important;
        transition: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stSegmentedControl"] button:hover,
    div[data-testid="stSegmentedControl"] button:focus,
    div[data-testid="stSegmentedControl"] button:active {
        background-color: #1F2633 !important;
        background: #1F2633 !important;
        color: #94A3B8 !important;
        border: 1px solid #29313C !important;
        border-color: #29313C !important;
        box-shadow: none !important;
        outline: none !important;
        transition: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[aria-selected="true"] {
        background-color: #29313C !important;
        background: #29313C !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-color: #3B82F6 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"]:hover,
    div[data-testid="stSegmentedControl"] button[aria-checked="true"]:focus,
    div[data-testid="stSegmentedControl"] button[aria-selected="true"]:hover,
    div[data-testid="stSegmentedControl"] button[aria-selected="true"]:focus {
        background-color: #29313C !important;
        background: #29313C !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-color: #3B82F6 !important;
        box-shadow: none !important;
        outline: none !important;
        transition: none !important;
    }
    div[data-testid="stSegmentedControl"] button p,
    div[data-testid="stSegmentedControl"] button span {
        color: inherit !important;
    }

    /* Dataframe tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #29313C !important;
        border-radius: 6px !important;
        background-color: #11161D !important;
    }

    /* Executive & Beginner-Friendly Summary Cards */
    .summary-section {
        background-color: #151B23;
        border: 1px solid #21262D;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    .summary-section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #8B949E;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .summary-dual-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 10px;
    }
    .summary-mini-card {
        background-color: #151B23;
        border: 1px solid #21262D;
        border-radius: 6px;
        padding: 8px 12px;
    }
    .summary-takeaway {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(30, 41, 59, 0.25) 100%);
        border-left: 3px solid #3B82F6;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        font-size: 0.82rem;
        line-height: 1.45;
        color: #C9D1D9;
        margin-top: 10px;
    }

    /* Transparency Box */
    .transparency-box {
        background-color: #151B23;
        border-left: 3px solid #3B82F6;
        padding: 10px 16px;
        margin-bottom: 14px;
        border-radius: 0 4px 4px 0;
        font-size: 0.8rem;
        color: #8B949E;
    }
    .transparency-box strong {
        color: #E5E7EB;
    }
</style>
"""


def apply_dark_theme(
    fig: go.Figure,
    title: Optional[str] = None,
    height: int = 420,
    show_legend: bool = True,
) -> go.Figure:
    """
    Apply high-contrast dark dashboard styling to any Plotly figure.
    Conforms to #0D1117 background and #11161D panel specifications.
    """
    layout_update: Dict[str, Any] = {
        "paper_bgcolor": COLOR_PANEL,
        "plot_bgcolor": COLOR_PANEL,
        "height": height,
        "margin": dict(l=48, r=24, t=44 if title else 24, b=40),
        "font": dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=11,
            color=COLOR_TEXT,
        ),
        "showlegend": show_legend,
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=COLOR_MUTED),
        ),
        "xaxis": dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(size=10, color=COLOR_MUTED),
            title_font=dict(size=11, color=COLOR_TEXT),
            showline=True,
            linecolor=COLOR_BORDER,
        ),
        "yaxis": dict(
            gridcolor=COLOR_GRID,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(size=10, color=COLOR_MUTED),
            title_font=dict(size=11, color=COLOR_TEXT),
            showline=True,
            linecolor=COLOR_BORDER,
        ),
        "hovermode": "x unified",
        "hoverlabel": dict(
            bgcolor="#161B22",
            bordercolor=COLOR_PRIMARY,
            font=dict(size=11, color="#FFFFFF"),
        ),
    }

    if title:
        layout_update["title"] = dict(
            text=f"<b>{title.upper()}</b>",
            font=dict(size=12, color=COLOR_TEXT),
            x=0.01,
            y=0.96,
        )

    fig.update_layout(**layout_update)
    return fig
