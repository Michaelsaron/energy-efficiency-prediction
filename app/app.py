# app/app.py
"""
Professional Streamlit UI for Energy Efficiency Prediction.

Main capabilities
-----------------
- Single-page prediction form and result
- MySQL prediction history
- SHAP explainability with graceful fallback
- Global feature importance
- Batch prediction from CSV
- Downloadable PDF prediction report
- Residual analysis
- Light and dark interface themes
"""

from __future__ import annotations

import base64
import html
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.auth import logout_button, require_auth
from src.database import (
    clear_prediction_history as db_clear_prediction_history,
    initialise_database,
    load_prediction_history as db_load_prediction_history,
    save_prediction,
)
from src.feature_engineering import BASE_FEATURES
from src.predict import EnergyPredictor, predict_heating_load
from src.utils import MODELS_DIR, load_data, load_json


# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Efficiency Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# USER AUTHENTICATION
# ---------------------------------------------------------------------
current_user = require_auth()


# SESSION STATE
# ---------------------------------------------------------------------
DEFAULT_STATE = {
    "theme_mode": "Light",
    "prediction_result": None,
    "batch_result": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "predictor" not in st.session_state:
    try:
        st.session_state.predictor = EnergyPredictor()
    except Exception as exc:
        st.session_state.predictor = None
        st.session_state.predictor_error = str(exc)


# DATA LOADERS
# ---------------------------------------------------------------------
def _version(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_csv_versioned(path_text: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path_text)


@st.cache_data(show_spinner=False)
def _read_json_versioned(path_text: str, modified: float) -> dict[str, Any]:
    del modified
    path = Path(path_text)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else {}


def latest_csv(path: Path) -> pd.DataFrame:
    return (
        _read_csv_versioned(str(path), _version(path))
        if path.exists()
        else pd.DataFrame()
    )


def latest_json(path: Path) -> dict[str, Any]:
    return _read_json_versioned(str(path), _version(path))


def load_cached_data() -> pd.DataFrame:
    dataset_path = PROJECT_ROOT / "data" / "energy_efficiency_data.csv"
    return _read_csv_versioned(str(dataset_path), _version(dataset_path))


def load_cached_metadata() -> dict[str, Any]:
    metadata_result = latest_json(PROJECT_ROOT / "models" / "model_metadata.json")
    evaluation = latest_json(PROJECT_ROOT / "models" / "evaluation_metrics.json")
    metadata_result["evaluation"] = evaluation

    ranking = latest_csv(
        PROJECT_ROOT / "outputs" / "metrics" / "deployment_ranking.csv"
    )
    if not ranking.empty:
        selected = ranking.iloc[0]
        metadata_result.update(
            {
                "best_model": selected.get(
                    "Algorithm", metadata_result.get("best_model")
                ),
                "best_r2": selected.get("R2", metadata_result.get("best_r2")),
                "best_rmse": selected.get("RMSE", metadata_result.get("best_rmse")),
                "best_mae": selected.get("MAE", metadata_result.get("best_mae")),
                "best_mse": selected.get("MSE", metadata_result.get("best_mse")),
                "training_time": selected.get(
                    "Training Time (s)",
                    metadata_result.get("training_time_seconds"),
                ),
                "cv_r2_mean": selected.get(
                    "CV R2 Mean", metadata_result.get("cv_r2_mean")
                ),
                "cv_r2_std": selected.get(
                    "CV R2 Std", metadata_result.get("cv_r2_std")
                ),
            }
        )
    return metadata_result


try:
    df = load_cached_data()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

metadata = load_cached_metadata()
evaluation_metrics = metadata.get("evaluation", {})
feature_info = latest_json(PROJECT_ROOT / "models" / "feature_info.json")

# CONSTANTS
# ---------------------------------------------------------------------
FEATURE_LABELS = {
    "Relative_Compactness": "Relative compactness",
    "Wall_Area": "Wall area",
    "Roof_Area": "Roof area",
    "Overall_Height": "Overall height",
    "Orientation": "Orientation",
    "Glazing_Area": "Glazing area",
    "Glazing_Area_Distribution": "Glazing distribution",
}

FEATURE_DESCRIPTIONS = {
    "Relative_Compactness": "How compact the overall building shape is.",
    "Wall_Area": "Total area covered by exterior walls.",
    "Roof_Area": "Total area of the roof.",
    "Overall_Height": "Height of the building.",
    "Orientation": "Primary building orientation encoded from 2 to 5.",
    "Glazing_Area": "Ratio describing window or glazed area.",
    "Glazing_Area_Distribution": "How glazing is distributed around the building.",
}

ORIENTATION_OPTIONS = {
    "North": 2,
    "East": 3,
    "South": 4,
    "West": 5,
}

GLAZING_OPTIONS = {
    "None": 0,
    "Uniform": 1,
    "North": 2,
    "East": 3,
    "South": 4,
    "West": 5,
}

TEAM = [
    ("Saron", PROJECT_ROOT / "assets" / "team" / "saron.png"),
    ("Yedidya", PROJECT_ROOT / "assets" / "team" / "yedidya.png"),
    ("Kidst", PROJECT_ROOT / "assets" / "team" / "kidst.png"),
]

PAGES = [
    "Home",
    "Project Description",
    "Dataset Information",
    "Model Information",
    "Prediction",
    "Model Insights",
    "Batch Prediction",
    "Prediction History",
    "Model Comparison",
    "Team Members",
]


# THEME
# ---------------------------------------------------------------------
is_dark = st.session_state.theme_mode == "Dark"

if is_dark:
    background = "#0B0F14"
    surface = "#11161D"
    surface_alt = "#171D25"
    text = "#FFFFFF"
    muted = "rgba(255,255,255,0.64)"
    border = "rgba(255,255,255,0.12)"
    accent = "#FFFFFF"
    accent_text = "#0B0F14"
    shadow = "0 12px 32px rgba(0,0,0,0.24)"
else:
    background = "#FFFFFF"
    surface = "#FFFFFF"
    surface_alt = "#F5F6F7"
    text = "#0B0F14"
    muted = "rgba(11,15,20,0.64)"
    border = "rgba(11,15,20,0.12)"
    accent = "#0B0F14"
    accent_text = "#FFFFFF"
    shadow = "0 12px 32px rgba(11,15,20,0.07)"

# Restrained semantic colours used only where data meaning benefits.
blue = "#3B82F6"
green = "#10B981"
amber = "#F59E0B"
red = "#EF4444"
purple = "#8B5CF6"
cyan = "#06B6D4"
chart_palette = [blue, green, amber, purple, cyan, red]

st.markdown(
    f"""
    <style>
        :root {{
            --bg: {background};
            --surface: {surface};
            --surface-alt: {surface_alt};
            --text: {text};
            --muted: {muted};
            --border: {border};
            --accent: {accent};
            --accent-text: {accent_text};
            --shadow: {shadow};
            --blue: {blue};
            --green: {green};
            --amber: {amber};
            --red: {red};
            --purple: {purple};
            --cyan: {cyan};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                         BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background: var(--bg);
            color: var(--text);
        }}

        .block-container {{
            max-width: 1280px;
            padding-top: 2.1rem;
            padding-bottom: 4rem;
        }}

        [data-testid="stDecoration"] {{ display: none; }}
        footer {{ visibility: hidden; }}
        h1, h2, h3, h4, p, label {{ color: var(--text); }}

        [data-testid="stSidebar"] {{
            background: var(--surface);
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.35rem;
        }}

        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            padding: 0.58rem 0.72rem;
            border-radius: 10px;
            margin-bottom: 0.08rem;
        }}

        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
            background: var(--surface-alt);
        }}

        [data-testid="stSidebar"] [role="radiogroup"] {{
            gap: 0.05rem;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: var(--border);
        }}

        .brand {{
            padding: 0.3rem 0 1.25rem;
        }}

        .brand-name {{
            color: var(--text);
            font-size: 1.02rem;
            font-weight: 820;
            letter-spacing: -0.03em;
        }}

        .brand-copy {{
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.5;
            margin-top: 0.35rem;
        }}

        .page-header {{
            margin-bottom: 2.1rem;
        }}

        .eyebrow {{
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 820;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }}

        .page-title {{
            color: var(--text);
            font-size: clamp(2.15rem, 5vw, 3.8rem);
            font-weight: 850;
            line-height: 1.02;
            letter-spacing: -0.065em;
            max-width: 960px;
            margin: 0;
        }}

        .page-description {{
            color: var(--muted);
            max-width: 780px;
            margin-top: 1rem;
            font-size: 1rem;
            line-height: 1.75;
        }}

        .section-title {{
            color: var(--text);
            font-size: 1.43rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 2.45rem 0 0.35rem;
        }}

        .section-copy {{
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.65;
            margin-bottom: 1.1rem;
        }}

        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 1.4rem;
        }}

        .metric-card {{
            min-height: 116px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: var(--shadow);
            padding: 1.15rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .metric-value {{
            color: var(--text);
            font-size: 1.62rem;
            font-weight: 840;
            letter-spacing: -0.045em;
            overflow-wrap: anywhere;
        }}

        .metric-label {{
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 680;
            margin-top: 0.45rem;
        }}

        .hero {{
            border: 1px solid var(--border);
            border-radius: 26px;
            background: var(--surface);
            box-shadow: var(--shadow);
            overflow: hidden;
            min-height: 460px;
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
        }}

        .hero-copy-wrap {{
            padding: clamp(1.8rem, 5vw, 4rem);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .hero-small {{
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 820;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }}

        .hero-title {{
            color: var(--text);
            font-size: clamp(2.8rem, 7vw, 5.4rem);
            line-height: 0.95;
            font-weight: 880;
            letter-spacing: -0.08em;
            margin-top: 1rem;
        }}

        .hero-description {{
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.75;
            max-width: 690px;
            margin-top: 1.35rem;
        }}

        .hero-visual {{
            min-height: 460px;
            background:
                radial-gradient(circle at 30% 25%, rgba(59,130,246,0.22), transparent 28%),
                radial-gradient(circle at 78% 22%, rgba(16,185,129,0.18), transparent 25%),
                radial-gradient(circle at 65% 78%, rgba(139,92,246,0.17), transparent 30%),
                var(--surface-alt);
            position: relative;
            overflow: hidden;
        }}

        .building {{
            position: absolute;
            left: 17%;
            bottom: 13%;
            width: 66%;
            height: 65%;
            border: 2px solid var(--text);
            border-radius: 18px 18px 8px 8px;
            background: var(--surface);
            box-shadow: var(--shadow);
        }}

        .building-roof {{
            position: absolute;
            left: 10%;
            right: 10%;
            top: -16%;
            height: 28%;
            border: 2px solid var(--text);
            background: var(--surface);
            transform: skewY(-7deg);
            border-radius: 10px;
        }}

        .window-grid {{
            position: absolute;
            left: 12%;
            right: 12%;
            top: 21%;
            bottom: 16%;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 9%;
        }}

        .window {{
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--blue);
            opacity: 0.9;
        }}

        .energy-line {{
            position: absolute;
            border-radius: 999px;
            height: 8px;
            background: linear-gradient(90deg, var(--blue), var(--green), var(--amber));
        }}

        .line-one {{ width: 45%; top: 15%; left: 9%; transform: rotate(-12deg); }}
        .line-two {{ width: 38%; right: 5%; top: 46%; transform: rotate(14deg); }}
        .line-three {{ width: 48%; left: 6%; bottom: 10%; transform: rotate(7deg); }}

        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
        }}

        .feature-card {{
            min-height: 168px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 17px;
            padding: 1.2rem;
            box-shadow: var(--shadow);
        }}

        .feature-number {{
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            font-size: 0.82rem;
            font-weight: 850;
        }}

        .feature-title {{
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 800;
            margin-top: 1rem;
        }}

        .feature-description {{
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.55;
            margin-top: 0.45rem;
        }}

        .content-row {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.7rem;
        }}

        .content-label {{
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 780;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }}

        .content-value {{
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 780;
            margin-top: 0.3rem;
        }}

        .content-copy {{
            color: var(--muted);
            font-size: 0.87rem;
            line-height: 1.65;
            margin-top: 0.5rem;
        }}

        .result-card {{
            min-height: 400px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: var(--shadow);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .result-label {{
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 820;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }}

        .result-number {{
            color: var(--text);
            font-size: clamp(4rem, 10vw, 7.7rem);
            line-height: 0.95;
            font-weight: 880;
            letter-spacing: -0.085em;
            margin-top: 1rem;
        }}

        .result-unit {{
            color: var(--muted);
            font-size: 1rem;
            font-weight: 680;
            margin-top: 0.65rem;
        }}

        .result-band {{
            display: inline-flex;
            width: fit-content;
            border-radius: 999px;
            padding: 0.42rem 0.74rem;
            color: #FFFFFF;
            font-size: 0.78rem;
            font-weight: 800;
            margin-top: 1rem;
        }}

        .result-footer {{
            border-top: 1px solid var(--border);
            padding-top: 1rem;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.6;
        }}

        .team-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }}

        .team-image {{
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: cover;
            display: block;
            filter: grayscale(100%);
        }}

        .team-name {{
            color: var(--text);
            font-size: 1rem;
            font-weight: 820;
            padding: 1rem 1.05rem 1.1rem;
        }}

        [data-testid="stForm"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
            padding: 1.4rem 1.4rem 0.3rem;
        }}

        [data-testid="stForm"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSlider"] label,
        [data-testid="stFileUploader"] label {{
            color: var(--text);
            font-weight: 720;
        }}

        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {{
            background: var(--surface);
            color: var(--text);
            border-color: var(--border);
            border-radius: 10px;
        }}

        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {{
            min-height: 2.8rem;
            border-radius: 10px;
            font-weight: 790;
            border: 1px solid var(--border);
            box-shadow: none;
        }}

        .stFormSubmitButton > button,
        .stButton > button[kind="primary"] {{
            background: var(--accent);
            color: var(--accent-text);
            border-color: var(--accent);
        }}

        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button {{
            background: var(--surface);
            color: var(--text);
        }}

        [data-testid="stDataFrame"],
        [data-testid="stExpander"],
        [data-testid="stPlotlyChart"],
        [data-testid="stFileUploaderDropzone"] {{
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            background: var(--surface);
        }}

        [data-testid="stPlotlyChart"] {{
            box-shadow: var(--shadow);
            padding: 0.4rem;
        }}

        @media (max-width: 900px) {{
            .hero {{
                grid-template-columns: 1fr;
            }}
            .hero-visual {{
                min-height: 330px;
            }}
            .feature-grid {{
                grid-template-columns: 1fr 1fr;
            }}
        }}

        @media (max-width: 640px) {{
            .block-container {{
                padding-top: 1.35rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            .feature-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# DATABASE
# ---------------------------------------------------------------------
def save_prediction_to_database(
    prediction: float,
    inputs: dict[str, Any],
) -> None:
    save_prediction(
        user_id=int(current_user["id"]),
        username=str(current_user["username"]),
        prediction=float(prediction),
        model_name=get_model_name(),
        inputs=inputs,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def load_prediction_history(limit: int = 500) -> pd.DataFrame:
    return db_load_prediction_history(
        user_id=int(current_user["id"]),
        limit=limit,
    )


def clear_prediction_history() -> None:
    db_clear_prediction_history(int(current_user["id"]))


initialise_database()


# HELPERS
# ---------------------------------------------------------------------
def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <header class="page-header">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h1 class="page-title">{html.escape(title)}</h1>
            <p class="page-description">{html.escape(description)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, description: str | None = None) -> None:
    copy = (
        f'<div class="section-copy">{html.escape(description)}</div>'
        if description
        else ""
    )
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>{copy}',
        unsafe_allow_html=True,
    )


def metric_card(value: Any, label: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-label">{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_model_name() -> str:
    value = str(metadata.get("best_model", "Not available")).replace("_", " ")
    return value if len(value) <= 24 else value[:22] + "…"


def find_target_column(data: pd.DataFrame) -> str | None:
    for candidate in ["Heating_Load", "Heating Load", "heating_load", "Y1"]:
        if candidate in data.columns:
            return candidate
    return None


def feature_count(data: pd.DataFrame) -> int:
    known = [column for column in BASE_FEATURES if column in data.columns]
    return len(known) if known else max(len(data.columns) - 2, 0)


def comparison_path() -> Path:
    return MODELS_DIR.parent / "outputs" / "metrics" / "model_comparison.csv"


def normalise_comparison_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the latest notebook column names to the names used by the UI.

    This changes data labels only; it does not change the interface design.
    """
    aliases = {
        "Algorithm Name": "Algorithm",
        "R2 Score": "R2",
        "Cross Validation R2 Mean": "CV R2 Mean",
        "Cross Validation R2 Std": "CV R2 Std",
        "Cross Validation MAE": "CV MAE",
        "Cross Validation RMSE": "CV RMSE",
        "Train-Test Gap": "Overfit Gap",
        "Train-Test R2 Gap": "Overfit Gap",
        "CV MAE Mean": "CV MAE",
        "CV RMSE Mean": "CV RMSE",
    }

    normalised = data.rename(
        columns={
            source: target
            for source, target in aliases.items()
            if source in data.columns and target not in data.columns
        }
    ).copy()

    return normalised


def style_figure(fig, height: int = 430):
    fig.update_layout(
        height=height,
        margin=dict(l=26, r=26, t=66, b=28),
        paper_bgcolor=surface,
        plot_bgcolor=surface,
        font=dict(color=text, family="Inter, Arial, sans-serif"),
        title=dict(x=0.02, xanchor="left", font=dict(size=17, color=text)),
        legend=dict(title=None),
        hoverlabel=dict(bgcolor=text, font_color=background),
    )
    fig.update_xaxes(
        gridcolor=border,
        linecolor=border,
        tickfont=dict(color=muted),
        title_font=dict(color=muted),
    )
    fig.update_yaxes(
        gridcolor=border,
        zeroline=False,
        linecolor=border,
        tickfont=dict(color=muted),
        title_font=dict(color=muted),
    )
    return fig


def demand_band(value: float) -> tuple[str, str, str]:
    if value < 15:
        return (
            "Low heating demand",
            green,
            "The estimated heating demand is relatively low.",
        )
    if value < 30:
        return (
            "Moderate heating demand",
            blue,
            "The estimated heating demand is in the middle range.",
        )
    if value < 40:
        return (
            "Elevated heating demand",
            amber,
            "The building may require more heating energy.",
        )
    return (
        "High heating demand",
        red,
        "The estimated heating demand is comparatively high.",
    )


def transformed_features(data: pd.DataFrame) -> pd.DataFrame:
    predictor = st.session_state.predictor
    if predictor is None:
        raise RuntimeError("The trained predictor could not be loaded.")
    return predictor.engineer.create_features(data.copy())


def unwrap_estimator(model):
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    if hasattr(model, "named_steps") and model.named_steps:
        return list(model.named_steps.values())[-1]
    return model


@st.cache_data(show_spinner=False)
def global_feature_importance() -> pd.DataFrame:
    predictor = EnergyPredictor()
    raw = df[BASE_FEATURES].copy()
    engineered = predictor.engineer.create_features(raw)
    estimator = unwrap_estimator(predictor.model)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float).ravel())
    else:
        raise RuntimeError(
            "The selected model does not expose built-in feature importance."
        )

    names = list(engineered.columns)
    if len(names) != len(values):
        names = [f"Feature {index + 1}" for index in range(len(values))]

    result = pd.DataFrame({"Feature": names, "Importance": values})
    result = result.sort_values("Importance", ascending=False)
    total = result["Importance"].sum()
    if total > 0:
        result["Importance"] = result["Importance"] / total
    return result


def residual_diagnostics() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "metrics" / "unseen_test_predictions.csv"
    residuals = latest_csv(path)
    if residuals.empty:
        raise RuntimeError("Run: python -m src.evaluate")
    required = {"Actual", "Predicted", "Residual", "Absolute_Error"}
    missing = required.difference(residuals.columns)
    if missing:
        raise RuntimeError(
            "Unseen-test results are missing: " + ", ".join(sorted(missing))
        )
    return residuals


def local_explanation(
    raw_inputs: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    predictor = EnergyPredictor()
    raw_frame = pd.DataFrame([raw_inputs])
    engineered = predictor.engineer.create_features(raw_frame)
    estimator = unwrap_estimator(predictor.model)

    # Preferred: SHAP values.
    try:
        import shap

        background_raw = df[BASE_FEATURES].sample(
            min(100, len(df)),
            random_state=42,
        )
        background = predictor.engineer.create_features(background_raw)

        explainer = shap.Explainer(estimator, background)
        explanation = explainer(engineered)

        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 0]
        values = values[0]

        result = pd.DataFrame(
            {
                "Feature": engineered.columns,
                "Contribution": values,
                "Value": engineered.iloc[0].to_numpy(),
            }
        ).sort_values("Contribution", key=np.abs, ascending=False)

        return result, "SHAP"

    except Exception:
        # Graceful fallback for environments without SHAP support.
        if hasattr(estimator, "feature_importances_"):
            weights = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            weights = np.asarray(estimator.coef_, dtype=float).ravel()
        else:
            weights = np.ones(engineered.shape[1], dtype=float)

        if len(weights) != engineered.shape[1]:
            weights = np.ones(engineered.shape[1], dtype=float)

        centred = engineered.iloc[0].to_numpy(dtype=float)
        contribution = centred * weights

        result = pd.DataFrame(
            {
                "Feature": engineered.columns,
                "Contribution": contribution,
                "Value": engineered.iloc[0].to_numpy(),
            }
        ).sort_values("Contribution", key=np.abs, ascending=False)

        return result, "Model-based fallback"


def create_pdf_report(
    prediction: float,
    inputs: dict[str, Any],
    explanation: pd.DataFrame | None = None,
) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PDF report generation requires reportlab. "
            "Install it with: pip install reportlab"
        ) from exc

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallMuted",
            parent=styles["BodyText"],
            fontSize=8.5,
            textColor=colors.HexColor("#64748B"),
            leading=12,
        )
    )

    band, _, band_description = demand_band(prediction)

    story = [
        Paragraph("Energy Efficiency Prediction Report", styles["Title"]),
        Spacer(1, 7 * mm),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}",
            styles["SmallMuted"],
        ),
        Spacer(1, 8 * mm),
        Paragraph("Prediction", styles["Heading2"]),
        Paragraph(
            f"<b>{prediction:.2f} kWh/m²</b>",
            styles["Heading1"],
        ),
        Paragraph(f"{band}. {band_description}", styles["BodyText"]),
        Spacer(1, 7 * mm),
        Paragraph("Submitted building parameters", styles["Heading2"]),
    ]

    input_rows = [["Parameter", "Value"]]
    for key, value in inputs.items():
        input_rows.append([FEATURE_LABELS.get(key, key), str(value)])

    input_table = Table(input_rows, colWidths=[95 * mm, 65 * mm])
    input_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B0F14")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#F8FAFC"),
                    ],
                ),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([input_table, Spacer(1, 8 * mm)])

    if explanation is not None and not explanation.empty:
        story.append(Paragraph("Leading prediction drivers", styles["Heading2"]))
        explanation_rows = [["Feature", "Contribution"]]
        for _, row in explanation.head(8).iterrows():
            explanation_rows.append(
                [str(row["Feature"]), f"{float(row['Contribution']):.4f}"]
            )

        explanation_table = Table(
            explanation_rows,
            colWidths=[105 * mm, 55 * mm],
        )
        explanation_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B0F14")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.white,
                            colors.HexColor("#F8FAFC"),
                        ],
                    ),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend([explanation_table, Spacer(1, 8 * mm)])

    story.extend(
        [
            Paragraph("Important note", styles["Heading2"]),
            Paragraph(
                "This report contains a machine-learning estimate and is not "
                "an official building energy certificate or engineering assessment.",
                styles["BodyText"],
            ),
        ]
    )

    document.build(story)
    return buffer.getvalue()


# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-name">Energy Efficiency</div>
            <div class="brand-copy">Machine learning prediction workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("SIGNED IN")
    st.markdown(f"**{html.escape(str(current_user['username']))}**")
    st.caption(html.escape(str(current_user["email"])))
    logout_button()

    st.divider()

    selected_theme = st.radio(
        "Theme",
        ["Light", "Dark"],
        horizontal=True,
        index=0 if st.session_state.theme_mode == "Light" else 1,
    )

    if selected_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = selected_theme
        st.rerun()

    st.divider()
    st.caption("ACTIVE MODEL")
    st.markdown(f"**{get_model_name()}**")

    if metadata.get("best_r2") is not None:
        st.caption(f"R² · {float(metadata['best_r2']):.4f}")


# HOME
# ---------------------------------------------------------------------
if selected_page == "Home":
    st.markdown(
        """
        <section class="hero">
            <div class="hero-copy-wrap">
                <div class="hero-small">Energy efficiency prediction</div>
                <div class="hero-title">Better building decisions from clear data.</div>
                <div class="hero-description">
                    Predict heating demand, understand why the model reached its
                    estimate, compare algorithms, and review saved prediction history
                    in one focused application.
                </div>
            </div>
            <div class="hero-visual">
                <div class="energy-line line-one"></div>
                <div class="energy-line line-two"></div>
                <div class="energy-line line-three"></div>
                <div class="building">
                    <div class="building-roof"></div>
                    <div class="window-grid">
                        <div class="window"></div><div class="window"></div><div class="window"></div>
                        <div class="window"></div><div class="window"></div><div class="window"></div>
                        <div class="window"></div><div class="window"></div><div class="window"></div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Project overview")
    columns = st.columns(4)
    overview = [
        (f"{len(df):,}", "Dataset samples"),
        (feature_count(df), "Input features"),
        (get_model_name(), "Production model"),
        (
            f"{float(metadata.get('best_r2', 0)):.5f}"
            if metadata.get("best_r2") is not None
            else "Not available",
            "Test R²",
        ),
    ]
    for column, item in zip(columns, overview):
        with column:
            metric_card(*item)

    section_heading(
        "Application capabilities",
        "Advanced features are organised into dedicated pages so the interface remains easy to understand.",
    )

    feature_items = [
        (
            "01",
            "MySQL Prediction History",
            "Predictions remain available after the browser session ends.",
            blue,
        ),
        (
            "02",
            "SHAP Explainability",
            "Understand which inputs raised or lowered an individual prediction.",
            purple,
        ),
        (
            "03",
            "Feature Importance",
            "See the variables that matter most to the trained model.",
            green,
        ),
        (
            "04",
            "Batch Prediction",
            "Upload a CSV and generate predictions for many buildings.",
            cyan,
        ),
        (
            "05",
            "Download Report",
            "Export a professional PDF summary of a prediction.",
            amber,
        ),
        (
            "06",
            "Residual Analysis",
            "Review prediction errors and model behaviour visually.",
            red,
        ),
    ]

    cards = "".join(
        (
            f'<div class="feature-card">'
            f'<div class="feature-number" style="background:{colour};">{number}</div>'
            f'<div class="feature-title">{html.escape(title)}</div>'
            f'<div class="feature-description">{html.escape(description)}</div>'
            f"</div>"
        )
        for number, title, description, colour in feature_items
    )

    st.markdown(
        f'<div class="feature-grid">{cards}</div>',
        unsafe_allow_html=True,
    )


# PROJECT DESCRIPTION
# ---------------------------------------------------------------------
elif selected_page == "Project Description":
    page_header(
        "Project description",
        "A practical energy-efficiency prediction system.",
        "This project uses supervised machine learning to estimate building heating load from architectural design characteristics.",
    )

    section_heading("Problem")
    st.markdown(
        """
        <div class="card">
            <div class="content-copy">
                Building design decisions influence the amount of energy required
                to maintain comfortable indoor conditions. Estimating heating demand
                early can support more informed and efficient design choices.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Objective")
    st.markdown(
        """
        <div class="card">
            <div class="content-copy">
                The application accepts eight architectural characteristics and
                returns a machine-learning estimate of heating load. It also provides
                explainability, batch processing, persistent history, and model diagnostics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Workflow")
    workflow = [
        "Understand the real-world business problem and expected prediction value.",
        "Load the dataset, describe every feature and identify the regression target.",
        "Split training and test data before learned preprocessing to prevent leakage.",
        "Check missing values, duplicates, distributions, skewness, correlations and outliers.",
        "Apply only three meaningful engineered features and model-specific scaling.",
        "Train and compare all required regression algorithms using held-out and cross-validation metrics.",
        "Select the deployment model using test R², RMSE and validation stability.",
        "Evaluate generalisation on unseen test data, residuals and learning curves.",
        "Serve single and batch predictions with SHAP, history, authentication and downloadable reports.",
    ]

    for index, item in enumerate(workflow, start=1):
        st.markdown(
            f"""
            <div class="content-row">
                <div class="content-label">Step {index}</div>
                <div class="content-value">{html.escape(item)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("Business and modelling justification")
    project_details = [
        (
            "Business problem",
            "Estimate building heating load early from architectural characteristics so designers can compare energy implications before construction.",
        ),
        (
            "Dataset source",
            "UCI Energy Efficiency dataset, loaded from the project data folder.",
        ),
        (
            "Problem type",
            "Regression, because Heating_Load is a continuous numerical target.",
        ),
        (
            "Business value",
            "Supports energy-conscious design decisions, comparison of alternatives and faster preliminary assessment.",
        ),
        (
            "Generalisation evidence",
            "Formal performance is reported from held-out test records that were not used to fit the model.",
        ),
    ]
    for label, value in project_details:
        st.markdown(
            f'<div class="content-row"><div class="content-value">{html.escape(label)}</div>'
            f'<div class="content-copy">{html.escape(value)}</div></div>',
            unsafe_allow_html=True,
        )


# DATASET INFORMATION
# ---------------------------------------------------------------------
elif selected_page == "Dataset Information":
    page_header(
        "Dataset information",
        "The data behind the prediction.",
        "A structured view of the dataset dimensions, features, target, summary statistics, and heating-load distribution.",
    )

    target = find_target_column(df)
    columns = st.columns(4)
    values = [
        (f"{len(df):,}", "Rows"),
        (len(df.columns), "Columns"),
        (feature_count(df), "Input features"),
        (target or "Not detected", "Heating target"),
    ]
    for column, item in zip(columns, values):
        with column:
            metric_card(*item)

    section_heading(
        "Feature guide",
        "The seven production inputs are grouped into building form, envelope, orientation, and glazing information.",
    )

    # REMOVED "Surface_Area" from building envelope group
    groups = [
        ("Building form", ["Relative_Compactness", "Overall_Height"], blue),
        ("Building envelope", ["Wall_Area", "Roof_Area"], green),
        ("Direction", ["Orientation"], amber),
        ("Glazing", ["Glazing_Area", "Glazing_Area_Distribution"], purple),
    ]

    group_columns = st.columns(4)
    for column, (group_name, keys, colour) in zip(group_columns, groups):
        with column:
            items = "".join(
                f"""
                <div style="margin-top:0.75rem;">
                    <div style="color:var(--text);font-weight:760;font-size:0.88rem;">
                        {html.escape(FEATURE_LABELS[key])}
                    </div>
                    <div style="color:var(--muted);font-size:0.78rem;line-height:1.45;margin-top:0.2rem;">
                        {html.escape(FEATURE_DESCRIPTIONS[key])}
                    </div>
                </div>
                """
                for key in keys
            )
            st.markdown(
                f"""
                <div class="card" style="min-height:260px;border-top:4px solid {colour};">
                    <div class="content-value">{html.escape(group_name)}</div>
                    {items}
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_heading("Dataset preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        section_heading("Summary statistics")
        summary = (
            df[numeric_columns]
            .describe()
            .transpose()
            .reset_index()
            .rename(columns={"index": "Variable"})
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    if target and pd.api.types.is_numeric_dtype(df[target]):
        section_heading(
            "Heating-load distribution",
            "Colour bands make the major ranges easier to distinguish without making the interface visually noisy.",
        )
        histogram = px.histogram(
            df,
            x=target,
            nbins=24,
            title="Heating-load distribution",
            labels={target: "Heating load", "count": "Buildings"},
            color_discrete_sequence=[blue],
        )
        histogram.update_traces(marker_line_color=surface, marker_line_width=0.7)
        st.plotly_chart(style_figure(histogram), use_container_width=True)


# MODEL INFORMATION
# ---------------------------------------------------------------------
elif selected_page == "Model Information":
    page_header(
        "Model information",
        "The selected production model.",
        "Review the model currently used for predictions and understand the saved evaluation metrics.",
    )

    columns = st.columns(4)
    values = [
        (get_model_name(), "Best model"),
        (
            f"{float(evaluation_metrics.get('Test R2', metadata.get('best_r2'))):.4f}"
            if evaluation_metrics.get("Test R2", metadata.get("best_r2")) is not None
            else "Not available",
            "Unseen test R²",
        ),
        (
            f"{float(evaluation_metrics.get('Test RMSE', metadata.get('best_rmse'))):.4f}"
            if evaluation_metrics.get("Test RMSE", metadata.get("best_rmse"))
            is not None
            else "Not available",
            "RMSE",
        ),
        (
            f"{float(evaluation_metrics.get('Test MAE', metadata.get('best_mae'))):.4f}"
            if evaluation_metrics.get("Test MAE", metadata.get("best_mae")) is not None
            else "Not available",
            "MAE",
        ),
    ]
    for column, item in zip(columns, values):
        with column:
            metric_card(*item)

    section_heading("Metric interpretation")
    metric_help = [
        (
            "R²",
            "How much target variation is explained by the model. Higher is better.",
        ),
        (
            "RMSE",
            "Typical prediction error with greater emphasis on larger mistakes. Lower is better.",
        ),
        ("MAE", "Average absolute prediction error. Lower is better."),
        (
            "Cross-validation",
            "Whether performance remains stable across several data partitions.",
        ),
    ]
    for name, explanation in metric_help:
        st.markdown(
            f"""
            <div class="content-row">
                <div class="content-value">{html.escape(name)}</div>
                <div class="content-copy">{html.escape(explanation)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("Selection and generalisation")
    selection_text = metadata.get(
        "selection_justification",
        "The deployment model is selected using held-out test R², then lower RMSE and lower cross-validation variability.",
    )
    st.markdown(
        f'<div class="card"><div class="content-copy">{html.escape(str(selection_text))}</div></div>',
        unsafe_allow_html=True,
    )

    gap = evaluation_metrics.get("R2 Train-Test Gap", metadata.get("train_test_r2_gap"))
    if gap is not None:
        gap_value = float(gap)
        status = (
            "Low generalisation gap"
            if abs(gap_value) <= 0.03
            else "Review possible overfitting"
        )
        st.markdown(
            f'<div class="content-row"><div class="content-value">{html.escape(status)}</div>'
            f'<div class="content-copy">Train–test R² gap: {gap_value:.4f}. '
            "A smaller absolute gap indicates more consistent behaviour on unseen data.</div></div>",
            unsafe_allow_html=True,
        )

    section_heading("Latest evaluation values")

    latest_values = [
        ("MSE", metadata.get("best_mse")),
        ("Training time (s)", metadata.get("training_time")),
        ("Cross-validation R² mean", metadata.get("cv_r2_mean")),
        ("Cross-validation R² standard deviation", metadata.get("cv_r2_std")),
    ]

    for label, value in latest_values:
        if value is not None and not pd.isna(value):
            st.markdown(
                f"""
                <div class="content-row">
                    <div class="content-value">{html.escape(label)}</div>
                    <div class="content-copy">{float(value):.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# PREDICTION - MERGED FORM AND RESULT
# ---------------------------------------------------------------------
elif selected_page == "Prediction":
    page_header(
        "Prediction",
        "Enter the building characteristics and receive the result.",
        "The form, prediction, explanation, submitted parameters, and PDF report are kept together on one page.",
    )

    form_column, result_column = st.columns([1.35, 0.85], gap="large")

    with form_column:
        with st.form("prediction_form", clear_on_submit=False):
            left, right = st.columns(2, gap="large")

            with left:
                relative_compactness = st.slider(
                    "Relative compactness",
                    min_value=0.60,
                    max_value=1.00,
                    value=0.80,
                    step=0.01,
                )
                # REMOVED Surface_Area input here
                wall_area = st.number_input(
                    "Wall area (m²)",
                    min_value=250.0,
                    max_value=450.0,
                    value=350.0,
                    step=0.5,
                )
                roof_area = st.number_input(
                    "Roof area (m²)",
                    min_value=100.0,
                    max_value=220.0,
                    value=150.0,
                    step=0.5,
                )

            with right:
                overall_height = st.selectbox(
                    "Overall height (m)",
                    options=[3.5, 7.0],
                    index=0,
                )
                orientation_label = st.selectbox(
                    "Orientation",
                    options=list(ORIENTATION_OPTIONS.keys()),
                    index=1,
                )
                glazing_area = st.slider(
                    "Glazing area ratio",
                    min_value=0.00,
                    max_value=0.40,
                    value=0.10,
                    step=0.05,
                )
                glazing_label = st.selectbox(
                    "Glazing distribution",
                    options=list(GLAZING_OPTIONS.keys()),
                    index=2,
                )

            submitted = st.form_submit_button(
                "Generate prediction",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            model_inputs = {
                "Relative_Compactness": relative_compactness,
                "Wall_Area": wall_area,
                "Roof_Area": roof_area,
                "Overall_Height": overall_height,
                "Orientation": ORIENTATION_OPTIONS[orientation_label],
                "Glazing_Area": glazing_area,
                "Glazing_Area_Distribution": GLAZING_OPTIONS[glazing_label],
            }

            display_inputs = {
                **model_inputs,
                "Orientation": orientation_label,
                "Glazing_Area_Distribution": glazing_label,
            }

            try:
                with st.spinner("Generating prediction and explanation..."):
                    prediction = float(predict_heating_load(model_inputs))
                    explanation, explanation_method = local_explanation(model_inputs)

                result = {
                    "value": prediction,
                    "model_inputs": model_inputs,
                    "display_inputs": display_inputs,
                    "explanation": explanation,
                    "explanation_method": explanation_method,
                }
                st.session_state.prediction_result = result
                save_prediction_to_database(prediction, model_inputs)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with result_column:
        result = st.session_state.prediction_result

        if result is None:
            st.markdown(
                """
                <div class="result-card">
                    <div>
                        <div class="result-label">Prediction result</div>
                        <div style="color:var(--muted);font-size:0.94rem;line-height:1.65;margin-top:1rem;">
                            Complete the building form and select “Generate prediction”.
                            The result and explanation will appear here.
                        </div>
                    </div>
                    <div class="result-footer">
                        The app stores completed predictions in the MySQL prediction history.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            band, band_colour, band_copy = demand_band(float(result["value"]))
            st.markdown(
                f"""
                <div class="result-card">
                    <div>
                        <div class="result-label">Predicted heating load</div>
                        <div class="result-number">{float(result["value"]):.2f}</div>
                        <div class="result-unit">kWh/m²</div>
                        <div class="result-band" style="background:{band_colour};">
                            {html.escape(band)}
                        </div>
                    </div>
                    <div class="result-footer">
                        {html.escape(band_copy)}
                        This is a model estimate, not an official energy certificate.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    result = st.session_state.prediction_result
    if result is not None:
        section_heading(
            "Why the model produced this result",
            f"Explanation method: {result['explanation_method']}. Positive values increase the prediction; negative values reduce it.",
        )

        explanation_df = result["explanation"].head(12).copy()
        explanation_df["Direction"] = np.where(
            explanation_df["Contribution"] >= 0,
            "Increases prediction",
            "Reduces prediction",
        )

        explanation_figure = px.bar(
            explanation_df.sort_values("Contribution"),
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Direction",
            title="Leading prediction drivers",
            color_discrete_map={
                "Increases prediction": red,
                "Reduces prediction": blue,
            },
        )
        explanation_figure.add_vline(x=0, line_color=muted, line_width=1)
        st.plotly_chart(
            style_figure(explanation_figure, 500),
            use_container_width=True,
        )

        detail_left, detail_right = st.columns([1.1, 0.9], gap="large")

        with detail_left:
            section_heading("Submitted parameters")
            display_table = pd.DataFrame(
                [
                    {
                        "Parameter": FEATURE_LABELS.get(key, key),
                        "Value": value,
                    }
                    for key, value in result["display_inputs"].items()
                ]
            )
            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True,
            )

        with detail_right:
            section_heading("Download report")
            st.markdown(
                """
                <div class="card">
                    <div class="content-copy">
                        The PDF includes the prediction, demand description,
                        submitted parameters, and leading model drivers.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            try:
                report = create_pdf_report(
                    float(result["value"]),
                    result["display_inputs"],
                    result["explanation"],
                )
                st.download_button(
                    "Download prediction report",
                    data=report,
                    file_name="energy_efficiency_prediction_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.warning(str(exc))


# MODEL INSIGHTS
# ---------------------------------------------------------------------
elif selected_page == "Model Insights":
    page_header(
        "Model insights",
        "Understand importance and prediction errors.",
        "Feature importance shows what the model relies on globally. Residual analysis reveals where predictions differ from actual values.",
    )

    section_heading(
        "Feature importance",
        "Importance is normalised so the displayed values add up to one.",
    )

    try:
        importance = global_feature_importance().head(15)
        importance_figure = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Global feature importance",
            color="Importance",
            color_continuous_scale=["#CBD5E1", blue, purple],
        )
        importance_figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(
            style_figure(importance_figure, 520),
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Feature importance is unavailable: {exc}")

    section_heading(
        "Residual analysis",
        "Residual equals actual heating load minus predicted heating load. Values near zero indicate closer predictions.",
    )

    try:
        residuals = residual_diagnostics()

        metrics = st.columns(4)
        residual_values = [
            (f"{residuals['Residual'].mean():.3f}", "Mean residual"),
            (f"{residuals['Absolute_Error'].mean():.3f}", "Mean absolute error"),
            (f"{np.sqrt(np.mean(residuals['Residual'] ** 2)):.3f}", "RMSE"),
            (f"{residuals['Residual'].std():.3f}", "Residual spread"),
        ]
        for column, item in zip(metrics, residual_values):
            with column:
                metric_card(*item)

        left, right = st.columns(2, gap="large")

        with left:
            scatter = px.scatter(
                residuals,
                x="Predicted",
                y="Residual",
                color="Absolute_Error",
                title="Residuals versus predicted values",
                color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            )
            scatter.add_hline(y=0, line_dash="dash", line_color=muted)
            st.plotly_chart(style_figure(scatter), use_container_width=True)

        with right:
            histogram = px.histogram(
                residuals,
                x="Residual",
                nbins=30,
                title="Residual distribution",
                color_discrete_sequence=[purple],
            )
            histogram.add_vline(x=0, line_dash="dash", line_color=muted)
            st.plotly_chart(style_figure(histogram), use_container_width=True)

        actual_vs_predicted = px.scatter(
            residuals,
            x="Actual",
            y="Predicted",
            color="Absolute_Error",
            title="Actual versus predicted heating load",
            color_continuous_scale=["#3B82F6", "#F59E0B", "#EF4444"],
        )
        lower = min(residuals["Actual"].min(), residuals["Predicted"].min())
        upper = max(residuals["Actual"].max(), residuals["Predicted"].max())
        actual_vs_predicted.add_shape(
            type="line",
            x0=lower,
            y0=lower,
            x1=upper,
            y1=upper,
            line=dict(color=muted, dash="dash"),
        )
        st.plotly_chart(
            style_figure(actual_vs_predicted),
            use_container_width=True,
        )

        st.caption(
            "These diagnostics use held-out test data that was not used for training. "
            "Use held-out test predictions for formal model evaluation."
        )

    except Exception as exc:
        st.warning(f"Residual analysis is unavailable: {exc}")

    section_heading(
        "Learning curve",
        "Training and validation R² across increasing sample sizes reveal whether more data may improve generalisation.",
    )
    learning_data = latest_csv(
        PROJECT_ROOT / "outputs" / "metrics" / "learning_curve.csv"
    )
    if not learning_data.empty:
        curve_long = learning_data.melt(
            id_vars="Training Samples",
            value_vars=["Train R2 Mean", "Validation R2 Mean"],
            var_name="Series",
            value_name="R2",
        )
        curve_figure = px.line(
            curve_long,
            x="Training Samples",
            y="R2",
            color="Series",
            markers=True,
            title="Training and validation learning curve",
        )
        st.plotly_chart(style_figure(curve_figure), use_container_width=True)
    else:
        st.info("Run `python -m src.evaluate` to generate the learning curve.")


# BATCH PREDICTION
# ---------------------------------------------------------------------
elif selected_page == "Batch Prediction":
    page_header(
        "Batch prediction",
        "Predict heating load for many buildings.",
        "Upload a CSV containing the seven required input columns. The app validates the file before producing downloadable results.",
    )

    section_heading("Required columns")
    required_table = pd.DataFrame(
        {
            "Column": BASE_FEATURES,
            "Description": [FEATURE_DESCRIPTIONS[key] for key in BASE_FEATURES],
        }
    )
    st.dataframe(required_table, use_container_width=True, hide_index=True)

    # REMOVED Surface_Area from template
    template = pd.DataFrame(
        [
            {
                "Relative_Compactness": 0.98,
                "Wall_Area": 294.0,
                "Roof_Area": 110.25,
                "Overall_Height": 7.0,
                "Orientation": 2,
                "Glazing_Area": 0.0,
                "Glazing_Area_Distribution": 0,
            }
        ]
    )

    template_col, _ = st.columns([1, 2])
    with template_col:
        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="batch_prediction_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [
                column for column in BASE_FEATURES if column not in batch_df.columns
            ]

            if missing:
                st.error(
                    "The uploaded file is missing required columns: "
                    + ", ".join(missing)
                )
            else:
                st.success(
                    f"The file contains {len(batch_df):,} rows and all required columns."
                )
                st.dataframe(
                    batch_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                if st.button(
                    "Run batch prediction",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Generating batch predictions..."):
                        predictor = EnergyPredictor()
                        values = predictor.predict(batch_df[BASE_FEATURES])
                        result_df = batch_df.copy()
                        result_df["Predicted_Heating_Load"] = np.asarray(
                            values,
                            dtype=float,
                        )

                    st.session_state.batch_result = result_df

        except Exception as exc:
            st.error(f"The CSV could not be processed: {exc}")

    if st.session_state.batch_result is not None:
        section_heading("Batch results")
        st.dataframe(
            st.session_state.batch_result,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download batch predictions",
            data=st.session_state.batch_result.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# PREDICTION HISTORY
# ---------------------------------------------------------------------
elif selected_page == "Prediction History":
    page_header(
        "Prediction history",
        "Persistent records stored in MySQL.",
        "Each completed single prediction is stored locally with its timestamp, result, model name, and submitted inputs.",
    )

    history = load_prediction_history()

    if history.empty:
        st.info("No saved predictions are available yet.")
    else:
        columns = st.columns(4)
        history_metrics = [
            (len(history), "Saved predictions"),
            (f"{history['prediction'].mean():.2f}", "Average prediction"),
            (f"{history['prediction'].min():.2f}", "Lowest prediction"),
            (f"{history['prediction'].max():.2f}", "Highest prediction"),
        ]
        for column, item in zip(columns, history_metrics):
            with column:
                metric_card(*item)

        section_heading("Prediction trend")
        trend = history.sort_values("created_at")
        trend_figure = px.line(
            trend,
            x="created_at",
            y="prediction",
            markers=True,
            title="Saved prediction history",
            labels={
                "created_at": "Date",
                "prediction": "Heating load (kWh/m²)",
            },
            color_discrete_sequence=[blue],
        )
        st.plotly_chart(style_figure(trend_figure), use_container_width=True)

        section_heading("Saved records")
        visible = history.drop(columns=["inputs_json"]).rename(
            columns={
                "id": "ID",
                "created_at": "Created",
                "prediction": "Heating load",
                "model_name": "Model",
            }
        )
        st.dataframe(
            visible,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download prediction history",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        clear_col, _ = st.columns([1, 3])
        with clear_col:
            if st.button("Clear history", use_container_width=True):
                clear_prediction_history()
                st.rerun()


# MODEL COMPARISON
# ---------------------------------------------------------------------
elif selected_page == "Model Comparison":
    page_header(
        "Model comparison",
        "Compare all trained algorithms.",
        "Review predictive performance, validation stability, and training time before selecting the production model.",
    )

    path = comparison_path()
    if not path.exists():
        st.warning(
            "Model comparison data was not found. Run `python -m src.train` first."
        )
        st.stop()

    comparison_df = normalise_comparison_columns(pd.read_csv(path))
    required = {"Algorithm", "R2", "RMSE", "MAE"}
    missing = required.difference(comparison_df.columns)
    if missing:
        st.error("The comparison file is missing: " + ", ".join(sorted(missing)))
        st.stop()

    best = comparison_df.sort_values("R2", ascending=False).iloc[0]
    columns = st.columns(4)
    values = [
        (best["Algorithm"], "Best model"),
        (f"{best['R2']:.5f}", "Best Test R²"),
        (f"{best['RMSE']:.5f}", "Best RMSE"),
        (len(comparison_df), "Models compared"),
    ]
    for column, item in zip(columns, values):
        with column:
            metric_card(*item)

    section_heading("Performance table")
    visible_columns = [
        column
        for column in [
            "Rank",
            "Algorithm",
            "Training Time (s)",
            "MAE",
            "MSE",
            "RMSE",
            "R2",
            "Train R2",
            "Overfit Gap",
            "CV R2 Mean",
            "CV R2 Std",
            "CV MAE",
            "CV RMSE",
            "Deployment Score",
            "Generalisation",
            "Stability",
            "Complexity",
        ]
        if column in comparison_df.columns
    ]
    st.dataframe(
        comparison_df.sort_values("R2", ascending=False)[visible_columns],
        use_container_width=True,
        hide_index=True,
    )

    section_heading("R² comparison")
    r2_figure = px.bar(
        comparison_df.sort_values("R2", ascending=True),
        x="R2",
        y="Algorithm",
        orientation="h",
        title="R² score by model",
        text="R2",
        color="R2",
        color_continuous_scale=["#CBD5E1", blue, purple],
    )
    r2_figure.update_layout(coloraxis_showscale=False)
    r2_figure.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(
        style_figure(r2_figure, max(430, 54 * len(comparison_df))),
        use_container_width=True,
    )

    section_heading("Prediction error")
    error_data = comparison_df.melt(
        id_vars="Algorithm",
        value_vars=["RMSE", "MAE"],
        var_name="Metric",
        value_name="Error",
    )
    error_figure = px.bar(
        error_data,
        x="Algorithm",
        y="Error",
        color="Metric",
        barmode="group",
        title="RMSE and MAE by model",
        color_discrete_map={"RMSE": red, "MAE": blue},
    )
    st.plotly_chart(style_figure(error_figure), use_container_width=True)

    if "Training Time (s)" in comparison_df.columns:
        section_heading("Training time")
        time_figure = px.bar(
            comparison_df.sort_values("Training Time (s)", ascending=True),
            x="Training Time (s)",
            y="Algorithm",
            orientation="h",
            title="Training time by model",
            color="Training Time (s)",
            color_continuous_scale=["#CBD5E1", green],
        )
        time_figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(
            style_figure(time_figure, max(430, 54 * len(comparison_df))),
            use_container_width=True,
        )

    if "Deployment Score" in comparison_df.columns:
        section_heading(
            "Balanced deployment ranking",
            "The score combines test accuracy, prediction error, cross-validation stability, and generalisation.",
        )
        score_figure = px.bar(
            comparison_df.sort_values("Deployment Score", ascending=True),
            x="Deployment Score",
            y="Algorithm",
            orientation="h",
            title="Balanced deployment score by model",
            text="Deployment Score",
            color="Deployment Score",
            color_continuous_scale=["#CBD5E1", blue, purple],
        )
        score_figure.update_layout(coloraxis_showscale=False)
        score_figure.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
        )
        st.plotly_chart(
            style_figure(score_figure, max(430, 54 * len(comparison_df))),
            use_container_width=True,
        )

    if "Overfit Gap" in comparison_df.columns:
        section_heading(
            "Generalisation comparison",
            "Smaller train-test gaps indicate more consistent performance on unseen data.",
        )
        gap_figure = px.bar(
            comparison_df.sort_values("Overfit Gap", ascending=False),
            x="Overfit Gap",
            y="Algorithm",
            orientation="h",
            title="Train-test R² gap by model",
            color="Overfit Gap",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        )
        gap_figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(
            style_figure(gap_figure, max(430, 54 * len(comparison_df))),
            use_container_width=True,
        )

    section_heading(
        "Why the selected model is best",
        "These checks compare accuracy, validation consistency, and generalisation.",
    )

    selected_model_name = str(metadata.get("best_model", best["Algorithm"])).replace(
        "_", " "
    )
    selected_row = comparison_df[
        comparison_df["Algorithm"]
        .astype(str)
        .str.contains(
            selected_model_name.split()[0],
            case=False,
            na=False,
        )
    ]
    selected_row = selected_row.iloc[0] if not selected_row.empty else best

    evidence_columns = st.columns(4)
    evidence_values = [
        (f"{float(selected_row['R2']):.5f}", "Test R²"),
        (f"{float(selected_row['RMSE']):.5f}", "RMSE"),
        (
            f"{float(selected_row['CV R2 Mean']):.5f}"
            if "CV R2 Mean" in selected_row.index
            else "Not available",
            "CV R² mean",
        ),
        (
            f"{float(selected_row['Overfit Gap']):.5f}"
            if "Overfit Gap" in selected_row.index
            else "Not available",
            "Train–test gap",
        ),
    ]
    for column, item in zip(evidence_columns, evidence_values):
        with column:
            metric_card(*item)

    if {"Train R2", "R2"}.issubset(comparison_df.columns):
        top_models = comparison_df.sort_values("R2", ascending=False).head(5).copy()
        train_test_data = top_models.melt(
            id_vars="Algorithm",
            value_vars=["Train R2", "R2"],
            var_name="Dataset",
            value_name="R²",
        )
        train_test_data["Dataset"] = train_test_data["Dataset"].replace(
            {"Train R2": "Training R²", "R2": "Test R²"}
        )

        train_test_figure = px.bar(
            train_test_data,
            x="Algorithm",
            y="R²",
            color="Dataset",
            barmode="group",
            title="Training and test R² for the top five models",
        )
        train_test_figure.update_yaxes(
            range=[max(0, train_test_data["R²"].min() - 0.03), 1.005]
        )
        st.plotly_chart(
            style_figure(train_test_figure),
            use_container_width=True,
        )

    if {"CV R2 Mean", "R2"}.issubset(comparison_df.columns):
        consistency_data = (
            comparison_df.sort_values("R2", ascending=False).head(5).copy()
        )
        consistency_long = consistency_data.melt(
            id_vars="Algorithm",
            value_vars=["R2", "CV R2 Mean"],
            var_name="Evaluation",
            value_name="R²",
        )
        consistency_long["Evaluation"] = consistency_long["Evaluation"].replace(
            {"R2": "Held-out Test R²", "CV R2 Mean": "Cross-validation R²"}
        )

        consistency_figure = px.bar(
            consistency_long,
            x="Algorithm",
            y="R²",
            color="Evaluation",
            barmode="group",
            title="Test and cross-validation performance",
        )
        consistency_figure.update_yaxes(
            range=[max(0, consistency_long["R²"].min() - 0.03), 1.005]
        )
        st.plotly_chart(
            style_figure(consistency_figure),
            use_container_width=True,
        )

    st.markdown(
        f"""
        <div class="card">
            <div class="content-value">Selection conclusion</div>
            <div class="content-copy">
                {html.escape(get_model_name())} ranked first using the balanced
                deployment criteria. The decision considers held-out Test R²,
                RMSE, MAE, cross-validation performance, validation variability,
                and the train–test gap rather than relying on R² alone.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download comparison data",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )


# TEAM MEMBERS
# ---------------------------------------------------------------------
else:
    page_header(
        "Team members",
        "The people behind the project.",
        "Meet the team members who contributed to the development of this energy-efficiency prediction application.",
    )

    columns = st.columns(len(TEAM), gap="large")
    for column, (name, image_path) in zip(columns, TEAM):
        with column:
            if image_path.exists():
                encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
                image_markup = (
                    f'<img class="team-image" '
                    f'src="data:image/png;base64,{encoded}" '
                    f'alt="{html.escape(name)}">'
                )
            else:
                image_markup = (
                    '<div class="team-image" '
                    'style="background:var(--surface-alt);"></div>'
                )

            st.markdown(
                f"""
                <div class="team-card">
                    {image_markup}
                    <div class="team-name">{html.escape(name)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
