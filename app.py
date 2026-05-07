"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CopaVision AI  |  Match Outcome Predictor  |  Phase 1             ║
║          Streamlit Dashboard — Production Ready                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:
    streamlit run app.py

Dependencies:
    pip install streamlit pandas numpy scikit-learn plotly joblib

File structure expected:
    app.py              ← this file
    models/
        copavision_rf.pkl
        copavision_lr.pkl
    team_stats.json     ← pre-computed Elo + form stats per team
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CopaVision AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark football analytics theme
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global theme ── */
:root {
    --bg-deep:    #050b14;
    --bg-card:    #0e1621;
    --bg-panel:   #121d2e;
    --border:     #1e3050;
    --accent:     #00d4ff;
    --accent2:    #00ff9d;
    --accent3:    #ff6b35;
    --gold:       #ffd700;
    --text:       #e8f1ff;
    --muted:      #5a7a9a;
    --win:        #00ff9d;
    --loss:       #ff4560;
    --draw:       #00d4ff;
}

/* ── App background ── */
.stApp {
    background-color: var(--bg-deep);
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(0,212,255,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(0,255,157,0.04) 0%, transparent 50%);
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {color: var(--text) !important;}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {border-color: var(--accent);}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
}
[data-testid="stMetricLabel"] {color: var(--muted) !important; font-size: 0.75rem !important;}
[data-testid="stMetricDelta"] {font-size: 0.8rem !important;}

/* ── Selectbox & inputs ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {border-color: var(--accent) !important;}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    color: #050b14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(0,212,255,0.3) !important;
}
.stButton > button:active {transform: translateY(0px) !important;}

/* ── Tabs ── */
[data-testid="stTab"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px;
    color: var(--muted) !important;
}
[data-testid="stTab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Dividers ── */
hr {border-color: var(--border) !important; margin: 1.2rem 0 !important;}

/* ── Info/warning boxes ── */
.stAlert {background: var(--bg-card) !important; border-radius: 10px !important;}

/* ── Custom card class (used via markdown/html) ── */
.cv-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.cv-card-accent {border-left: 3px solid var(--accent);}
.cv-card-win   {border-left: 3px solid var(--win);}
.cv-card-draw  {border-left: 3px solid var(--draw);}
.cv-card-loss  {border-left: 3px solid var(--loss);}

/* ── Section headings ── */
.cv-heading {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--text);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.cv-subheading {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.05rem;
    color: var(--muted);
    letter-spacing: 0.5px;
    margin-bottom: 1.2rem;
}

/* ── Result badge ── */
.result-badge {
    display: inline-block;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 28px;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.badge-home  {background: rgba(0,255,157,0.15); color: #00ff9d; border: 1px solid #00ff9d;}
.badge-away  {background: rgba(255,107,53,0.15); color: #ff6b35; border: 1px solid #ff6b35;}
.badge-draw  {background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid #00d4ff;}

/* ── Spinner override ── */
.stSpinner > div {border-top-color: var(--accent) !important;}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpanderToggleIcon"] {color: var(--accent) !important;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "home_recent_points", "away_recent_points",
    "home_avg_goals_scored", "away_avg_goals_scored",
    "home_avg_goals_conceded", "away_avg_goals_conceded",
    "home_rolling_gd", "away_rolling_gd",
    "elo_diff", "home_elo", "away_elo",
    "neutral_venue", "tournament_importance",
]

TOURNAMENT_IMPORTANCE_MAP = {
    "FIFA World Cup":                    5,
    "UEFA Euro (EURO)":                  5,
    "Copa America":                      5,
    "AFC Asian Cup":                     4,
    "African Cup of Nations":            4,
    "Gold Cup (CONCACAF)":               4,
    "FIFA World Cup Qualification":      3,
    "UEFA Euro Qualification":           3,
    "UEFA Nations League":               3,
    "CONMEBOL Qualifying":               3,
    "CAF Qualification":                 2,
    "Regional Tournament":               2,
    "International Friendly":            1,
}

# Plotly dark theme base
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "home":  "#00ff9d",
    "away":  "#ff6b35",
    "draw":  "#00d4ff",
    "bg":    "#0e1621",
    "panel": "#121d2e",
    "border":"#1e3050",
    "muted": "#5a7a9a",
    "text":  "#e8f1ff",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained ML models. Cached so they only load once per session."""
    models = {}
    model_dir = BASE_DIR / "models"
    for name, fname in [("Random Forest", "copavision_rf.pkl"),
                         ("Logistic Regression", "copavision_lr.pkl")]:
        path = model_dir / fname
        if path.exists():
            models[name] = joblib.load(path)
        else:
            st.error(f"Model not found: {path}")
    return models


@st.cache_data(show_spinner=False)
def load_team_stats() -> dict:
    """Load pre-computed Elo + form stats per team."""
    path = BASE_DIR / "team_stats.json"
    if not path.exists():
        st.error("team_stats.json not found. Run the Phase 1 pipeline first.")
        return {}
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def get_team_features(team: str, stats: dict) -> dict:
    """Return a team's feature dict, with safe defaults for unknown teams."""
    if team in stats:
        s = stats[team]
        return {
            "elo":              s["elo"],
            "recent_points":    s["recent_points"],
            "avg_scored":       s["avg_scored"],
            "avg_conceded":     s["avg_conceded"],
            "rolling_gd":       s["rolling_gd"],
            "matches_played":   s.get("matches_played", 0),
        }
    # Fallback for teams with no history
    return {
        "elo": 1500.0, "recent_points": 1.0,
        "avg_scored": 1.0, "avg_conceded": 1.0,
        "rolling_gd": 0.0, "matches_played": 0,
    }


def build_feature_vector(home_team: str, away_team: str,
                          neutral: bool, tournament: str,
                          stats: dict) -> np.ndarray:
    """
    Construct the 13-feature vector that matches the training pipeline exactly.
    Feature order MUST match FEATURE_COLS — any reordering breaks the model.
    """
    h = get_team_features(home_team, stats)
    a = get_team_features(away_team, stats)
    imp = TOURNAMENT_IMPORTANCE_MAP.get(tournament, 2)

    vector = [
        h["recent_points"],      # home_recent_points
        a["recent_points"],      # away_recent_points
        h["avg_scored"],         # home_avg_goals_scored
        a["avg_scored"],         # away_avg_goals_scored
        h["avg_conceded"],       # home_avg_goals_conceded
        a["avg_conceded"],       # away_avg_goals_conceded
        h["rolling_gd"],         # home_rolling_gd
        a["rolling_gd"],         # away_rolling_gd
        h["elo"] - a["elo"],     # elo_diff
        h["elo"],                # home_elo
        a["elo"],                # away_elo
        int(neutral),            # neutral_venue
        imp,                     # tournament_importance
    ]
    return np.array(vector, dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_probability_donut(home_prob: float, draw_prob: float,
                            away_prob: float, home_team: str,
                            away_team: str) -> go.Figure:
    """Donut chart showing Home / Draw / Away probability breakdown."""
    fig = go.Figure(go.Pie(
        labels=[f"{home_team} Win", "Draw", f"{away_team} Win"],
        values=[home_prob, draw_prob, away_prob],
        hole=0.62,
        marker_colors=[COLORS["home"], COLORS["draw"], COLORS["away"]],
        textinfo="label+percent",
        textfont=dict(family="Barlow Condensed", size=14, color=COLORS["text"]),
        hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>",
        direction="clockwise",
        sort=False,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20, l=20, r=20),
        height=300,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{max(home_prob, draw_prob, away_prob)*100:.0f}%</b>",
            x=0.5, y=0.5, font=dict(size=30, family="Barlow Condensed",
                                     color=COLORS["text"]),
            showarrow=False,
        )],
    )
    return fig


def make_probability_bars(home_prob: float, draw_prob: float,
                            away_prob: float, home_team: str,
                            away_team: str) -> go.Figure:
    """Horizontal probability bar chart."""
    labels = [f"{home_team} Win", "Draw", f"{away_team} Win"]
    values = [home_prob * 100, draw_prob * 100, away_prob * 100]
    bar_colors = [COLORS["home"], COLORS["draw"], COLORS["away"]]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family="Barlow Condensed", size=16, color=COLORS["text"]),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 105], showgrid=False, visible=False),
        yaxis=dict(showgrid=False, tickfont=dict(family="Barlow Condensed",
                                                  size=14, color=COLORS["text"])),
        margin=dict(t=10, b=10, l=10, r=70),
        height=200,
        bargap=0.35,
    )
    return fig


def make_team_radar(h_feats: dict, a_feats: dict,
                     home_team: str, away_team: str) -> go.Figure:
    """Radar chart comparing two teams across key metrics (normalised 0–1)."""
    categories = ["Elo Rating", "Recent Form", "Attack", "Defence\n(inverted)", "Goal Diff"]

    def norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9)))

    h_vals = [
        norm(h_feats["elo"],           1200, 2200),
        norm(h_feats["recent_points"], 0,    3),
        norm(h_feats["avg_scored"],    0,    4),
        norm(1 / (h_feats["avg_conceded"] + 0.5), 0, 2),  # lower conceded → better
        norm(h_feats["rolling_gd"],    -3,   3),
    ]
    a_vals = [
        norm(a_feats["elo"],           1200, 2200),
        norm(a_feats["recent_points"], 0,    3),
        norm(a_feats["avg_scored"],    0,    4),
        norm(1 / (a_feats["avg_conceded"] + 0.5), 0, 2),
        norm(a_feats["rolling_gd"],    -3,   3),
    ]

    fig = go.Figure()
    for name, vals, color in [
        (home_team, h_vals, COLORS["home"]),
        (away_team, a_vals, COLORS["away"]),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if color.startswith("rgb") else "rgba(0,255,157,0.15)" if color == COLORS["home"] else "rgba(255,107,53,0.15)",
            line=dict(color=color, width=2),
            name=name,
            hovertemplate=f"<b>{name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(14,22,33,0.6)",
            radialaxis=dict(visible=True, range=[0, 1],
                            showticklabels=False, gridcolor=COLORS["border"]),
            angularaxis=dict(
                tickfont=dict(family="Barlow Condensed", size=12, color=COLORS["text"]),
                gridcolor=COLORS["border"],
            ),
        ),
        legend=dict(
            font=dict(family="Barlow Condensed", color=COLORS["text"], size=13),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=380,
        margin=dict(t=30, b=30, l=30, r=30),
    )
    return fig


def make_feature_importance_chart(model) -> go.Figure:
    """Horizontal bar chart for RF feature importances."""
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    idx = np.argsort(importances)
    labels = [FEATURE_COLS[i].replace("_", " ").title() for i in idx]
    vals   = importances[idx]

    # Colour by feature group
    palette = []
    for i in idx:
        col = FEATURE_COLS[i]
        if "elo" in col:           palette.append(COLORS["draw"])
        elif "points" in col:      palette.append(COLORS["home"])
        elif "goal" in col or "gd" in col: palette.append("#d2a8ff")
        else:                      palette.append(COLORS["muted"])

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker=dict(color=palette),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"], family="DM Sans"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, visible=False, range=[0, vals.max() * 1.2]),
        yaxis=dict(tickfont=dict(family="DM Sans", size=11, color=COLORS["text"])),
        margin=dict(t=10, b=10, l=10, r=70),
        height=420,
    )
    return fig


def make_elo_gauge(elo: float, team: str, color: str) -> go.Figure:
    """Gauge chart displaying Elo rating."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=elo,
        number=dict(font=dict(family="Barlow Condensed", size=36, color=COLORS["text"])),
        gauge=dict(
            axis=dict(range=[1000, 2300], tickwidth=1,
                      tickcolor=COLORS["muted"],
                      tickfont=dict(color=COLORS["muted"], size=10)),
            bar=dict(color=color, thickness=0.3),
            bgcolor=COLORS["bg"],
            borderwidth=0,
            steps=[
                dict(range=[1000, 1400], color="rgba(90,122,154,0.1)"),
                dict(range=[1400, 1700], color="rgba(90,122,154,0.15)"),
                dict(range=[1700, 2300], color="rgba(90,122,154,0.2)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.7, value=elo),
        ),
        title=dict(text=team, font=dict(family="Barlow Condensed",
                                         size=16, color=COLORS["muted"])),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=40, b=10, l=30, r=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RESULT FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_result_badge(prediction: int, home_team: str, away_team: str) -> str:
    if prediction == 0:
        return f'<span class="result-badge badge-home">⚽ {home_team} Win</span>'
    elif prediction == 1:
        return f'<span class="result-badge badge-away">⚽ {away_team} Win</span>'
    else:
        return f'<span class="result-badge badge-draw">🤝 Draw</span>'


def confidence_label(prob: float) -> str:
    if prob >= 0.70: return "🔥 High Confidence"
    if prob >= 0.50: return "📊 Moderate Confidence"
    if prob >= 0.35: return "⚖️  Low Confidence"
    return "🎲 Uncertain"


def form_bar(recent_points: float) -> str:
    """Generate a simple emoji form representation."""
    # recent_points is avg of last 5 games; 3=W, 1=D, 0=L
    filled = int(round(recent_points / 3 * 5))
    return "🟢" * filled + "⚫" * (5 - filled)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(team_stats: dict, models: dict):
    with st.sidebar:
        # Logo / brand
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
            <div style="font-family:'Barlow Condensed',sans-serif;
                        font-size:2rem; font-weight:900; color:#00d4ff;
                        letter-spacing:3px;">COPA<span style="color:#00ff9d;">VISION</span></div>
            <div style="font-family:'Barlow Condensed',sans-serif;
                        font-size:0.85rem; color:#5a7a9a; letter-spacing:4px;">
                A I  ·  P H A S E  1</div>
        </div>
        <hr style="border-color:#1e3050; margin:0 0 1.5rem;">
        """, unsafe_allow_html=True)

        # Navigation
        st.markdown('<p style="font-family:\'Barlow Condensed\',sans-serif; '
                    'font-size:0.7rem; color:#5a7a9a; letter-spacing:3px; '
                    'text-transform:uppercase; margin-bottom:0.5rem;">NAVIGATION</p>',
                    unsafe_allow_html=True)

        page = st.radio(
            label="page",
            options=["⚽  Match Predictor", "📊  Team Explorer",
                     "🧠  Model Insights", "ℹ️   About"],
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Model selector
        st.markdown('<p style="font-family:\'Barlow Condensed\',sans-serif; '
                    'font-size:0.7rem; color:#5a7a9a; letter-spacing:3px; '
                    'text-transform:uppercase; margin-bottom:0.5rem;">ML MODEL</p>',
                    unsafe_allow_html=True)
        model_choice = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Stats strip
        st.markdown(f"""
        <div style="text-align:center;">
            <p style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem;
               color:#5a7a9a; letter-spacing:3px; text-transform:uppercase; margin-bottom:0.8rem;">
               DATABASE</p>
            <div style="display:flex; justify-content:space-around;">
                <div>
                    <div style="font-family:'Barlow Condensed',sans-serif;
                         font-size:1.6rem; font-weight:700; color:#00d4ff;">
                         {len(team_stats)}</div>
                    <div style="font-size:0.7rem; color:#5a7a9a;">Teams</div>
                </div>
                <div>
                    <div style="font-family:'Barlow Condensed',sans-serif;
                         font-size:1.6rem; font-weight:700; color:#00ff9d;">
                         19k+</div>
                    <div style="font-size:0.7rem; color:#5a7a9a;">Matches</div>
                </div>
                <div>
                    <div style="font-family:'Barlow Condensed',sans-serif;
                         font-size:1.6rem; font-weight:700; color:#d2a8ff;">
                         2020</div>
                    <div style="font-size:0.7rem; color:#5a7a9a;">Latest</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Phase roadmap
        st.markdown("""
        <p style="font-family:'Barlow Condensed',sans-serif; font-size:0.7rem;
           color:#5a7a9a; letter-spacing:3px; text-transform:uppercase; margin-bottom:0.8rem;">
           ROADMAP</p>
        <div style="font-size:0.82rem; line-height:2;">
            <span style="color:#00ff9d;">✓</span> Phase 1 · Match Predictor<br>
            <span style="color:#5a7a9a;">○</span> Phase 2 · Player Dashboard<br>
            <span style="color:#5a7a9a;">○</span> Phase 3 · Sentiment Tracker<br>
            <span style="color:#5a7a9a;">○</span> Phase 4 · Live Match Feed
        </div>
        """, unsafe_allow_html=True)

    return page, model_choice


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — MATCH PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
def page_match_predictor(models: dict, team_stats: dict, model_choice: str):
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cv-heading">⚽ Match Outcome Predictor</div>
    <div class="cv-subheading">
        Select two international teams and get AI-powered match predictions powered by Elo ratings and ML
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e3050;">', unsafe_allow_html=True)

    # ── Input Section ─────────────────────────────────────────────────────────
    all_teams = sorted(team_stats.keys())
    popular = ["Brazil", "Argentina", "France", "Germany", "Spain",
               "England", "Portugal", "Italy", "Netherlands", "Belgium",
               "Croatia", "Uruguay", "Mexico", "USA", "Japan",
               "South Korea", "Senegal", "Morocco", "Australia", "Poland"]
    # Put popular teams first in the dropdown
    ordered_teams = [t for t in popular if t in all_teams] + \
                    [t for t in all_teams if t not in popular]

    col_l, col_mid, col_r = st.columns([5, 1, 5])

    with col_l:
        st.markdown('<p style="font-family:\'Barlow Condensed\',sans-serif; '
                    'font-size:0.75rem; color:#5a7a9a; letter-spacing:2px; '
                    'text-transform:uppercase;">HOME TEAM</p>', unsafe_allow_html=True)
        home_team = st.selectbox("Home Team", ordered_teams,
                                  index=ordered_teams.index("Brazil"),
                                  label_visibility="collapsed", key="home")

    with col_mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center; font-family:\'Barlow Condensed\','
                    'sans-serif; font-size:1.8rem; color:#5a7a9a; padding-top:0.3rem;">VS</div>',
                    unsafe_allow_html=True)

    with col_r:
        st.markdown('<p style="font-family:\'Barlow Condensed\',sans-serif; '
                    'font-size:0.75rem; color:#5a7a9a; letter-spacing:2px; '
                    'text-transform:uppercase;">AWAY TEAM</p>', unsafe_allow_html=True)
        away_default = ordered_teams.index("Argentina") if "Argentina" in ordered_teams else 1
        away_team = st.selectbox("Away Team", ordered_teams,
                                  index=away_default,
                                  label_visibility="collapsed", key="away")

    # Same-team guard
    if home_team == away_team:
        st.warning("⚠️ Home and Away teams must be different. Please select two distinct teams.")
        return

    # Tournament & venue row
    col_t, col_n = st.columns([3, 1])
    with col_t:
        st.markdown('<p style="font-family:\'Barlow Condensed\',sans-serif; '
                    'font-size:0.75rem; color:#5a7a9a; letter-spacing:2px; '
                    'text-transform:uppercase; margin-top:0.6rem;">TOURNAMENT</p>',
                    unsafe_allow_html=True)
        tournament = st.selectbox(
            "Tournament",
            list(TOURNAMENT_IMPORTANCE_MAP.keys()),
            index=0,
            label_visibility="collapsed",
        )
    with col_n:
        st.markdown('<p style="font-family:\'Barlow Condensed\',sans-serif; '
                    'font-size:0.75rem; color:#5a7a9a; letter-spacing:2px; '
                    'text-transform:uppercase; margin-top:0.6rem;">NEUTRAL VENUE</p>',
                    unsafe_allow_html=True)
        neutral = st.toggle("Neutral Ground", value=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Button ─────────────────────────────────────────────────────────
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        predict_clicked = st.button("🔮  Generate Prediction", use_container_width=True)

    st.markdown('<hr style="border-color:#1e3050;">', unsafe_allow_html=True)

    # ── Prediction Output ─────────────────────────────────────────────────────
    if predict_clicked:
        model = models[model_choice]
        h_feats = get_team_features(home_team, team_stats)
        a_feats = get_team_features(away_team, team_stats)

        with st.spinner("⚙️  Running prediction engine…"):
            time.sleep(0.6)  # Small delay for UX polish
            X = build_feature_vector(home_team, away_team, neutral, tournament, team_stats)
            probs = model.predict_proba(X)[0]
            prediction = int(np.argmax(probs))

        home_prob, away_prob, draw_prob = probs[0], probs[1], probs[2]
        max_prob = max(home_prob, draw_prob, away_prob)

        # ── Result headline ────────────────────────────────────────────────────
        badge = get_result_badge(prediction, home_team, away_team)
        conf  = confidence_label(max_prob)
        venue_label = "🌐 Neutral Ground" if neutral else "🏟️ Home Advantage"
        imp = TOURNAMENT_IMPORTANCE_MAP.get(tournament, 2)

        st.markdown(f"""
        <div class="cv-card cv-card-accent" style="text-align:center; padding: 1.8rem 2rem;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.8rem;
                 color:#5a7a9a; letter-spacing:3px; text-transform:uppercase; margin-bottom:0.6rem;">
                PREDICTED OUTCOME</div>
            {badge}
            <div style="margin-top:0.8rem; font-size:1.0rem; color:#e8f1ff;">
                {conf} &nbsp;·&nbsp;
                <span style="color:#00d4ff;">{max_prob*100:.1f}%</span> confidence
            </div>
            <div style="margin-top:0.4rem; font-size:0.82rem; color:#5a7a9a;">
                {venue_label} &nbsp;·&nbsp; Tournament Weight: {'⭐' * imp}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability breakdown ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        tab_prob, tab_radar, tab_insights = st.tabs(
            ["📊  Probabilities", "🕸️  Team Radar", "🔍  Match Insights"]
        )

        with tab_prob:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(
                    make_probability_donut(home_prob, draw_prob, away_prob,
                                           home_team, away_team),
                    use_container_width=True, config={"displayModeBar": False},
                )
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(
                    make_probability_bars(home_prob, draw_prob, away_prob,
                                          home_team, away_team),
                    use_container_width=True, config={"displayModeBar": False},
                )

            # Metric cards row
            m1, m2, m3 = st.columns(3)
            m1.metric(f"⚽ {home_team} Win", f"{home_prob*100:.1f}%",
                      delta=f"{'+'  if home_prob > 0.33 else ''}{(home_prob-0.33)*100:.1f}% vs base")
            m2.metric("🤝 Draw", f"{draw_prob*100:.1f}%",
                      delta=f"{(draw_prob-0.33)*100:.1f}% vs base")
            m3.metric(f"⚽ {away_team} Win", f"{away_prob*100:.1f}%",
                      delta=f"{'+'  if away_prob > 0.33 else ''}{(away_prob-0.33)*100:.1f}% vs base")

        with tab_radar:
            st.plotly_chart(
                make_team_radar(h_feats, a_feats, home_team, away_team),
                use_container_width=True, config={"displayModeBar": False},
            )
            # Elo gauges
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(make_elo_gauge(h_feats["elo"], home_team, COLORS["home"]),
                                use_container_width=True, config={"displayModeBar": False})
            with g2:
                st.plotly_chart(make_elo_gauge(a_feats["elo"], away_team, COLORS["away"]),
                                use_container_width=True, config={"displayModeBar": False})

        with tab_insights:
            _render_match_insights(home_team, away_team, h_feats, a_feats,
                                    neutral, tournament, home_prob, away_prob, draw_prob)

    else:
        # Pre-prediction state: show team quick-stats if both selected
        if home_team and away_team and home_team != away_team:
            _render_team_preview(home_team, away_team, team_stats)


def _render_team_preview(home_team: str, away_team: str, team_stats: dict):
    """Show a lightweight preview card for both teams before prediction."""
    h = get_team_features(home_team, team_stats)
    a = get_team_features(away_team, team_stats)

    st.markdown('<div class="cv-subheading" style="margin-top:0.5rem;">'
                'Team Overview  —  click Generate Prediction to run the model</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, team, feats, color in [
        (c1, home_team, h, COLORS["home"]),
        (c2, away_team, a, COLORS["away"]),
    ]:
        with col:
            st.markdown(f"""
            <div class="cv-card" style="border-left: 3px solid {color};">
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.3rem;
                     font-weight:700; color:{color}; letter-spacing:1px;">{team}</div>
                <div style="font-size:0.8rem; color:#5a7a9a; margin-bottom:1rem;">
                     {feats['matches_played']} international matches on record</div>
                <table style="width:100%; font-size:0.9rem; border-collapse:collapse;">
                    <tr><td style="color:#5a7a9a; padding:3px 0;">🏆 Elo Rating</td>
                        <td style="text-align:right; color:#e8f1ff; font-family:'Barlow Condensed',sans-serif;
                            font-weight:600;">{feats['elo']:.0f}</td></tr>
                    <tr><td style="color:#5a7a9a; padding:3px 0;">📈 Recent Form (avg pts)</td>
                        <td style="text-align:right; color:#e8f1ff;">{feats['recent_points']:.2f} / 3.00</td></tr>
                    <tr><td style="color:#5a7a9a; padding:3px 0;">⚽ Avg Goals Scored</td>
                        <td style="text-align:right; color:#00ff9d;">{feats['avg_scored']:.2f}</td></tr>
                    <tr><td style="color:#5a7a9a; padding:3px 0;">🛡️ Avg Goals Conceded</td>
                        <td style="text-align:right; color:#ff6b35;">{feats['avg_conceded']:.2f}</td></tr>
                    <tr><td style="color:#5a7a9a; padding:3px 0;">📊 Rolling Goal Diff</td>
                        <td style="text-align:right; color:#00d4ff;">{feats['rolling_gd']:+.2f}</td></tr>
                    <tr><td style="color:#5a7a9a; padding:3px 0;">⚡ Form Strip (last 5)</td>
                        <td style="text-align:right;">{form_bar(feats['recent_points'])}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)


def _render_match_insights(home_team, away_team, h, a,
                             neutral, tournament,
                             home_prob, away_prob, draw_prob):
    """Qualitative narrative insights about the match-up."""
    elo_diff = h["elo"] - a["elo"]
    imp = TOURNAMENT_IMPORTANCE_MAP.get(tournament, 2)

    # Strength insight
    if abs(elo_diff) > 200:
        strength = (f"{'**' + home_team + '**'} holds a commanding Elo advantage "
                    f"of **{abs(elo_diff):.0f}** points — a significant gap "
                    f"that historically translates to a win probability above 65%.")
    elif abs(elo_diff) > 80:
        stronger = home_team if elo_diff > 0 else away_team
        strength = (f"**{stronger}** edges the Elo comparison by "
                    f"**{abs(elo_diff):.0f}** points — a moderate advantage "
                    f"that typically gives them a 10–15% edge.")
    else:
        strength = (f"The teams are **evenly matched** on Elo "
                    f"(difference: {abs(elo_diff):.0f} pts). This fixture is a genuine "
                    f"coin-toss — expect a competitive, tight contest.")

    # Form insight
    form_diff = h["recent_points"] - a["recent_points"]
    if abs(form_diff) > 0.8:
        hotter = home_team if form_diff > 0 else away_team
        form_msg = f"**{hotter}** is in significantly better recent form."
    elif abs(form_diff) > 0.3:
        hotter = home_team if form_diff > 0 else away_team
        form_msg = f"**{hotter}** has a slight recent form edge."
    else:
        form_msg = "Both teams are in comparable recent form."

    # Venue insight
    venue_msg = ("On a **neutral ground**, home advantage is eliminated — "
                 "this levels the playing field slightly for the away side."
                 if neutral else
                 f"**{home_team}** benefits from home advantage, "
                 f"historically worth ~0.5 goals and a 5–8% boost in win probability.")

    # Tournament context
    imp_text = {1: "low-stakes friendly", 2: "regional tournament",
                3: "competitive qualifier", 4: "major continental cup",
                5: "elite world-stage competition"}
    tourn_msg = (f"This is a **{imp_text.get(imp, 'competitive')}** "
                 f"(importance tier {imp}/5). Higher-stakes matches "
                 f"tend to be more conservative — draw probability often "
                 f"rises slightly as teams protect leads.")

    # Uncertainty
    entropy = -sum(p * np.log(p + 1e-9) for p in [home_prob, away_prob, draw_prob])
    max_entropy = np.log(3)
    uncertainty_pct = (entropy / max_entropy) * 100

    st.markdown(f"""
    <div class="cv-card cv-card-accent" style="margin-bottom:0.8rem;">
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
             color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
             margin-bottom:0.6rem;">⚡ STRENGTH ANALYSIS</div>
        <p style="color:#e8f1ff; font-size:0.92rem; line-height:1.6;">{strength}</p>
    </div>
    <div class="cv-card" style="margin-bottom:0.8rem;">
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
             color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
             margin-bottom:0.6rem;">📈 RECENT FORM</div>
        <p style="color:#e8f1ff; font-size:0.92rem; line-height:1.6;">{form_msg}</p>
    </div>
    <div class="cv-card" style="margin-bottom:0.8rem;">
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
             color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
             margin-bottom:0.6rem;">🏟️ VENUE CONTEXT</div>
        <p style="color:#e8f1ff; font-size:0.92rem; line-height:1.6;">{venue_msg}</p>
    </div>
    <div class="cv-card" style="margin-bottom:0.8rem;">
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
             color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
             margin-bottom:0.6rem;">🏆 TOURNAMENT CONTEXT</div>
        <p style="color:#e8f1ff; font-size:0.92rem; line-height:1.6;">{tourn_msg}</p>
    </div>
    <div class="cv-card" style="border-left: 3px solid #ffd700;">
        <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
             color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
             margin-bottom:0.6rem;">🎲 MODEL UNCERTAINTY</div>
        <p style="color:#e8f1ff; font-size:0.92rem; line-height:1.6;">
            Prediction entropy is <b style="color:#ffd700;">{uncertainty_pct:.0f}%</b> of maximum uncertainty.
            {('The model is confident in this prediction.' if uncertainty_pct < 50
              else 'The model sees this as a genuinely open contest — treat probabilities as a guide, not a guarantee.')}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — TEAM EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_team_explorer(team_stats: dict):
    st.markdown("""
    <div class="cv-heading">📊 Team Explorer</div>
    <div class="cv-subheading">Browse Elo ratings, form, and stats for all 298 international teams</div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e3050;">', unsafe_allow_html=True)

    # Build dataframe
    rows = []
    for team, s in team_stats.items():
        rows.append({
            "Team": team,
            "Elo Rating": s["elo"],
            "Recent Form (avg pts)": s["recent_points"],
            "Goals Scored": s["avg_scored"],
            "Goals Conceded": s["avg_conceded"],
            "Goal Difference": s["rolling_gd"],
            "Matches Played": s.get("matches_played", 0),
        })
    df = pd.DataFrame(rows).sort_values("Elo Rating", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed ranking

    # Top 15 Elo chart
    top15 = df.head(15)
    fig_top = go.Figure(go.Bar(
        x=top15["Team"],
        y=top15["Elo Rating"],
        marker=dict(
            color=top15["Elo Rating"],
            colorscale=[[0, "#1e3050"], [0.5, "#00d4ff"], [1.0, "#00ff9d"]],
            showscale=False,
        ),
        text=top15["Elo Rating"].round(0).astype(int),
        textposition="outside",
        textfont=dict(family="Barlow Condensed", size=12, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>Elo: %{y:.0f}<extra></extra>",
    ))
    fig_top.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=380,
        xaxis=dict(tickfont=dict(family="Barlow Condensed", size=12, color=COLORS["text"]),
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(range=[df["Elo Rating"].min() * 0.95, df["Elo Rating"].max() * 1.05],
                   gridcolor=COLORS["border"],
                   tickfont=dict(color=COLORS["muted"])),
        margin=dict(t=20, b=20, l=20, r=20),
        title=dict(text="Top 15 Teams by Elo Rating", font=dict(
            family="Barlow Condensed", size=16, color=COLORS["text"])),
    )
    st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})

    # Scatter: Elo vs Goals Scored
    with st.expander("🔍  Elo vs. Attack Strength (scatter)", expanded=False):
        fig_scatter = px.scatter(
            df, x="Elo Rating", y="Goals Scored",
            hover_name="Team",
            size="Matches Played", size_max=18,
            color="Goal Difference",
            color_continuous_scale=[[0,"#ff4560"],[0.5,"#5a7a9a"],[1,"#00ff9d"]],
            template=PLOTLY_TEMPLATE,
            labels={"Goals Scored": "Avg Goals Scored (last 5)", "Elo Rating": "Elo Rating"},
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=420, margin=dict(t=20, b=20),
            coloraxis_colorbar=dict(tickfont=dict(color=COLORS["muted"])),
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    # Filterable table
    st.markdown("<br>", unsafe_allow_html=True)
    search = st.text_input("🔎 Search team", placeholder="e.g. Brazil, Germany, Japan…")
    filtered = df[df["Team"].str.contains(search, case=False)] if search else df

    st.dataframe(
        filtered.style.background_gradient(
            subset=["Elo Rating"], cmap="Blues"
        ).format({
            "Elo Rating": "{:.0f}",
            "Recent Form (avg pts)": "{:.2f}",
            "Goals Scored": "{:.2f}",
            "Goals Conceded": "{:.2f}",
            "Goal Difference": "{:+.2f}",
        }),
        use_container_width=True,
        height=420,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def page_model_insights(models: dict, model_choice: str):
    st.markdown("""
    <div class="cv-heading">🧠 Model Insights</div>
    <div class="cv-subheading">Feature importance, model architecture, and performance metrics</div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e3050;">', unsafe_allow_html=True)

    model = models[model_choice]

    # Performance cards
    st.markdown('<div class="cv-subheading">Model Performance on 2017–2020 Test Set</div>',
                unsafe_allow_html=True)
    perf_data = {
        "Random Forest":        {"accuracy": 54.5, "f1": 0.510, "params": "300 trees, depth 8"},
        "Logistic Regression":  {"accuracy": 57.7, "f1": 0.433, "params": "C=0.1, L2, lbfgs"},
    }
    m1, m2, m3, m4 = st.columns(4)
    d = perf_data.get(model_choice, {"accuracy": 0, "f1": 0, "params": "N/A"})
    m1.metric("Accuracy",  f"{d['accuracy']:.1f}%")
    m2.metric("Macro F1",  f"{d['f1']:.3f}")
    m3.metric("Train Size", "16,113 matches")
    m4.metric("Test Size",  "3,025 matches")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance (RF only)
    if hasattr(model, "feature_importances_"):
        st.markdown('<div class="cv-subheading">Feature Importance — Random Forest (Gini)</div>',
                    unsafe_allow_html=True)
        fig = make_feature_importance_chart(model)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Feature legend
        with st.expander("📖  Why These Features?"):
            st.markdown("""
| Feature | Why It Matters |
|---|---|
| **elo_diff / home_elo / away_elo** | Elo is the gold standard for continuous team strength. It adjusts for opponent quality and recency. The single most predictive signal. |
| **home/away_recent_points** | Recent form (last 5 matches, pts basis) captures momentum and current squad fitness. |
| **home/away_avg_goals_scored** | Offensive firepower. Teams that score more tend to win more — obvious but quantifiably significant. |
| **home/away_avg_goals_conceded** | Defensive solidity. Conceding fewer goals is as important as scoring them. |
| **home/away_rolling_gd** | Net goal difference over 5 games — combined attacking + defensive signal. |
| **neutral_venue** | Eliminates home advantage, which is worth roughly 0.3–0.5 Elo points of expected improvement. |
| **tournament_importance** | World Cup games are played differently from friendlies. Stakes affect tactics and effort. |
            """)
    else:
        # Logistic Regression — show coefficients
        st.markdown('<div class="cv-subheading">Logistic Regression Coefficients</div>',
                    unsafe_allow_html=True)
        coef_df = pd.DataFrame(
            model.coef_,
            columns=FEATURE_COLS,
            index=["Home Win", "Away Win", "Draw"],
        ).T
        fig_coef = go.Figure()
        for cls, color in [("Home Win", COLORS["home"]),
                            ("Away Win", COLORS["away"]),
                            ("Draw",     COLORS["draw"])]:
            fig_coef.add_trace(go.Bar(
                name=cls, x=coef_df.index, y=coef_df[cls], marker_color=color,
                hovertemplate=f"<b>%{{x}}</b><br>{cls}: %{{y:.3f}}<extra></extra>",
            ))
        fig_coef.update_layout(
            template=PLOTLY_TEMPLATE, barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            xaxis=dict(tickangle=-35, tickfont=dict(size=10, color=COLORS["text"])),
            yaxis=dict(gridcolor=COLORS["border"]),
            legend=dict(font=dict(color=COLORS["text"])),
            margin=dict(t=20, b=80),
        )
        st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})

    # Architecture explainer
    with st.expander("🏗️  Model Architecture & Training Details"):
        st.markdown(f"""
**Model selected:** `{model_choice}`

**Training period:** 2000–2016 (16,113 matches)  
**Test period:** 2017–2020 (3,025 matches)  
**Split strategy:** Chronological — never random (prevents data leakage)

**Why time-based split?**  
A random split would let the model see 2019 matches during training and then predict
2010 matches — a form of temporal leakage. In sports analytics, we always train on the
past and test on the future, exactly mirroring deployment conditions.

**No-leakage guarantee:**  
Every feature is computed using only matches *before* the current match date.
The Elo engine processes matches sequentially, updating ratings only after
feature extraction. This is the most common mistake in sports ML pipelines.

**Class balance:**  
Home Win ~48% | Away Win ~28% | Draw ~24%.  
Random Forest uses `class_weight='balanced'` to compensate for the draw minority.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown("""
    <div class="cv-heading">ℹ️ About CopaVision AI</div>
    <div class="cv-subheading">Phase 1 — International Football Match Predictor</div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e3050;">', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("""
        <div class="cv-card cv-card-accent">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
                 color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
                 margin-bottom:0.8rem;">PROJECT OVERVIEW</div>
            <p style="color:#e8f1ff; line-height:1.7;">
                CopaVision AI is a football analytics platform that uses machine learning
                to predict international match outcomes. Phase 1 covers the core prediction
                engine trained on 41,500+ historical matches from 1872–2020.
            </p>
            <p style="color:#e8f1ff; line-height:1.7;">
                The model combines <b style="color:#00d4ff;">Elo ratings</b>,
                <b style="color:#00ff9d;">rolling form metrics</b>, and
                <b style="color:#d2a8ff;">tournament context</b>
                to generate probabilistic win/draw/loss predictions for any pair of
                international teams.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cv-card">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
                 color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
                 margin-bottom:0.8rem;">HOW TO RUN</div>
            <pre style="background:#050b14; color:#00ff9d; padding:1rem;
                 border-radius:8px; font-size:0.85rem; overflow-x:auto;">
# Install dependencies
pip install streamlit pandas numpy \\
    scikit-learn plotly joblib

# Launch app
streamlit run app.py</pre>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cv-card">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
                 color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
                 margin-bottom:0.8rem;">DEPLOY ON STREAMLIT CLOUD</div>
            <ol style="color:#e8f1ff; line-height:2; font-size:0.9rem;">
                <li>Push your project to a <b>public GitHub repo</b></li>
                <li>Go to <a href="https://share.streamlit.io" style="color:#00d4ff;">share.streamlit.io</a> and sign in</li>
                <li>Click <b>New App</b> → select your repo and branch</li>
                <li>Set <b>Main file path</b> to <code>app.py</code></li>
                <li>Add a <code>requirements.txt</code> (see right column)</li>
                <li>Click <b>Deploy</b> — live in ~2 minutes 🚀</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="cv-card" style="margin-bottom:0.8rem;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
                 color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
                 margin-bottom:0.8rem;">FILE STRUCTURE</div>
            <pre style="background:#050b14; color:#e8f1ff; padding:1rem;
                 border-radius:8px; font-size:0.82rem;">
copavision-ai/
│
├── app.py               ← this file
├── requirements.txt
├── team_stats.json      ← Elo + form data
│
├── models/
│   ├── copavision_rf.pkl
│   └── copavision_lr.pkl
│
├── data/
│   └── results.csv      ← raw dataset
│
└── utils/               ← (Phase 2+)
    ├── features.py
    ├── elo.py
    └── viz.py</pre>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cv-card" style="margin-bottom:0.8rem;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
                 color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
                 margin-bottom:0.8rem;">REQUIREMENTS.TXT</div>
            <pre style="background:#050b14; color:#00ff9d; padding:1rem;
                 border-radius:8px; font-size:0.85rem;">
streamlit>=1.32
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
plotly>=5.20
joblib>=1.3</pre>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="cv-card cv-card-accent">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:0.75rem;
                 color:#5a7a9a; letter-spacing:2px; text-transform:uppercase;
                 margin-bottom:0.8rem;">PHASE ROADMAP</div>
            <div style="font-size:0.88rem; line-height:2.2; color:#e8f1ff;">
                <span style="color:#00ff9d;">✓ Phase 1</span> &nbsp;Match Outcome Predictor<br>
                <span style="color:#00d4ff;">→ Phase 2</span> &nbsp;Player Dashboard<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add a <code>pages/02_players.py</code> file with<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FIFA/Transfermarkt player stats<br><br>
                <span style="color:#5a7a9a;">○ Phase 3</span> &nbsp;Sentiment Tracker<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Twitter/Reddit API + VADER/BERT<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sentiment pre-match signals<br><br>
                <span style="color:#5a7a9a;">○ Phase 4</span> &nbsp;Live Match Feed<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Rapid API football → real-time<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score updates & live Elo shifts
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Accuracy expectations
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📉  Realistic Accuracy Expectations & Limitations"):
        st.markdown("""
| Benchmark | Accuracy |
|---|---|
| Random guessing (3 classes) | 33.3% |
| Always predict Home Win | ~48% |
| **CopaVision AI Phase 1** | **54–58%** |
| State-of-the-art (squad data + deep learning) | 60–65% |
| Human football experts | ~60% |

**Why football is hard to predict:**
- Individual brilliance and errors are genuinely stochastic
- Injuries and suspensions aren't in historical data
- Tactics adapt match-to-match (chess, not poker)
- Home crowd effects vary enormously by stadium
- VAR and referee decisions introduce randomness

**Phase 1 limitations:**
- No player-level data (squad strength, injury list)
- No head-to-head historical records
- Friendly matches dilute the training signal
- Model trained up to 2020 — club-level Elo not factored in
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load data
    with st.spinner("Loading CopaVision AI…"):
        models = load_models()
        team_stats = load_team_stats()

    if not models or not team_stats:
        st.error("Critical files missing. Ensure models/ and team_stats.json exist.")
        st.stop()

    # Sidebar (returns page & model choice)
    page, model_choice = render_sidebar(team_stats, models)

    # Page routing
    page_key = page.split("  ")[-1].strip()
    if page_key == "Match Predictor":
        page_match_predictor(models, team_stats, model_choice)
    elif page_key == "Team Explorer":
        page_team_explorer(team_stats)
    elif page_key == "Model Insights":
        page_model_insights(models, model_choice)
    elif page_key == "About":
        page_about()


if __name__ == "__main__":
    main()
