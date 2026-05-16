"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     CopaVision AI  |  Phase 3: Football Sentiment Intelligence              ║
║     pages/03_Sentiment_Tracker.py                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Data sources (both free, no approval needed):
  - The Guardian API  →  open-platform.theguardian.com/access
  - NewsAPI           →  newsapi.org/register

Set API keys in Hugging Face Secrets (Settings → Variables and secrets):
  GUARDIAN_API_KEY
  NEWSAPI_KEY

If neither key is set, demo mode activates automatically.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Add scripts/ to path so modules are importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from news_collector    import NewsCollector
from text_cleaning     import clean_text, is_football_relevant
from sentiment_pipeline import score_articles
from momentum_engine   import (
    build_sentiment_timeline, detect_spikes,
    aggregate_team_sentiment, aggregate_player_sentiment,
    extract_trending_topics, build_emotion_timeline,
    compute_live_stats,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Intelligence | CopaVision AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — consistent with Phase 1 & 2 dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:#050b14; --bg-card:#0e1621; --bg-panel:#121d2e;
    --border:#1e3050;  --accent:#00d4ff;  --green:#00ff9d;
    --orange:#ff6b35;  --gold:#ffd700;    --text:#e8f1ff;
    --muted:#5a7a9a;   --purple:#d2a8ff;
}
.stApp {
    background-color:var(--bg-deep);
    background-image:
        radial-gradient(ellipse at 10% 0%,  rgba(0,212,255,0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 90% 100%, rgba(0,255,157,0.04) 0%, transparent 50%);
    font-family:'DM Sans',sans-serif; color:var(--text);
}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.5rem;padding-bottom:2rem;}

[data-testid="stSidebar"]{background:var(--bg-panel) !important;border-right:1px solid var(--border) !important;}
[data-testid="stSidebar"] *{color:var(--text) !important;}

[data-testid="stMetric"]{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:16px 20px !important;}
[data-testid="stMetricValue"]{font-family:'Barlow Condensed',sans-serif !important;font-size:1.9rem !important;font-weight:700 !important;color:var(--accent) !important;}
[data-testid="stMetricLabel"]{color:var(--muted) !important;font-size:0.75rem !important;}

.stButton>button{background:linear-gradient(135deg,#00d4ff,#0099cc);color:#050b14 !important;border:none !important;border-radius:8px !important;font-family:'Barlow Condensed',sans-serif !important;font-size:1rem !important;font-weight:700 !important;padding:0.5rem 1.6rem !important;transition:all 0.2s !important;text-transform:uppercase !important;}
.stButton>button:hover{transform:translateY(-2px) !important;box-shadow:0 6px 24px rgba(0,212,255,0.3) !important;}

[data-testid="stTab"]{font-family:'Barlow Condensed',sans-serif !important;font-size:1rem !important;color:var(--muted) !important;}
[data-testid="stTab"][aria-selected="true"]{color:var(--accent) !important;}
[data-testid="stExpander"]{background:var(--bg-card) !important;border:1px solid var(--border) !important;border-radius:10px !important;}

hr{border-color:var(--border) !important;margin:1.2rem 0 !important;}
.stSpinner>div{border-top-color:var(--accent) !important;}
[data-testid="stSelectbox"]>div>div{background:var(--bg-card) !important;border:1px solid var(--border) !important;border-radius:8px !important;}

.cv-heading{display:block;width:100%;font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;letter-spacing:1px;color:var(--text);text-transform:uppercase;margin:0 0 0.35rem 0;}
.cv-subheading{display:block;width:100%;font-family:'Barlow Condensed',sans-serif;font-size:0.95rem;color:var(--muted);letter-spacing:0.4px;margin:0 0 1rem 0;}
.cv-label{font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;color:var(--muted);letter-spacing:2.5px;text-transform:uppercase;margin-bottom:0.4rem;display:block;}
.cv-card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:0.9rem;}
.cv-card-blue{border-left:3px solid var(--accent);}
.cv-card-green{border-left:3px solid var(--green);}
.cv-card-orange{border-left:3px solid var(--orange);}
.cv-card-gold{border-left:3px solid var(--gold);}

/* Sentiment gauge colours */
.sent-pos{color:#00ff9d;font-weight:700;}
.sent-neg{color:#ff6b35;font-weight:700;}
.sent-neu{color:#00d4ff;font-weight:700;}

/* Live indicator pulse */
@keyframes pulse{0%{opacity:1;}50%{opacity:0.4;}100%{opacity:1;}}
.live-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#00ff9d;animation:pulse 1.5s infinite;margin-right:6px;vertical-align:middle;}
.live-badge{font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;color:#00ff9d;letter-spacing:1px;}

/* News card */
.news-card{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:0.5rem;}
.news-title{font-size:0.88rem;color:var(--text);font-weight:500;line-height:1.4;}
.news-meta{font-size:0.75rem;color:var(--muted);margin-top:4px;}

/* Emotion chip */
.emotion-chip{display:inline-block;font-family:'Barlow Condensed',sans-serif;font-size:0.8rem;font-weight:600;padding:3px 10px;border-radius:20px;margin:2px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "accent":"#00d4ff","green":"#00ff9d","orange":"#ff6b35",
    "gold":"#ffd700","purple":"#d2a8ff","muted":"#5a7a9a",
    "bg":"#0e1621","border":"#1e3050","text":"#e8f1ff",
}

EMOTION_COLORS = {
    "excitement":    "#ffd700",
    "celebration":   "#00ff9d",
    "frustration":   "#ff6b35",
    "anger":         "#ff4560",
    "disappointment":"#5a7a9a",
    "shock":         "#d2a8ff",
}

EMOTION_ICONS = {
    "excitement":"⚡","celebration":"🏆","frustration":"😤",
    "anger":"😡","disappointment":"😞","shock":"😲",
}

_PLOT = dict(template="plotly_dark",
             paper_bgcolor="rgba(0,0,0,0)",
             plot_bgcolor="rgba(0,0,0,0)")

REFRESH_SECONDS = 300   # 5-minute polling interval

SEARCH_PRESETS = {
    "Premier League":      ["Premier League", "Arsenal Liverpool Manchester City Chelsea"],
    "Champions League":    ["Champions League", "UCL semifinal final"],
    "La Liga":             ["La Liga", "Real Madrid Barcelona"],
    "World Cup":           ["World Cup football", "FIFA World Cup"],
    "Transfer News":       ["football transfer", "signing contract"],
    "Top Players":         ["Messi Ronaldo Mbappe Haaland Bellingham"],
    "Custom":              [],
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  — cached with TTL so it auto-refreshes
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_and_score(guardian_key: str,
                    newsapi_key: str,
                    queries: list,
                    days_back: int,
                    demo_mode: bool) -> pd.DataFrame:
    """
    Core pipeline: collect → clean → score → return DataFrame.
    Cached for REFRESH_SECONDS so Hugging Face doesn't hammer the APIs.
    """
    collector = NewsCollector(
        guardian_key=guardian_key or None,
        newsapi_key=newsapi_key or None,
    )

    if demo_mode or (not guardian_key and not newsapi_key):
        raw = collector.collect_demo(n=120)
    else:
        raw = collector.collect_all(
            queries=queries or None,
            max_per_source=25,
            days_back=days_back,
        )

    if not raw:
        return pd.DataFrame()

    # Clean text
    for art in raw:
        art["title"] = clean_text(art.get("title",""))
        art["body"]  = clean_text(art.get("body",""))

    # Filter for football relevance
    relevant = [a for a in raw
                if is_football_relevant(a["title"] + " " + a["body"])]

    if not relevant:
        relevant = raw   # fallback: keep all

    return score_articles(relevant)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _gauge(value: float, title: str) -> go.Figure:
    """Sentiment gauge — value from -1 to +1."""
    pct  = int((value + 1) / 2 * 100)
    color = C["green"] if value > 0.05 else C["orange"] if value < -0.05 else C["accent"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(family="Barlow Condensed", size=38,
                                           color=C["text"])),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor=C["muted"],
                      tickfont=dict(color=C["muted"], size=10),
                      tickvals=[0,25,50,75,100],
                      ticktext=["Very Neg","Neg","Neutral","Pos","Very Pos"]),
            bar=dict(color=color, thickness=0.28),
            bgcolor=C["bg"], borderwidth=0,
            steps=[
                dict(range=[0, 33],  color="rgba(255,107,53,0.12)"),
                dict(range=[33, 66], color="rgba(90,122,154,0.08)"),
                dict(range=[66, 100],color="rgba(0,255,157,0.12)"),
            ],
            threshold=dict(line=dict(color=color, width=3),
                           thickness=0.75, value=pct),
        ),
        title=dict(text=title, font=dict(family="Barlow Condensed",
                                          size=14, color=C["muted"])),
    ))
    fig.update_layout(**_PLOT, height=240, margin=dict(t=40,b=10,l=30,r=30))
    return fig


def _timeline_chart(tl: pd.DataFrame) -> go.Figure:
    """Sentiment timeline with rolling average and spike markers."""
    if tl.empty:
        return go.Figure()

    fig = go.Figure()

    # Bar chart for per-period average
    bar_colors = [C["green"] if v > 0.05 else C["orange"] if v < -0.05 else C["accent"]
                  for v in tl["avg_compound"]]
    fig.add_trace(go.Bar(
        x=tl["timestamp"], y=tl["avg_compound"],
        name="Period Sentiment", marker_color=bar_colors,
        opacity=0.55, hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>",
    ))

    # Rolling average line
    fig.add_trace(go.Scatter(
        x=tl["timestamp"], y=tl["rolling_avg"],
        name="Rolling Average", mode="lines",
        line=dict(color=C["accent"], width=2.5, dash="solid"),
        hovertemplate="<b>%{x}</b><br>Rolling Avg: %{y:.3f}<extra></extra>",
    ))

    # Zero line
    fig.add_hline(y=0, line=dict(color=C["muted"], dash="dash", width=1))

    fig.update_layout(
        **_PLOT, height=340,
        xaxis=dict(tickfont=dict(color=C["muted"]), gridcolor=C["border"]),
        yaxis=dict(tickfont=dict(color=C["muted"]), gridcolor=C["border"],
                   title=dict(text="Sentiment Score",
                              font=dict(color=C["muted"]))),
        legend=dict(font=dict(color=C["text"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20,b=40,l=50,r=20),
        title=dict(text="Sentiment Timeline", font=dict(
            family="Barlow Condensed", size=14, color=C["text"])),
    )
    return fig


def _emotion_area_chart(etl: pd.DataFrame) -> go.Figure:
    """Stacked area chart of emotions over time."""
    if etl.empty:
        return go.Figure()

    fig = go.Figure()
    emotion_cols = [c for c in EMOTION_COLORS if c in etl.columns]

    for emotion in emotion_cols:
        fig.add_trace(go.Scatter(
            x=etl["timestamp"], y=etl[emotion],
            name=f"{EMOTION_ICONS.get(emotion,'')} {emotion.title()}",
            fill="tozeroy", mode="lines",
            line=dict(color=EMOTION_COLORS[emotion], width=1.5),
            fillcolor=EMOTION_COLORS[emotion] + "22",
            hovertemplate=f"<b>{emotion.title()}</b><br>%{{x}}<br>Score: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        **_PLOT, height=320,
        xaxis=dict(tickfont=dict(color=C["muted"]), gridcolor=C["border"]),
        yaxis=dict(tickfont=dict(color=C["muted"]), gridcolor=C["border"],
                   title=dict(text="Emotion Intensity",
                              font=dict(color=C["muted"]))),
        legend=dict(font=dict(color=C["text"], size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20,b=40,l=50,r=20),
        title=dict(text="Emotion Timeline", font=dict(
            family="Barlow Condensed", size=14, color=C["text"])),
    )
    return fig


def _team_bar_chart(team_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar — team sentiment comparison."""
    if team_df.empty:
        return go.Figure()

    top = team_df.head(12)
    colors = [C["green"] if v > 0.05 else C["orange"] if v < -0.05 else C["accent"]
              for v in top["avg_compound"]]

    fig = go.Figure(go.Bar(
        x=top["avg_compound"], y=top["team"],
        orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in top["avg_compound"]],
        textposition="outside",
        textfont=dict(family="Barlow Condensed", size=12, color=C["text"]),
        hovertemplate="<b>%{y}</b><br>Sentiment: %{x:.3f}<extra></extra>",
        customdata=top["article_count"],
        hovertext=[f"{c} articles" for c in top["article_count"]],
    ))
    fig.add_vline(x=0, line=dict(color=C["muted"], dash="dash", width=1))
    fig.update_layout(
        **_PLOT, height=380,
        xaxis=dict(tickfont=dict(color=C["muted"]), gridcolor=C["border"],
                   title=dict(text="Avg Sentiment Score",
                              font=dict(color=C["muted"]))),
        yaxis=dict(tickfont=dict(family="DM Sans", size=11, color=C["text"])),
        margin=dict(t=20,b=30,l=140,r=70),
        title=dict(text="Team Sentiment Comparison",
                   font=dict(family="Barlow Condensed", size=14, color=C["text"])),
    )
    return fig


def _emotion_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart of dominant emotion distribution."""
    if df.empty or "dominant_emotion" not in df.columns:
        return go.Figure()

    counts = df["dominant_emotion"].value_counts()
    colors = [EMOTION_COLORS.get(e, C["muted"]) for e in counts.index]

    fig = go.Figure(go.Pie(
        labels=[f"{EMOTION_ICONS.get(e,'')} {e.title()}" for e in counts.index],
        values=counts.values, hole=0.58,
        marker_colors=colors,
        textinfo="label+percent",
        textfont=dict(family="Barlow Condensed", size=13, color=C["text"]),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOT, height=300,
        showlegend=False,
        margin=dict(t=20,b=20,l=20,r=20),
        title=dict(text="Emotion Distribution",
                   font=dict(family="Barlow Condensed", size=14, color=C["text"])),
    )
    return fig


def _trending_bar(trend_df: pd.DataFrame) -> go.Figure:
    """Bar chart of trending football topics."""
    if trend_df.empty:
        return go.Figure()

    top = trend_df.head(12)
    colors = [C["green"] if v > 0.05 else C["orange"] if v < -0.05 else C["accent"]
              for v in top["avg_sentiment"]]

    fig = go.Figure(go.Bar(
        x=top["trend_score"], y=top["topic"],
        orientation="h", marker_color=colors,
        text=[f"  {m} mentions" for m in top["mentions"]],
        textposition="outside",
        textfont=dict(size=11, color=C["text"]),
        hovertemplate="<b>%{y}</b><br>Trend Score: %{x:.1f}<br>Sentiment: %{customdata:.3f}<extra></extra>",
        customdata=top["avg_sentiment"],
    ))
    fig.update_layout(
        **_PLOT, height=380,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(family="DM Sans", size=11, color=C["text"])),
        margin=dict(t=20,b=20,l=110,r=80),
        title=dict(text="Trending Football Topics",
                   font=dict(family="Barlow Condensed", size=14, color=C["text"])),
    )
    return fig


def _sentiment_pie(stats: dict) -> go.Figure:
    """Simple 3-slice pie: pos / neg / neu."""
    fig = go.Figure(go.Pie(
        labels=["Positive","Neutral","Negative"],
        values=[stats["pos_pct"], stats["neu_pct"], stats["neg_pct"]],
        hole=0.5,
        marker_colors=[C["green"], C["accent"], C["orange"]],
        textinfo="label+percent",
        textfont=dict(family="Barlow Condensed", size=13, color=C["text"]),
        hovertemplate="<b>%{label}</b>: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOT, height=260, showlegend=False,
        margin=dict(t=20,b=10,l=10,r=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.8rem 0 1.2rem;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.9rem;
                 font-weight:900;color:#00d4ff;letter-spacing:3px;">
                 COPA<span style="color:#00ff9d;">VISION</span></div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.75rem;
                 color:#5a7a9a;letter-spacing:4px;">A I  ·  P H A S E  3</div>
        </div><hr>
        """, unsafe_allow_html=True)

        st.markdown('<span class="cv-label">NAVIGATION</span>', unsafe_allow_html=True)
        page = st.radio("nav", [
            "📡  Live Dashboard",
            "📈  Sentiment Timeline",
            "🎭  Emotion Analysis",
            "⚽  Team Comparison",
            "🔥  Trending Topics",
            "📰  Latest Headlines",
        ], label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Search configuration
        st.markdown('<span class="cv-label">DATA COLLECTION</span>',
                    unsafe_allow_html=True)
        preset = st.selectbox("Topic Preset", list(SEARCH_PRESETS.keys()),
                              index=0, help="Choose what football content to analyse")

        custom_query = ""
        if preset == "Custom":
            custom_query = st.text_input("Custom search terms",
                                          placeholder="e.g. Mbappe Champions League")

        days_back = st.slider("Days of news to collect",
                               min_value=1, max_value=7, value=3,
                               help="Larger range = more articles but slower load")

        st.markdown("<hr>", unsafe_allow_html=True)

        g_key = os.environ.get("GUARDIAN_API_KEY", "")
        n_key = os.environ.get("NEWSAPI_KEY", "")
        demo_mode = not (g_key or n_key)
        if demo_mode:
            st.info("Demo mode active: no API keys found.")
        else:
            st.success("Live data enabled.")

        refresh = st.button("🔄 Refresh Data Now", width="stretch")

        st.markdown("<hr>", unsafe_allow_html=True)

    # Build queries from preset
    queries = SEARCH_PRESETS.get(preset, [])
    if preset == "Custom" and custom_query:
        queries = [custom_query]

    return page.strip().split("  ")[-1].strip(), queries, days_back, demo_mode, refresh, g_key, n_key


# ─────────────────────────────────────────────────────────────────────────────
# HELPER WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
def _section(title, subtitle=""):
    st.markdown(f'<h2 class="cv-heading">{title}</h2>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="cv-subheading">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1e3050;">', unsafe_allow_html=True)


def _live_badge(last_updated: str, total: int, demo: bool):
    mode = "🎮 Demo Data" if demo else "📡 Live Data"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:1rem;">
        <span><span class="live-dot"></span>
        <span class="live-badge">{mode}</span></span>
        <span style="font-size:0.8rem;color:{C['muted']};">
            {total} articles analysed · Last updated {last_updated}
            · Auto-refreshes every {REFRESH_SECONDS//60} min
        </span>
    </div>
    """, unsafe_allow_html=True)


def _sentiment_chip(label: str):
    color = (C["green"] if label == "Positive"
             else C["orange"] if label == "Negative"
             else C["accent"])
    icon  = "🟢" if label == "Positive" else "🔴" if label == "Negative" else "🔵"
    return f'<span style="color:{color};font-weight:600;">{icon} {label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — LIVE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_live_dashboard(df, stats, demo):
    _section("📡 Live Sentiment Dashboard",
             "Real-time football sentiment intelligence powered by news analysis")
    _live_badge(stats["last_updated"], stats["total_articles"], demo)

    # ── 5 KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    sentiment_val = stats["overall_sentiment"]
    k1.metric("🌡️ Overall Sentiment",
              f"{sentiment_val:+.3f}",
              delta="Positive" if sentiment_val > 0.05 else "Negative" if sentiment_val < -0.05 else "Neutral")
    k2.metric("🟢 Positive Articles", f"{stats['pos_pct']}%")
    k3.metric("🔴 Negative Articles", f"{stats['neg_pct']}%")
    k4.metric("📰 Articles Analysed", f"{stats['total_articles']:,}")
    k5.metric("⚡ Dominant Emotion", stats["dominant_emotion"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Pie ───────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns([2, 2, 3])
    with g1:
        st.plotly_chart(_gauge(stats["overall_sentiment"], "Overall Sentiment"),
                        width="stretch", config={"displayModeBar": False})
    with g2:
        st.plotly_chart(_sentiment_pie(stats),
                        width="stretch", config={"displayModeBar": False})
    with g3:
        st.markdown(f"""
        <div class="cv-card cv-card-blue" style="height:100%;">
            <span class="cv-label">🏟️ Match Intelligence Summary</span>
            <div style="font-size:0.9rem;line-height:2.2;margin-top:6px;">
                <div>📌 Most discussed team &nbsp;
                    <b style="color:{C['accent']};">{stats['most_mentioned_team']}</b></div>
                <div>⭐ Most discussed player &nbsp;
                    <b style="color:{C['gold']};">{stats['most_mentioned_player']}</b></div>
                <div>🎭 Dominant emotion &nbsp;
                    <b style="color:{C['green']};">{stats['dominant_emotion']}</b></div>
                <div>🟢 Positive coverage &nbsp;
                    <b style="color:{C['green']};">{stats['pos_pct']}%</b></div>
                <div>🔴 Negative coverage &nbsp;
                    <b style="color:{C['orange']};">{stats['neg_pct']}%</b></div>
                <div>⏱️ Last refreshed &nbsp;
                    <b style="color:{C['muted']};">{stats['last_updated']}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Spike alerts ──────────────────────────────────────────────────────────
    tl = build_sentiment_timeline(df, freq="1h")
    spikes = detect_spikes(tl, threshold=0.2)

    if not spikes.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cv-card cv-card-gold">
            <span class="cv-label">⚡ Sentiment Spikes Detected</span>
            <div style="font-size:0.85rem;color:{C['muted']};margin-bottom:8px;">
                Moments where fan sentiment shifted sharply — often correlating 
                with goals, red cards, or breaking transfer news.
            </div>
        """, unsafe_allow_html=True)

        for _, spike in spikes.head(3).iterrows():
            ts_str = str(spike["timestamp"])[:16]
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:5px 0;
                 border-bottom:1px solid {C['border']};">
                <span style="color:{C['text']};font-size:0.85rem;">
                    {spike['direction']} — {spike['spike_type']}</span>
                <span style="color:{C['muted']};font-size:0.82rem;">{ts_str}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — SENTIMENT TIMELINE
# ─────────────────────────────────────────────────────────────────────────────
def page_timeline(df, demo):
    _section("📈 Sentiment Timeline",
             "How football sentiment has shifted over time — spike detection and momentum curves")
    _live_badge(datetime.utcnow().strftime("%H:%M UTC"), len(df), demo)

    col_freq, col_window = st.columns([2, 3])
    with col_freq:
        st.markdown('<span class="cv-label">Time Bucket Size</span>',
                    unsafe_allow_html=True)
        freq = st.selectbox("freq", ["30min","1h","3h","6h","1d"],
                             index=1, label_visibility="collapsed",
                             format_func=lambda x: {"30min":"30 Minutes","1h":"1 Hour",
                                                     "3h":"3 Hours","6h":"6 Hours",
                                                     "1d":"1 Day"}[x])
    with col_window:
        st.markdown("""
        <div style="font-size:0.82rem;color:#5a7a9a;margin-top:28px;">
            💡 Shorter buckets show rapid reactions (goals, red cards).
            Longer buckets reveal broader sentiment trends.
        </div>
        """, unsafe_allow_html=True)

    tl = build_sentiment_timeline(df, freq=freq)
    if not tl.empty:
        st.plotly_chart(_timeline_chart(tl), width="stretch",
                        config={"displayModeBar": False})

    # Spike table
    spikes = detect_spikes(tl, threshold=0.15)
    if not spikes.empty:
        st.markdown('<div class="cv-subheading" style="margin-top:1rem;">⚡ Detected Sentiment Spikes</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem;color:#5a7a9a;margin-bottom:0.8rem;">
            A spike = sentiment changed by 0.15+ in one period.
            Typical causes: goal scored, red card, transfer announcement, 
            controversial VAR decision.
        </div>
        """, unsafe_allow_html=True)
        display_spikes = spikes.copy()
        display_spikes["timestamp"] = display_spikes["timestamp"].astype(str).str[:16]
        display_spikes.columns = ["Time", "Sentiment", "Direction",
                                   "Magnitude", "Likely Event"]
        st.dataframe(display_spikes, width="stretch", height=250)
    else:
        st.info("No major sentiment spikes detected in this time window. "
                "Try a shorter bucket size or a wider date range.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — EMOTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_emotions(df, demo):
    _section("🎭 Emotion Analysis",
             "What fans are feeling — excitement, frustration, anger, celebration and more")
    _live_badge(datetime.utcnow().strftime("%H:%M UTC"), len(df), demo)

    st.markdown("""
    <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
         border-radius:10px;padding:12px 16px;margin-bottom:1rem;font-size:0.85rem;
         color:#e8f1ff;line-height:1.7;">
        🎭 <b>How emotion scoring works:</b>
        Each article is scanned for football-specific emotion keywords.
        <b>Excitement</b> = words like "incredible", "hat-trick", "stunning".
        <b>Anger</b> = "red card", "VAR", "referee", "outrageous".
        <b>Celebration</b> = "champion", "title", "trophy", "glory".
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.plotly_chart(_emotion_donut(df), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        # Emotion score cards
        emotion_cols = [c for c in EMOTION_COLORS if c in df.columns]
        emotion_avgs = {e: df[e].mean() for e in emotion_cols}
        sorted_em = sorted(emotion_avgs.items(), key=lambda x: x[1], reverse=True)

        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;padding:7px 0;border-bottom:1px solid {C["border"]};">'
            f'<span style="color:{C["text"]};font-size:0.88rem;">'
            f'{EMOTION_ICONS.get(e,"")}&nbsp;{e.title()}</span>'
            f'<div style="flex:1;margin:0 12px;">'
            f'<div style="background:{C["border"]};border-radius:4px;height:7px;">'
            f'<div style="width:{int(v*100)}%;height:7px;border-radius:4px;'
            f'background:{EMOTION_COLORS[e]};"></div></div></div>'
            f'<span style="color:{EMOTION_COLORS[e]};font-weight:700;font-size:0.88rem;'
            f'min-width:40px;text-align:right;">{v:.2f}</span></div>'
            for e, v in sorted_em
        )
        st.markdown(f"""
        <div class="cv-card cv-card-blue">
            <span class="cv-label">Emotion Intensity Scores (0–1)</span>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    # Emotion timeline
    etl = build_emotion_timeline(df, freq="1h")
    if not etl.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(_emotion_area_chart(etl), width="stretch",
                        config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — TEAM COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def page_team_comparison(df, demo):
    _section("⚽ Team Sentiment Comparison",
             "Which clubs and nations are generating positive vs negative coverage right now")
    _live_badge(datetime.utcnow().strftime("%H:%M UTC"), len(df), demo)

    team_df = aggregate_team_sentiment(df)

    if team_df.empty:
        st.warning("Not enough team-tagged articles. Try a broader date range.")
        return

    st.plotly_chart(_team_bar_chart(team_df), width="stretch",
                    config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    most_pos = team_df[team_df["avg_compound"] > 0].nlargest(1, "avg_compound")
    most_neg = team_df[team_df["avg_compound"] < 0].nsmallest(1, "avg_compound")
    most_disc = team_df.nlargest(1, "article_count")

    for col, label, row, color in [
        (c1, "🟢 Most Positive Coverage",   most_pos,  C["green"]),
        (c2, "🔴 Most Negative Coverage",   most_neg,  C["orange"]),
        (c3, "📰 Most Discussed",           most_disc, C["accent"]),
    ]:
        with col:
            if not row.empty:
                r = row.iloc[0]
                st.markdown(f"""
                <div class="cv-card" style="border-left:3px solid {color};text-align:center;">
                    <div class="cv-label">{label}</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;
                         font-weight:800;color:{color};margin:6px 0;">{r['team']}</div>
                    <div style="font-size:0.85rem;color:{C['muted']};">
                        Sentiment: <b style="color:{color};">{r['avg_compound']:+.3f}</b>
                        &nbsp;|&nbsp; {r['article_count']} articles
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Player sentiment
    player_df = aggregate_player_sentiment(df)
    if not player_df.empty:
        st.markdown('<div class="cv-subheading" style="margin-top:1rem;">👤 Player Sentiment Rankings</div>',
                    unsafe_allow_html=True)
        display = player_df[["player","avg_compound","article_count",
                              "pos_pct","neg_pct","sentiment_label"]].copy()
        display.columns = ["Player","Avg Sentiment","Articles",
                           "Positive %","Negative %","Label"]
        display["Avg Sentiment"] = display["Avg Sentiment"].round(3)
        display["Positive %"] = (display["Positive %"] * 100).round(1)
        display["Negative %"] = (display["Negative %"] * 100).round(1)
        st.dataframe(display, width="stretch", height=320)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — TRENDING TOPICS
# ─────────────────────────────────────────────────────────────────────────────
def page_trending(df, demo):
    _section("🔥 Trending Football Topics",
             "The words and topics dominating football discussions right now")
    _live_badge(datetime.utcnow().strftime("%H:%M UTC"), len(df), demo)

    trend_df = extract_trending_topics(df, top_n=20)

    if trend_df.empty:
        st.warning("Not enough data to extract trending topics.")
        return

    st.plotly_chart(_trending_bar(trend_df), width="stretch",
                    config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="cv-subheading">Topic Detail Table</p>',
                unsafe_allow_html=True)
    display = trend_df.copy()
    display.columns = ["Topic","Mentions","Avg Sentiment","Trend Score"]
    st.dataframe(display, width="stretch", height=400)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — LATEST HEADLINES
# ─────────────────────────────────────────────────────────────────────────────
def page_headlines(df, demo):
    _section("📰 Latest Football Headlines",
             "Most recent articles with sentiment scores")
    _live_badge(datetime.utcnow().strftime("%H:%M UTC"), len(df), demo)

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<span class="cv-label">Filter by Sentiment</span>',
                    unsafe_allow_html=True)
        sent_filter = st.selectbox("sent", ["All","Positive","Negative","Neutral"],
                                    label_visibility="collapsed")
    with f2:
        st.markdown('<span class="cv-label">Filter by Source</span>',
                    unsafe_allow_html=True)
        sources = ["All"] + sorted(df["source"].unique().tolist()) \
                  if not df.empty else ["All"]
        src_filter = st.selectbox("src", sources, label_visibility="collapsed")
    with f3:
        st.markdown('<span class="cv-label">Results to show</span>',
                    unsafe_allow_html=True)
        n_show = st.slider("n", 10, 100, 30, label_visibility="collapsed")

    filtered = df.copy()
    if sent_filter != "All":
        filtered = filtered[filtered["label"] == sent_filter]
    if src_filter != "All":
        filtered = filtered[filtered["source"] == src_filter]
    filtered = filtered.head(n_show)

    st.markdown(f'<div style="font-size:0.82rem;color:{C["muted"]};margin-bottom:0.8rem;">'
                f'Showing {len(filtered)} articles</div>', unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        label    = row.get("label", "Neutral")
        compound = row.get("compound", 0)
        color    = (C["green"] if label == "Positive"
                    else C["orange"] if label == "Negative"
                    else C["accent"])
        emotion  = row.get("dominant_emotion", "neutral")
        pub_str  = str(row.get("published_at", ""))[:16]
        team     = row.get("team", "")
        player   = row.get("player", "")

        tags = ""
        if team   and team   != "General": tags += f'<span class="emotion-chip" style="background:rgba(0,212,255,0.1);color:{C["accent"]};border:1px solid {C["accent"]};">⚽ {team}</span>'
        if player and player != "General": tags += f'<span class="emotion-chip" style="background:rgba(255,215,0,0.1);color:{C["gold"]};border:1px solid {C["gold"]};">👤 {player}</span>'
        tags += f'<span class="emotion-chip" style="background:rgba(255,255,255,0.05);color:{EMOTION_COLORS.get(emotion,C["muted"])};border:1px solid {EMOTION_COLORS.get(emotion,C["border"])};">{EMOTION_ICONS.get(emotion,"")} {emotion.title()}</span>'

        st.markdown(f"""
        <div class="news-card" style="border-left:3px solid {color};">
            <div class="news-title">{row.get('title','No title')}</div>
            <div class="news-meta">
                📡 {row.get('source','—')} &nbsp;·&nbsp; 🕐 {pub_str}
                &nbsp;·&nbsp;
                <span style="color:{color};font-weight:600;">{label} ({compound:+.3f})</span>
            </div>
            <div style="margin-top:6px;">{tags}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    page, queries, days_back, demo_mode, refresh, g_key, n_key = render_sidebar()

    if refresh:
        st.cache_data.clear()

    with st.spinner("🔄 Collecting and analysing football news…"):
        df = fetch_and_score(
            guardian_key=g_key,
            newsapi_key=n_key,
            queries=queries,
            days_back=days_back,
            demo_mode=demo_mode,
        )

    if df.empty:
        st.error("No articles collected. Check your API keys or try Demo Mode.")
        st.stop()

    stats = compute_live_stats(df)

    if page == "Live Dashboard":
        page_live_dashboard(df, stats, demo_mode)
    elif page == "Sentiment Timeline":
        page_timeline(df, demo_mode)
    elif page == "Emotion Analysis":
        page_emotions(df, demo_mode)
    elif page == "Team Comparison":
        page_team_comparison(df, demo_mode)
    elif page == "Trending Topics":
        page_trending(df, demo_mode)
    elif page == "Latest Headlines":
        page_headlines(df, demo_mode)


if __name__ == "__main__":
    main()
