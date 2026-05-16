"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       CopaVision AI  |  Phase 2: Player Performance Intelligence            ║
║       Rebuilt for Beginner-Friendly UX — v2                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Save as:  pages/02_Player_Dashboard.py

Data files required (in data/ folder next to app.py):
    data/player_stats.csv
    data/player_per90.csv
    data/similarity_features.csv
    data/clustered_player_features.csv
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Player Intelligence | CopaVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — same dark football theme as Phase 1 + new helper classes
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:#050b14; --bg-card:#0e1621; --bg-panel:#121d2e;
    --border:#1e3050;  --accent:#00d4ff;  --green:#00ff9d;
    --orange:#ff6b35;  --gold:#ffd700;    --purple:#d2a8ff;
    --text:#e8f1ff;    --muted:#5a7a9a;
}
.stApp {
    background-color:var(--bg-deep);
    background-image:
        radial-gradient(ellipse at 15% 0%,rgba(0,212,255,0.06) 0%,transparent 50%),
        radial-gradient(ellipse at 85% 100%,rgba(0,255,157,0.04) 0%,transparent 50%);
    font-family:'DM Sans',sans-serif; color:var(--text);
}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.2rem;padding-bottom:2rem;}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--bg-panel)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}

/* Metrics */
[data-testid="stMetric"]{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:14px 18px!important;}
[data-testid="stMetricValue"]{font-family:'Barlow Condensed',sans-serif!important;font-size:1.8rem!important;font-weight:700!important;color:var(--accent)!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.74rem!important;}

/* Inputs */
[data-testid="stSelectbox"]>div>div{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
[data-testid="stMultiSelect"]>div>div{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:8px!important;}

/* Buttons */
.stButton>button{
    background:linear-gradient(135deg,#00d4ff,#0099cc);
    color:#050b14!important;border:none!important;border-radius:8px!important;
    font-family:'Barlow Condensed',sans-serif!important;font-size:1.05rem!important;
    font-weight:700!important;letter-spacing:1px!important;padding:0.5rem 1.6rem!important;
    text-transform:uppercase!important;transition:all 0.2s!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 24px rgba(0,212,255,0.3)!important;}

/* Tabs */
[data-testid="stTab"]{font-family:'Barlow Condensed',sans-serif!important;font-size:0.95rem!important;color:var(--muted)!important;}
[data-testid="stTab"][aria-selected="true"]{color:var(--accent)!important;}

/* Expanders */
[data-testid="stExpander"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:10px!important;}
hr{border-color:var(--border)!important;margin:1rem 0!important;}
.stAlert{background:var(--bg-card)!important;border-radius:10px!important;}
.stSpinner>div{border-top-color:var(--accent)!important;}

/* ── Custom components ── */
.cv-card{background:var(--bg-card);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:0.8rem;}
.cv-card-blue  {border-left:3px solid var(--accent);}
.cv-card-green {border-left:3px solid var(--green);}
.cv-card-orange{border-left:3px solid var(--orange);}
.cv-card-gold  {border-left:3px solid var(--gold);}
.cv-card-purple{border-left:3px solid var(--purple);}

.cv-heading{font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:700;letter-spacing:1px;color:var(--text);text-transform:uppercase;margin-bottom:0.2rem;}
.cv-sub{font-family:'Barlow Condensed',sans-serif;font-size:0.95rem;color:var(--muted);letter-spacing:0.4px;margin-bottom:0.9rem;}
.cv-label{font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;color:var(--muted);letter-spacing:2.5px;text-transform:uppercase;margin-bottom:0.3rem;display:block;}

/* Onboarding / how-to banner */
.howto-banner{
    background:linear-gradient(135deg,rgba(0,212,255,0.08) 0%,rgba(0,255,157,0.05) 100%);
    border:1px solid rgba(0,212,255,0.25);border-radius:14px;
    padding:1.2rem 1.6rem;margin-bottom:1.2rem;
}
.howto-step{
    display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;
}
.howto-num{
    font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;font-weight:900;
    color:var(--accent);min-width:28px;line-height:1;
}
.howto-text{font-size:0.88rem;color:var(--text);line-height:1.5;}
.howto-text b{color:var(--accent);}

/* Player header badge */
.player-badge{
    background:linear-gradient(135deg,var(--bg-card) 0%,#0a1828 100%);
    border:1px solid var(--border);border-left:4px solid var(--accent);
    border-radius:14px;padding:1.4rem 1.8rem;margin-bottom:1rem;
}
.player-name-big{font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:900;color:var(--text);letter-spacing:1px;}
.player-meta{font-size:0.85rem;color:var(--muted);margin-top:0.3rem;line-height:1.8;}

/* Glossary pill */
.gloss{display:inline-block;background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
       color:var(--accent);font-size:0.75rem;padding:2px 9px;border-radius:20px;margin:2px 3px 2px 0;}

/* Archetype chip */
.arch-chip{display:inline-block;font-family:'Barlow Condensed',sans-serif;font-size:0.9rem;
           font-weight:700;letter-spacing:1px;text-transform:uppercase;
           padding:5px 14px;border-radius:6px;margin:3px 3px 3px 0;}

/* Percentile bar */
.pct-row{margin:6px 0;}
.pct-label-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;}
.pct-name{font-size:0.82rem;color:var(--muted);}
.pct-val{font-size:0.82rem;font-weight:600;color:var(--text);}
.pct-track{background:#1e3050;border-radius:4px;height:9px;}
.pct-fill{height:9px;border-radius:4px;}

/* Stat table row */
.stat-row{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e3050;}
.stat-label{font-size:0.83rem;color:var(--muted);}
.stat-val{font-size:0.83rem;font-weight:600;}

/* Tooltip definition box */
.def-box{background:#0a1422;border-left:3px solid var(--gold);border-radius:8px;
         padding:10px 14px;margin:6px 0;font-size:0.82rem;color:var(--text);line-height:1.6;}
.def-term{font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;
          color:var(--gold);margin-bottom:3px;}

/* Section step badge */
.step-badge{display:inline-flex;align-items:center;justify-content:center;
            background:var(--accent);color:#050b14;font-family:'Barlow Condensed',sans-serif;
            font-size:1rem;font-weight:900;width:26px;height:26px;border-radius:50%;
            margin-right:8px;flex-shrink:0;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
MIN_MINUTES = 450

C = {   # colour palette shorthand
    "accent": "#00d4ff", "green": "#00ff9d", "orange": "#ff6b35",
    "gold": "#ffd700", "purple": "#d2a8ff", "muted": "#5a7a9a",
    "bg": "#0e1621", "border": "#1e3050", "text": "#e8f1ff",
}

POSITION_GROUPS = {
    "⛳ Goalkeeper":              ["Goalkeeper"],
    "🛡️ Defender":               ["Center Back","Left Back","Right Back","Left Center Back",
                                   "Right Center Back","Left Wing Back","Right Wing Back"],
    "⚙️ Midfielder":             ["Center Midfield","Left Center Midfield","Right Center Midfield",
                                   "Center Defensive Midfield","Left Defensive Midfield",
                                   "Right Defensive Midfield","Left Midfield","Right Midfield"],
    "🌪️ Attacking Mid / Winger": ["Center Attacking Midfield","Left Attacking Midfield",
                                   "Right Attacking Midfield","Left Wing","Right Wing"],
    "⚡ Forward":                 ["Center Forward","Left Center Forward","Right Center Forward",
                                   "Secondary Striker"],
}

# Plain-English position group names (strip emoji for internal logic)
def _pos_group_key(label):
    return label.split(" ", 1)[1] if " " in label else label

CLUSTER_ARCHETYPES = {
    0: {"name":"Defensive Anchor",     "color":"#ff6b35","icon":"🛡️",
        "desc":"Wins headers, blocks shots, makes last-ditch tackles. The rock at the back."},
    1: {"name":"Ball Winner",          "color":"#d2a8ff","icon":"⚔️",
        "desc":"Hunts the ball relentlessly. High in duels, interceptions, and pressing intensity."},
    2: {"name":"Deep-Lying Playmaker", "color":"#00d4ff","icon":"🎯",
        "desc":"The conductor. Passes the most, most accurately, and moves the ball forward."},
    3: {"name":"Clinical Finisher",    "color":"#ffd700","icon":"⚡",
        "desc":"Lives in the box. High xG, shots on target, and lethal conversion rate."},
    4: {"name":"Box-to-Box Engine",    "color":"#00ff9d","icon":"🔄",
        "desc":"Covers every blade of grass. Contributes in both attack and defence equally."},
    5: {"name":"Creative Winger",      "color":"#ff6b35","icon":"🌪️",
        "desc":"Takes on defenders 1v1, creates from wide areas, and delivers assists."},
    6: {"name":"High Press Striker",   "color":"#ffd700","icon":"🎯",
        "desc":"Scores goals AND pressures defenders. A nightmare to play against."},
    7: {"name":"Possession Carrier",   "color":"#00d4ff","icon":"🎪",
        "desc":"Drives forward with the ball, beats players, and progresses play."},
}

FEATURE_COLS_SIM = [
    "goals_per90_scaled","assists_per90_scaled","shots_per90_scaled",
    "shots_on_target_per90_scaled","xg_per90_scaled",
    "passes_attempted_per90_scaled","passes_completed_per90_scaled",
    "progressive_passes_per90_scaled","dribbles_attempted_per90_scaled",
    "dribbles_completed_per90_scaled","carries_per90_scaled",
    "tackles_per90_scaled","tackles_won_per90_scaled","interceptions_per90_scaled",
    "pressures_per90_scaled","aerial_duels_per90_scaled","aerial_duels_won_per90_scaled",
    "duels_per90_scaled","duels_won_per90_scaled","pass_accuracy_scaled",
    "dribble_success_rate_scaled","tackle_success_rate_scaled",
    "aerial_success_rate_scaled","duel_success_rate_scaled",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_player_stats():
    df = pd.read_csv(DATA_DIR / "player_stats.csv")
    nickname = df["player_nickname"].replace(["0", 0, "", "nan", "None"], np.nan)
    df["display_name"] = nickname.fillna(df["player_name"])
    df = df[df["position"] != "0"].copy()
    # Attach readable position group label
    def _group(pos):
        for label, positions in POSITION_GROUPS.items():
            if pos in positions:
                return label
        return "Other"
    df["pos_group"] = df["position"].apply(_group)
    return df

@st.cache_data(show_spinner=False)
def load_per90():
    df = pd.read_csv(DATA_DIR / "player_per90.csv")
    nickname = df["player_nickname"].replace(["0", 0, "", "nan", "None"], np.nan)
    df["display_name"] = nickname.fillna(df["player_name"])
    df = df[df["position"] != "0"].copy()
    return df

@st.cache_data(show_spinner=False)
def load_similarity():
    df = pd.read_csv(DATA_DIR / "similarity_features.csv").fillna(0)
    display_lookup = load_player_stats()[["player_id", "display_name"]].drop_duplicates("player_id")
    df = df.merge(display_lookup, on="player_id", how="left")
    df["display_name"] = df["display_name"].fillna(df["player_name"])
    return df

@st.cache_data(show_spinner=False)
def load_clusters():
    REQUIRED = {"player_id","cluster","pca_x","pca_y","tsne_x","tsne_y"}
    path = DATA_DIR / "clustered_player_features.csv"
    if path.exists():
        df = pd.read_csv(path)
        if REQUIRED.issubset(set(df.columns)):
            display_lookup = load_player_stats()[["player_id", "display_name"]].drop_duplicates("player_id")
            df = df.merge(display_lookup, on="player_id", how="left")
            df["display_name"] = df["display_name"].fillna(df["player_name"])
            return df
    return _build_clusters()

@st.cache_data(show_spinner=False)
def _build_clusters():
    sim = load_similarity()
    X = sim[FEATURE_COLS_SIM].values
    cl = sim[["player_id","player_name","primary_position","team_name","competition_name"]].copy()
    cl["cluster"] = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(X)
    pca = PCA(n_components=2, random_state=42).fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
    cl["pca_x"], cl["pca_y"] = pca[:,0], pca[:,1]
    cl["tsne_x"], cl["tsne_y"] = tsne[:,0], tsne[:,1]
    display_lookup = load_player_stats()[["player_id", "display_name"]].drop_duplicates("player_id")
    cl = cl.merge(display_lookup, on="player_id", how="left")
    cl["display_name"] = cl["display_name"].fillna(cl["player_name"])
    try:
        cl.to_csv(DATA_DIR / "clustered_player_features.csv", index=False)
    except Exception:
        pass
    return cl

# ─────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt(val, dec=2, suffix=""):
    if pd.isna(val): return "—"
    if isinstance(val, float): return f"{val:.{dec}f}{suffix}"
    return f"{val}{suffix}"

def pct_rank(series, val):
    s = series.dropna()
    if s.std() == 0: return 50.0
    return float((s < val).mean() * 100)

def pct_color(p):
    if p >= 85: return C["green"]
    if p >= 65: return C["accent"]
    if p >= 40: return C["gold"]
    return C["orange"]

def pct_bar(label, pct, raw="", explain=""):
    col = pct_color(pct)
    badge = ("🔥 Elite" if pct>=85 else "✅ Good" if pct>=65 else
             "➡️ Average" if pct>=40 else "📉 Below Avg")
    tip = f" · <span style='color:{C['muted']};font-size:0.75rem;'>{explain}</span>" if explain else ""
    return f"""
    <div class="pct-row">
        <div class="pct-label-row">
            <span class="pct-name">{label}{tip}</span>
            <span class="pct-val" style="color:{col};">
                {raw}&nbsp;<span style="font-size:0.75rem;opacity:0.8;">{pct:.0f}th · {badge}</span>
            </span>
        </div>
        <div class="pct-track"><div class="pct-fill" style="width:{min(pct,100):.0f}%;background:{col};"></div></div>
    </div>"""

def stat_row(label, val, color=None, tooltip=""):
    color = color or C["text"]
    return (f'<div class="stat-row">'
            f'<span class="stat-label">{label}</span>'
            f'<span class="stat-val" style="color:{color};">{val}</span>'
            f'</div>')

def _PLOT_BASE():
    return dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")

# ─────────────────────────────────────────────────────────────────────────────
# GLOSSARY COMPONENT  (shown inline wherever metrics appear)
# ─────────────────────────────────────────────────────────────────────────────
GLOSSARY = {
    "xG (Expected Goals)":
        "A probability score (0–1) for each shot — how likely it was to go in based on position, angle, and body part. "
        "Higher xG = better quality chances. A striker with xG 20 but only 10 goals is under-performing; 25 goals = over-performing.",
    "xA (Expected Assists)":
        "Same idea as xG but for passes that led to shots. Measures the quality of chances created, not just whether the finish went in.",
    "Per 90 (p90)":
        "Stats divided by 90 minutes played, so you can fairly compare a player who played 900 mins to one who played 3,600 mins.",
    "Percentile Rank":
        "Your rank vs ALL players in the database. 90th percentile = better than 90% of players at that stat. 50th = exactly average.",
    "Progressive Passes":
        "Passes that move the ball at least 10 metres closer to the opponent's goal. Measures forward intent in build-up play.",
    "Pressures":
        "How often the player actively tries to win the ball back by pressuring the opponent in possession. A key pressing metric.",
    "Duel Success Rate":
        "% of 1v1 physical battles won. High rate = dominant in physical contests.",
}

def glossary_expander():
    with st.expander("📖  Glossary — What do these numbers mean?", expanded=False):
        for term, definition in GLOSSARY.items():
            st.markdown(f"""
            <div class="def-box">
                <div class="def-term">{term}</div>
                {definition}
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Brand
        st.markdown("""
        <div style="text-align:center;padding:0.8rem 0 1rem;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;
                 font-weight:900;color:#00d4ff;letter-spacing:3px;">
                 COPA<span style="color:#00ff9d;">VISION</span></div>
            <div style="font-size:0.75rem;color:#5a7a9a;letter-spacing:4px;">A I · P H A S E  2</div>
        </div><hr>""", unsafe_allow_html=True)

        # Navigation with plain labels
        st.markdown('<span class="cv-label">CHOOSE A TOOL</span>', unsafe_allow_html=True)
        page = st.radio("nav", [
            "🔍  Search a Player",
            "⚖️   Compare Two Players",
            "🗺️   Explore Player Types",
            "🔗  Find Similar Players",
        ], label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem;line-height:1.8;color:#8a9ab0;">
            Choose a tool above, then use the page controls to filter players and compare stats.
        </div>""", unsafe_allow_html=True)

    # Strip emoji prefix for routing
    return page.strip().split("  ")[-1].strip()


# ─────────────────────────────────────────────────────────────────────────────
# ── SHARED: STEP-BY-STEP PLAYER PICKER  (used on 3 pages)
# ─────────────────────────────────────────────────────────────────────────────
def player_picker(ps_df, pp_df, key_prefix="pp", min_mins=MIN_MINUTES,
                  show_howto=True, label_override=None):
    """
    Guided 3-step player picker:
      Step 1 — choose position group
      Step 2 — choose competition (optionally also team)
      Step 3 — choose player from the filtered shortlist

    Returns: (player_row from ps_df, per90_row from pp_df, player_id)
    """
    # ── Onboarding banner ─────────────────────────────────────────────────────
    if show_howto:
        st.markdown("""
        <div class="howto-banner">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                 font-weight:700;color:#00d4ff;letter-spacing:1px;margin-bottom:10px;">
                 🧭 HOW TO PICK A PLAYER — 3 EASY STEPS</div>
            <div class="howto-step">
                <div class="howto-num">1</div>
                <div class="howto-text">
                    <b>Choose a position group</b> — are you looking for a goalkeeper, defender, midfielder, or forward?
                    Leave it on <b>"All positions"</b> to browse everyone.
                </div>
            </div>
            <div class="howto-step">
                <div class="howto-num">2</div>
                <div class="howto-text">
                    <b>Pick a competition</b> (league/tournament) and optionally a <b>team</b>.
                    This narrows the player list so it's not overwhelming.
                </div>
            </div>
            <div class="howto-step">
                <div class="howto-num">3</div>
                <div class="howto-text">
                    <b>Select the player</b> from the dropdown. Stats load instantly below.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 1: Position ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;margin-bottom:4px;">
        <div class="step-badge">1</div>
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
              font-weight:700;color:#e8f1ff;">Choose a position group</span>
    </div>""", unsafe_allow_html=True)

    pos_options = ["All positions"] + list(POSITION_GROUPS.keys())
    pos_labels  = {opt: opt for opt in pos_options}

    pos_choice = st.selectbox(
        "position group",
        pos_options,
        index=0,
        format_func=lambda x: x,
        label_visibility="collapsed",
        key=f"{key_prefix}_pos",
        help="Pick the type of player you want to explore. Select 'All positions' to see every player.",
    )

    # Filter by position
    if pos_choice == "All positions":
        filtered = ps_df.copy()
    else:
        allowed = POSITION_GROUPS.get(pos_choice, [])
        filtered = ps_df[ps_df["position"].isin(allowed)].copy()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 2: Competition + Team ─────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;margin-bottom:4px;">
        <div class="step-badge">2</div>
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
              font-weight:700;color:#e8f1ff;">Narrow by competition, team &amp; nationality <span style="color:#5a7a9a;font-size:0.85rem;">(optional)</span></span>
    </div>""", unsafe_allow_html=True)

    col_comp, col_team, col_country = st.columns(3)
    with col_comp:
        comps = ["All competitions"] + sorted(filtered["competition"].unique())
        comp_choice = st.selectbox(
            "competition", comps, index=0,
            label_visibility="collapsed",
            key=f"{key_prefix}_comp",
            help="e.g. 'Premier League', 'FIFA World Cup', 'La Liga'",
        )
    if comp_choice != "All competitions":
        filtered = filtered[filtered["competition"] == comp_choice]

    with col_team:
        teams = ["All teams"] + sorted(filtered["team"].unique())
        team_choice = st.selectbox(
            "team", teams, index=0,
            label_visibility="collapsed",
            key=f"{key_prefix}_team",
            help="Filter to one club or national team",
        )
    if team_choice != "All teams":
        filtered = filtered[filtered["team"] == team_choice]

    with col_country:
        countries = ["All nationalities"] + sorted(filtered["country"].dropna().astype(str).unique())
        country_choice = st.selectbox(
            "nationality", countries, index=0,
            label_visibility="collapsed",
            key=f"{key_prefix}_country",
            help="Filter by player nationality when available",
        )
    if country_choice != "All nationalities":
        filtered = filtered[filtered["country"].astype(str) == country_choice]

    if "age" in filtered.columns and filtered["age"].notna().any():
        min_age = int(np.nanmin(filtered["age"]))
        max_age = int(np.nanmax(filtered["age"]))
        if min_age < max_age:
            age_range = st.slider(
                "Age range",
                min_age, max_age, (min_age, max_age),
                key=f"{key_prefix}_age",
                help="Only shown when the processed dataset includes an age column.",
            )
            filtered = filtered[filtered["age"].between(age_range[0], age_range[1])]

    # Minimum minutes guard
    filtered = filtered[filtered["minutes_played"] >= min_mins]

    # How many players remain
    n_avail = filtered["display_name"].nunique()
    if n_avail == 0:
        st.warning("⚠️ No players match these filters. Try choosing 'All competitions' or a different position.")
        return None, None, None

    st.markdown(f"""
    <div style="font-size:0.82rem;color:{C['muted']};margin:6px 0 12px;">
        ✅ <b style="color:{C['text']};">{n_avail}</b> players available with ≥{min_mins} minutes played
    </div>""", unsafe_allow_html=True)

    # ── STEP 3: Player ─────────────────────────────────────────────────────────
    label = label_override or "3"
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:4px;">
        <div class="step-badge">{label}</div>
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
              font-weight:700;color:#e8f1ff;">Select the player</span>
    </div>""", unsafe_allow_html=True)

    player_names = sorted(filtered["display_name"].unique())
    selected = st.selectbox(
        "player", player_names,
        label_visibility="collapsed",
        key=f"{key_prefix}_player",
        help="Type a name to search, or scroll through the list.",
    )

    # Resolve rows
    ps_rows = filtered[filtered["display_name"] == selected]
    player_row = ps_rows.sort_values("minutes_played", ascending=False).iloc[0]
    pid = player_row["player_id"]

    pp_rows = pp_df[pp_df["player_id"] == pid]
    if pp_rows.empty:
        pp_rows = pp_df[pp_df["display_name"] == selected]
    per90_row = (pp_rows.sort_values("minutes_played", ascending=False).iloc[0]
                 if not pp_rows.empty else pd.Series(dtype=float))

    return player_row, per90_row, int(pid)


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE HEADER CARD
# ─────────────────────────────────────────────────────────────────────────────
def render_profile_header(player_row, cl_df):
    arch = {}
    if cl_df is not None:
        cl_rows = cl_df[cl_df["player_id"] == player_row["player_id"]]
        if not cl_rows.empty:
            try:
                arch = CLUSTER_ARCHETYPES.get(int(cl_rows.iloc[0]["cluster"]), {})
            except Exception:
                arch = {}

    ac = arch.get("color", C["accent"])
    an = arch.get("name",  "Unknown")
    ai = arch.get("icon",  "⚽")
    ad = arch.get("desc",  "")

    mins  = player_row.get("minutes_played", 0)
    match = player_row.get("matches_played", 0)
    comp  = player_row.get("competition", "—")
    team  = player_row.get("team", "—")
    pos   = player_row.get("position", "—")
    nat   = player_row.get("country", "—")
    name  = player_row.get("display_name", player_row.get("player_name","?"))

    st.markdown(f"""
    <div class="player-badge">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
            <div>
                <div class="player-name-big">{name}</div>
                <div class="player-meta">
                    🏟️ <b style="color:{C['text']};">{team}</b>
                    &nbsp;·&nbsp; 📍 {pos}
                    &nbsp;·&nbsp; 🌍 {nat}
                    &nbsp;·&nbsp; 🏆 {comp}
                </div>
                <div class="player-meta" style="margin-top:4px;font-size:0.8rem;">
                    ⏱️ {fmt(mins,0)} minutes played across {fmt(match,0)} matches
                </div>
            </div>
            <div style="text-align:right;">
                <span class="arch-chip"
                      style="background:rgba(255,255,255,0.07);color:{ac};border:1px solid {ac};">
                      {ai}&nbsp;{an}
                </span>
                <div style="font-size:0.78rem;color:{C['muted']};max-width:200px;margin-top:6px;
                     text-align:right;line-height:1.4;">{ad}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PERCENTILE BAR CHART (Plotly — beginner friendly labels)
# ─────────────────────────────────────────────────────────────────────────────
def make_pct_chart(per90_row, pp_df, title=""):
    METRICS = [
        ("Goals per 90 min",              "goals_per90",               "Goals scored per 90 min"),
        ("Assists per 90 min",            "assists_per90",              "Goal assists per 90 min"),
        ("Expected Goals (xG) / 90",      "xg_per90",                  "Shot quality score / 90"),
        ("Expected Assists (xA) / 90",    "xassists_per90",            "Chance creation quality"),
        ("Progressive Passes / 90",       "progressive_passes_per90",  "Forward-moving passes / 90"),
        ("Completed Dribbles / 90",       "dribbles_completed_per90",  "Dribbles won / 90"),
        ("Tackles / 90",                  "tackles_per90",             "Tackles attempted / 90"),
        ("Interceptions / 90",            "interceptions_per90",       "Passes cut out / 90"),
        ("Pressures / 90",                "pressures_per90",           "Press attempts / 90"),
        ("Pass Accuracy %",               "pass_accuracy",             "% passes reaching teammate"),
    ]
    labels, pcts, colors, raws = [], [], [], []
    for lbl, col, _ in METRICS:
        if col in pp_df.columns and col in per90_row.index:
            p = pct_rank(pp_df[col], per90_row.get(col, 0))
            labels.append(lbl); pcts.append(p)
            colors.append(pct_color(p))
            raws.append(fmt(per90_row.get(col, 0)))

    fig = go.Figure(go.Bar(
        x=pcts, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.0f}th percentile" for p in pcts],
        textposition="outside",
        textfont=dict(family="Barlow Condensed", size=11, color=C["text"]),
        customdata=raws,
        hovertemplate="<b>%{y}</b><br>Value: %{customdata}<br>Percentile: %{x:.0f}th<extra></extra>",
    ))
    # Add a vertical "Average" line at 50th percentile
    fig.add_vline(x=50, line=dict(color=C["muted"], dash="dash", width=1),
                  annotation_text="Average (50th)", annotation_position="top",
                  annotation_font=dict(color=C["muted"], size=10))

    fig.update_layout(
        **_PLOT_BASE(),
        xaxis=dict(range=[0,118], showgrid=False, visible=False),
        yaxis=dict(tickfont=dict(family="DM Sans", size=11, color=C["text"])),
        height=400, margin=dict(t=30,b=10,l=10,r=80),
        title=dict(text=title or "How does this player rank vs all players?",
                   font=dict(family="Barlow Condensed", size=13, color=C["text"])),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────────────────────────────────────
RADAR_CATS = {
    "⚽ Finishing":      ["goals_per90","xg_per90","shot_accuracy"],
    "🎨 Creativity":    ["assists_per90","xassists_per90","progressive_passes_per90"],
    "🚗 Ball Carrying": ["dribbles_completed_per90","carries_per90"],
    "🛡️ Defending":    ["tackles_per90","interceptions_per90","clearances_per90"],
    "💪 Physicality":  ["aerial_duels_per90","duels_won_per90"],
    "📨 Passing":      ["passes_completed_per90","pass_accuracy"],
}

def make_radar(r1, r2, n1, n2, pp_df):
    labels, v1, v2 = [], [], []

    for cat, cols in RADAR_CATS.items():
        for col in cols:
            if col in pp_df.columns:
                labels.append(
                    f"{cat}\n{col.replace('_per90','').replace('_',' ').title()}"
                )

                v1.append(pct_rank(pp_df[col], r1.get(col, 0)))
                v2.append(pct_rank(pp_df[col], r2.get(col, 0)))

    fig = go.Figure()

    for nm, vals, col, fill_col in [
        (n1, v1, C["green"],  "rgba(0,255,157,0.20)"),
        (n2, v2, C["accent"], "rgba(0,212,255,0.15)")
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor=fill_col,
            line=dict(color=col, width=2),
            name=nm,
            hovertemplate="<b>%{theta}</b><br>Percentile: %{r:.0f}<extra></extra>",
        ))

    fig.update_layout(
        **_PLOT_BASE(),
        polar=dict(
            bgcolor="rgba(14,22,33,0.6)",

            radialaxis=dict(
                visible=True,
                range=[0,100],
                showticklabels=False,
                gridcolor=C["border"]
            ),

            angularaxis=dict(
                tickfont=dict(
                    family="DM Sans",
                    size=9,
                    color=C["text"]
                ),
                gridcolor=C["border"]
            ),
        ),

        legend=dict(
            font=dict(
                family="Barlow Condensed",
                color=C["text"],
                size=13
            ),
            bgcolor="rgba(0,0,0,0)"
        ),

        height=420,
        margin=dict(t=40,b=40,l=50,r=50),
    )

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER SCATTER
# ─────────────────────────────────────────────────────────────────────────────
def make_cluster_scatter(cl_df, highlight_id=None, mode="pca"):
    x_col, y_col = (("tsne_x", "tsne_y") if mode == "tsne" else ("pca_x", "pca_y"))
    title = "t-SNE player style map" if mode == "tsne" else "PCA player style map"

    plot_df = cl_df.copy()
    plot_df["cluster"] = plot_df["cluster"].astype(int)
    plot_df["archetype"] = plot_df["cluster"].apply(
        lambda cid: CLUSTER_ARCHETYPES.get(cid, {}).get("name", f"Cluster {cid}")
    )
    plot_df["hover_team"] = plot_df.get("team_name", "Unknown")
    plot_df["hover_comp"] = plot_df.get("competition_name", "Unknown")
    plot_df["hover_pos"] = plot_df.get("primary_position", "Unknown")

    fig = go.Figure()
    for cid in sorted(plot_df["cluster"].unique()):
        subset = plot_df[plot_df["cluster"] == cid]
        arch = CLUSTER_ARCHETYPES.get(int(cid), {})
        name = arch.get("name", f"Cluster {cid}")
        color = arch.get("color", C["accent"])
        fig.add_trace(go.Scattergl(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            name=name,
            marker=dict(
                size=8,
                color=color,
                opacity=0.68,
                line=dict(width=0.5, color="rgba(232,241,255,0.25)"),
            ),
            customdata=np.stack([
                subset["display_name"].astype(str),
                subset["hover_pos"].astype(str),
                subset["hover_team"].astype(str),
                subset["hover_comp"].astype(str),
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Type: " + name + "<br>"
                "Position: %{customdata[1]}<br>"
                "Team: %{customdata[2]}<br>"
                "Competition: %{customdata[3]}<extra></extra>"
            ),
        ))

    if highlight_id is not None:
        highlighted = plot_df[plot_df["player_id"] == highlight_id]
        if not highlighted.empty:
            row = highlighted.iloc[0]
            fig.add_trace(go.Scattergl(
                x=[row[x_col]],
                y=[row[y_col]],
                mode="markers+text",
                name="Highlighted player",
                text=[row["display_name"]],
                textposition="top center",
                textfont=dict(family="Barlow Condensed", size=13, color=C["gold"]),
                marker=dict(
                    size=18,
                    color=C["gold"],
                    symbol="star",
                    line=dict(width=2, color=C["text"]),
                ),
                hovertemplate=(
                    f"<b>{row['display_name']}</b><br>"
                    f"Type: {row['archetype']}<br>"
                    f"Position: {row['hover_pos']}<br>"
                    f"Team: {row['hover_team']}<extra></extra>"
                ),
            ))

    fig.update_layout(
        **_PLOT_BASE(),
        title=dict(
            text=title,
            font=dict(family="Barlow Condensed", size=16, color=C["text"]),
        ),
        xaxis=dict(
            title="Style dimension 1",
            showgrid=True,
            gridcolor=C["border"],
            zeroline=False,
            tickfont=dict(color=C["muted"]),
            title_font=dict(color=C["muted"]),
        ),
        yaxis=dict(
            title="Style dimension 2",
            showgrid=True,
            gridcolor=C["border"],
            zeroline=False,
            tickfont=dict(color=C["muted"]),
            title_font=dict(color=C["muted"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(family="Barlow Condensed", size=11, color=C["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=560,
        margin=dict(t=92, b=40, l=20, r=20),
    )
    return fig


def _comparison_bars(r1, r2, n1, n2, pp_df):

    METRICS = [
        ("Goals / 90",       "goals_per90"),
        ("Assists / 90",     "assists_per90"),
        ("xG / 90",          "xg_per90"),
        ("xA / 90",          "xassists_per90"),
        ("Prog. Passes / 90","progressive_passes_per90"),
        ("Dribbles / 90",    "dribbles_completed_per90"),
        ("Tackles / 90",     "tackles_per90"),
        ("Interceptions / 90","interceptions_per90"),
        ("Pass Accuracy",    "pass_accuracy"),
        ("Pressures / 90",   "pressures_per90"),
    ]

    labels, v1, v2 = [], [], []

    for lbl, col in METRICS:
        if col in pp_df.columns:
            labels.append(lbl)

            v1.append(
                pct_rank(pp_df[col], r1.get(col, 0))
            )

            v2.append(
                pct_rank(pp_df[col], r2.get(col, 0))
            )

    st.markdown("""
    <div class="cv-card cv-card-blue" style="margin-bottom:0.8rem;">
        <b style="color:#00d4ff;">Reading this chart</b> —
        bars show <b>percentile rank</b> (0–100).
        Longer bar = better than more players at that stat.
        50 = average.
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=n1,
        x=labels,
        y=v1,
        marker_color=C["green"],
        text=[f"{v:.0f}" for v in v1],
        textposition="outside",
        textfont=dict(
            size=10,
            color=C["text"]
        ),
        hovertemplate="<b>%{x}</b><br>" + n1 + ": %{y:.0f}th percentile<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        name=n2,
        x=labels,
        y=v2,
        marker_color=C["accent"],
        text=[f"{v:.0f}" for v in v2],
        textposition="outside",
        textfont=dict(
            size=10,
            color=C["text"]
        ),
        hovertemplate="<b>%{x}</b><br>" + n2 + ": %{y:.0f}th percentile<extra></extra>"
    ))

    fig.add_hline(
        y=50,
        line=dict(
            color=C["muted"],
            dash="dash",
            width=1
        ),
        annotation_text="Average",
        annotation_position="right",
        annotation_font=dict(
            color=C["muted"],
            size=10
        )
    )

    fig.update_layout(
        **_PLOT_BASE(),
        barmode="group",

        xaxis=dict(
            tickfont=dict(
                family="Barlow Condensed",
                size=10,
                color=C["text"]
            )
        ),

        yaxis=dict(
            range=[0,125],
            gridcolor=C["border"],

            title=dict(
                text="Percentile Rank",
                font=dict(
                    color=C["muted"]
                )
            )
        ),

        legend=dict(
            font=dict(
                family="Barlow Condensed",
                color=C["text"],
                size=13
            ),
            bgcolor="rgba(0,0,0,0)"
        ),

        height=380,
        margin=dict(t=20,b=20,l=50,r=20)
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False}
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — SEARCH A PLAYER
# ─────────────────────────────────────────────────────────────────────────────
def page_search(ps_df, pp_df, cl_df):
    st.markdown('<div class="cv-heading">🔍 Search a Player</div>', unsafe_allow_html=True)
    st.markdown('<div class="cv-sub">Pick any player from the database and explore their full performance report</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Guided picker
    player_row, per90_row, pid = player_picker(ps_df, pp_df, key_prefix="s")
    if player_row is None:
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Profile header ────────────────────────────────────────────────────────
    render_profile_header(player_row, cl_df)

    # ── Key numbers strip ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.75rem;
         color:#5a7a9a;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
         KEY NUMBERS AT A GLANCE</div>""", unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("⚽ Goals",   fmt(player_row.get("goals",0), 0),
              help="Total goals scored")
    m2.metric("🎯 Assists", fmt(player_row.get("assists",0), 0),
              help="Total goal assists")
    m3.metric("📈 xG",      fmt(player_row.get("xg",0)),
              help="Expected Goals — how many goals the shots were worth")
    m4.metric("✨ xA",      fmt(player_row.get("xassists",0)),
              help="Expected Assists — quality of chances created")
    m5.metric("⏱️ Minutes", fmt(player_row.get("minutes_played",0), 0),
              help="Total minutes played")
    m6.metric("🏁 Matches", fmt(player_row.get("matches_played",0), 0),
              help="Total matches appeared in")

    # Glossary always available
    st.markdown("<br>", unsafe_allow_html=True)
    glossary_expander()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs([
        "📊 How do they rank?",
        "📋 All Statistics",
        "🎭 Playing Style",
    ])

    with t1:
        st.markdown("""
        <div class="cv-card cv-card-blue" style="margin-bottom:1rem;">
            <b style="color:#00d4ff;">What am I looking at?</b><br>
            <span style="font-size:0.85rem;color:#8a9ab0;">
            Each bar shows how this player ranks compared to <b>every other player</b> in our database.
            A bar at <b>90th percentile</b> means they are better than 90% of players at that stat.
            50th = perfectly average. The dashed line marks the average.
            </span>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(make_pct_chart(per90_row, pp_df), width="stretch",
                            config={"displayModeBar": False})
        with c2:
            # Strengths & weaknesses
            _render_strengths_weaknesses(per90_row, pp_df)

    with t2:
        _render_full_stats_beginner(player_row, per90_row)

    with t3:
        _render_playing_style(player_row, per90_row, pp_df, cl_df)


def _render_strengths_weaknesses(per90_row, pp_df):
    CHECKS = {
        "Goals/90":         "goals_per90",
        "Assists/90":       "assists_per90",
        "xG/90":            "xg_per90",
        "Prog. Passes/90":  "progressive_passes_per90",
        "Dribbles/90":      "dribbles_completed_per90",
        "Tackles/90":       "tackles_per90",
        "Interceptions/90": "interceptions_per90",
        "Pressures/90":     "pressures_per90",
        "Pass Accuracy":    "pass_accuracy",
        "Aerial Duels/90":  "aerial_duels_per90",
    }
    ranked = sorted(
        [(l, pct_rank(pp_df[c], per90_row.get(c,0)))
         for l,c in CHECKS.items() if c in pp_df.columns and c in per90_row.index],
        key=lambda x: x[1], reverse=True
    )
    strong = ranked[:3]
    weak   = ranked[-3:][::-1]

    st.markdown(f"""
    <div class="cv-card cv-card-green">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;
             color:#00ff9d;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">
             💪 BIGGEST STRENGTHS</div>
        {''.join(f'''<div class="stat-row">
            <span class="stat-label">✅ {l}</span>
            <span class="stat-val" style="color:#00ff9d;">{p:.0f}th percentile</span>
        </div>''' for l,p in strong)}
    </div>
    <div class="cv-card cv-card-orange" style="margin-top:0.6rem;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;
             color:#ff6b35;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">
             ⚠️ RELATIVE WEAKNESSES</div>
        <div style="font-size:0.78rem;color:{C['muted']};margin-bottom:8px;">
            These are still compared to ALL players — even a "weakness" can be fine for their position.
        </div>
        {''.join(f'''<div class="stat-row">
            <span class="stat-label">△ {l}</span>
            <span class="stat-val" style="color:#ff6b35;">{p:.0f}th percentile</span>
        </div>''' for l,p in weak)}
    </div>""", unsafe_allow_html=True)


def _render_full_stats_beginner(player_row, per90_row):
    st.markdown("""
    <div class="cv-card cv-card-blue" style="margin-bottom:1rem;">
        <b style="color:#00d4ff;">Reading this table</b><br>
        <span style="font-size:0.85rem;color:#8a9ab0;">
        The left column shows <b>career totals</b>. The right shows <b>per-90-minute rates</b>
        — this is more useful for comparing players who played different amounts of time.
        </span>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="cv-card cv-card-blue">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;
                 color:{C['accent']};text-transform:uppercase;letter-spacing:1.5px;
                 margin-bottom:10px;">⚽ ATTACKING</div>
            {stat_row("Goals (total)",           fmt(player_row.get("goals",0),0),      C["green"])}
            {stat_row("Shots (total)",            fmt(player_row.get("shots",0),0))}
            {stat_row("Shots on Target",          fmt(player_row.get("shots_on_target",0),0))}
            {stat_row("Shot Accuracy %",          fmt(player_row.get("shot_accuracy",0)) + "%")}
            {stat_row("xG — total",               fmt(player_row.get("xg",0)),           C["gold"])}
            <div style="font-size:0.75rem;color:{C['muted']};margin:8px 0 4px;font-style:italic;">Per 90 minutes:</div>
            {stat_row("Goals / 90 min",           fmt(per90_row.get("goals_per90",0)),   C["green"])}
            {stat_row("Shots / 90 min",           fmt(per90_row.get("shots_per90",0)))}
            {stat_row("xG / 90 min",              fmt(per90_row.get("xg_per90",0)),      C["gold"])}
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="cv-card cv-card-green">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;
                 color:{C['green']};text-transform:uppercase;letter-spacing:1.5px;
                 margin-bottom:10px;">🎯 CREATING &amp; PASSING</div>
            {stat_row("Assists (total)",          fmt(player_row.get("assists",0),0),    C["green"])}
            {stat_row("xA — total",               fmt(player_row.get("xassists",0)),     C["gold"])}
            {stat_row("Passes Attempted",         fmt(player_row.get("passes_attempted",0),0))}
            {stat_row("Passes Completed",         fmt(player_row.get("passes_completed",0),0))}
            {stat_row("Pass Accuracy %",          fmt(player_row.get("pass_accuracy",0)) + "%", C["accent"])}
            {stat_row("Progressive Passes",       fmt(player_row.get("progressive_passes",0),0))}
            {stat_row("Dribbles Won/Tried",
                f"{fmt(player_row.get('dribbles_completed',0),0)} / {fmt(player_row.get('dribbles_attempted',0),0)}")}
            {stat_row("Dribble Success %",        fmt(player_row.get("dribble_success_rate",0)) + "%")}
            <div style="font-size:0.75rem;color:{C['muted']};margin:8px 0 4px;font-style:italic;">Per 90 minutes:</div>
            {stat_row("Assists / 90",             fmt(per90_row.get("assists_per90",0)), C["green"])}
            {stat_row("Prog. Passes / 90",        fmt(per90_row.get("progressive_passes_per90",0)))}
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="cv-card cv-card-orange">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;
                 color:{C['orange']};text-transform:uppercase;letter-spacing:1.5px;
                 margin-bottom:10px;">🛡️ DEFENDING</div>
            {stat_row("Tackles (total)",          fmt(player_row.get("tackles",0),0))}
            {stat_row("Tackles Won",              fmt(player_row.get("tackles_won",0),0))}
            {stat_row("Tackle Success %",         fmt(player_row.get("tackle_success_rate",0)) + "%", C["accent"])}
            {stat_row("Interceptions",            fmt(player_row.get("interceptions",0),0),  C["accent"])}
            {stat_row("Pressures",                fmt(player_row.get("pressures",0),0))}
            {stat_row("Clearances",               fmt(player_row.get("clearances",0),0))}
            {stat_row("Aerial Duels Won/Total",
                f"{fmt(player_row.get('aerial_duels_won',0),0)} / {fmt(player_row.get('aerial_duels',0),0)}")}
            {stat_row("Aerial Success %",         fmt(player_row.get("aerial_success_rate",0)) + "%")}
            <div style="font-size:0.75rem;color:{C['muted']};margin:8px 0 4px;font-style:italic;">Per 90 minutes:</div>
            {stat_row("Tackles / 90",             fmt(per90_row.get("tackles_per90",0)))}
            {stat_row("Interceptions / 90",       fmt(per90_row.get("interceptions_per90",0)))}
        </div>""", unsafe_allow_html=True)


def _render_playing_style(player_row, per90_row, pp_df, cl_df):
    arch = {}
    if cl_df is not None:
        cl_rows = cl_df[cl_df["player_id"] == player_row["player_id"]]
        if not cl_rows.empty:
            try: arch = CLUSTER_ARCHETYPES.get(int(cl_rows.iloc[0]["cluster"]), {})
            except Exception: pass

    ac = arch.get("color",  C["accent"])
    an = arch.get("name",   "Unknown")
    ai = arch.get("icon",   "⚽")
    ad = arch.get("desc",   "")

    def idx(cols):
        scores = [pct_rank(pp_df[c], per90_row.get(c,0))
                  for c in cols if c in pp_df.columns and c in per90_row.index]
        return np.mean(scores) if scores else 50.0

    att = idx(["goals_per90","shots_per90","xg_per90","shot_accuracy"])
    cre = idx(["assists_per90","progressive_passes_per90","dribbles_completed_per90","pass_accuracy"])
    dfe = idx(["tackles_per90","interceptions_per90","pressures_per90","clearances_per90"])

    def index_bar(label, val, desc):
        col = pct_color(val)
        badge = ("Elite" if val>=85 else "Strong" if val>=65 else "Average" if val>=40 else "Low")
        return f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:0.88rem;color:{C['text']};font-weight:600;">{label}</span>
                <span style="font-size:0.88rem;color:{col};font-weight:700;">{val:.0f} / 100 — {badge}</span>
            </div>
            <div style="font-size:0.78rem;color:{C['muted']};margin-bottom:5px;">{desc}</div>
            <div class="pct-track">
                <div class="pct-fill" style="width:{min(val,100):.0f}%;background:{col};"></div>
            </div>
        </div>"""

    st.markdown(f"""
    <div class="cv-card cv-card-gold" style="margin-bottom:1rem;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <span style="font-size:2rem;">{ai}</span>
            <div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;
                     font-weight:700;color:{ac};letter-spacing:1px;">Player Type: {an}</div>
                <div style="font-size:0.85rem;color:{C['text']};margin-top:3px;">{ad}</div>
            </div>
        </div>
        <div style="font-size:0.78rem;color:{C['muted']};">
            Determined by machine learning — comparing their statistical fingerprint to 7,000+ players.
        </div>
    </div>
    <div class="cv-card">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.85rem;color:{C['muted']};
             text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;">
             PLAYING STYLE INDICES — scored out of 100 vs all players</div>
        {index_bar("⚽ Attacking Threat",   att, "Goals, shots, and xG quality")}
        {index_bar("🎨 Creative Influence", cre, "Assists, progressive passes, dribbles, passing accuracy")}
        {index_bar("🛡️ Defensive Work",    dfe, "Tackles, interceptions, pressures, clearances")}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — COMPARE TWO PLAYERS
# ─────────────────────────────────────────────────────────────────────────────
def page_compare(ps_df, pp_df, cl_df):
    st.markdown('<div class="cv-heading">⚖️ Compare Two Players</div>', unsafe_allow_html=True)
    st.markdown('<div class="cv-sub">Pick any two players and see how they stack up side-by-side</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Explanation banner
    st.markdown("""
    <div class="howto-banner">
        <b style="color:#00d4ff;">How comparison works</b><br>
        <span style="font-size:0.85rem;color:#8a9ab0;">
        Use the filters below to find <b>Player 1</b>, then scroll down to find <b>Player 2</b>.
        The radar chart shows each player's <b>percentile rank</b> (0–100) — so a bigger shape means
        a stronger all-round player. The bar chart lets you compare individual stats head-to-head.
        </span>
    </div>""", unsafe_allow_html=True)

    # Player 1 picker
    st.markdown("""
    <div style="background:rgba(0,255,157,0.06);border:1px solid rgba(0,255,157,0.2);
         border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;">
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
              font-weight:700;color:#00ff9d;letter-spacing:1px;">👤 PLAYER 1 (Green)</span>
    </div>""", unsafe_allow_html=True)
    p1_row, p1_pp, p1_id = player_picker(ps_df, pp_df, key_prefix="cmp1", show_howto=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # Player 2 picker
    st.markdown("""
    <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
         border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.8rem;">
        <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
              font-weight:700;color:#00d4ff;letter-spacing:1px;">👤 PLAYER 2 (Blue)</span>
    </div>""", unsafe_allow_html=True)
    p2_row, p2_pp, p2_id = player_picker(ps_df, pp_df, key_prefix="cmp2", show_howto=False)

    if p1_row is None or p2_row is None:
        return

    n1 = p1_row.get("display_name", "Player 1")
    n2 = p2_row.get("display_name", "Player 2")

    if p1_id == p2_id:
        st.warning("⚠️ You selected the same player twice. Please choose two different players.")
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    # Side-by-side header cards
    col1, col2 = st.columns(2)
    with col1:
        _mini_card(p1_row, cl_df, C["green"])
    with col2:
        _mini_card(p2_row, cl_df, C["accent"])

    st.markdown("<br>", unsafe_allow_html=True)
    glossary_expander()
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["🕸️ Radar Chart", "📊 Side-by-Side Stats", "📋 Full Table"])

    with t1:
        st.markdown("""
        <div class="cv-card cv-card-blue" style="margin-bottom:0.8rem;">
            <b style="color:#00d4ff;">Reading the radar</b> — Each point on the web = a stat category.
            <b>Further from centre = higher percentile</b>. A bigger overall shape = stronger player in that area.
            Numbers shown are percentile ranks (0–100), not raw stat values.
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(make_radar(p1_pp, p2_pp, n1, n2, pp_df),
                        width="stretch", config={"displayModeBar": False})

    with t2:
        _comparison_bars(p1_pp, p2_pp, n1, n2, pp_df)

    with t3:
        _comparison_table(p1_row, p2_row, p1_pp, p2_pp, n1, n2, pp_df)


def _mini_card(ps_row, cl_df, color):
    arch = {}
    if cl_df is not None:
        cl_rows = cl_df[cl_df["player_id"] == ps_row["player_id"]]
        if not cl_rows.empty:
            try: arch = CLUSTER_ARCHETYPES.get(int(cl_rows.iloc[0]["cluster"]), {})
            except Exception: pass
    ac = arch.get("color", color)
    an = arch.get("name", "—")
    ai = arch.get("icon", "⚽")
    name = ps_row.get("display_name", ps_row.get("player_name","?"))
    st.markdown(f"""
    <div class="cv-card" style="border-left:4px solid {color};">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;
             font-weight:800;color:{color};margin-bottom:4px;">{name}</div>
        <div style="font-size:0.82rem;color:{C['muted']};margin-bottom:10px;">
            {ps_row.get('team','—')} · {ps_row.get('position','—')} · {ps_row.get('competition','—')}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.85rem;">
            <div>⚽ Goals: <b style="color:{C['text']};">{fmt(ps_row.get('goals',0),0)}</b></div>
            <div>🎯 Assists: <b style="color:{C['text']};">{fmt(ps_row.get('assists',0),0)}</b></div>
            <div>📈 xG: <b style="color:{C['text']};">{fmt(ps_row.get('xg',0))}</b></div>
            <div>⏱️ Mins: <b style="color:{C['text']};">{fmt(ps_row.get('minutes_played',0),0)}</b></div>
        </div>
        <div style="margin-top:8px;">
            <span class="arch-chip"
                  style="background:rgba(255,255,255,0.06);color:{ac};border:1px solid {ac};
                  font-size:0.8rem;padding:4px 12px;">
                  {ai} {an}</span>
        </div>
    </div>""", unsafe_allow_html=True)





def _comparison_table(ps1, ps2, pp1, pp2, n1, n2, pp_df):
    st.markdown("""
    <div class="cv-card cv-card-blue" style="margin-bottom:0.8rem;">
        <b style="color:#00d4ff;">Reading this table</b> —
        <span style="color:#00ff9d;">Green</span> = that player wins this stat.
        Numbers in brackets e.g. <span style="color:#5a7a9a;">(82nd)</span> = percentile rank.
    </div>""", unsafe_allow_html=True)

    rows = []
    STATS = [
        ("Goals (total)",    "goals",    ps1,ps2, False),
        ("Assists (total)",  "assists",  ps1,ps2, False),
        ("xG (total)",       "xg",       ps1,ps2, False),
        ("xA (total)",       "xassists", ps1,ps2, False),
        ("Pass Accuracy %",  "pass_accuracy", ps1,ps2, False),
        ("Goals / 90",       "goals_per90",   pp1,pp2, True),
        ("Assists / 90",     "assists_per90", pp1,pp2, True),
        ("xG / 90",          "xg_per90",      pp1,pp2, True),
        ("Prog. Passes / 90","progressive_passes_per90", pp1,pp2, True),
        ("Dribbles / 90",    "dribbles_completed_per90", pp1,pp2, True),
        ("Tackles / 90",     "tackles_per90", pp1,pp2, True),
        ("Interceptions / 90","interceptions_per90", pp1,pp2, True),
        ("Pressures / 90",   "pressures_per90", pp1,pp2, True),
    ]
    for label, col, src1, src2, is_per90 in STATS:
        v1 = src1.get(col, np.nan); v2 = src2.get(col, np.nan)
        if pd.isna(v1) and pd.isna(v2): continue
        pct_col = col if is_per90 else col
        p1 = pct_rank(pp_df[pct_col], v1) if pct_col in pp_df.columns and not pd.isna(v1) else None
        p2 = pct_rank(pp_df[pct_col], v2) if pct_col in pp_df.columns and not pd.isna(v2) else None
        rows.append({
            "Stat": label,
            n1: f"{fmt(v1)} {'('+str(int(p1))+'th)' if p1 is not None else ''}",
            n2: f"{fmt(v2)} {'('+str(int(p2))+'th)' if p2 is not None else ''}",
            "_v1": float(v1) if not pd.isna(v1) else -999,
            "_v2": float(v2) if not pd.isna(v2) else -999,
        })

    df = pd.DataFrame(rows)
    display = df[["Stat", n1, n2]].copy()

    def highlight(row):
        original = df[df["Stat"] == row["Stat"]]
        if original.empty: return ["","",""]
        v1 = original["_v1"].values[0]; v2 = original["_v2"].values[0]
        if v1 > v2:
            return ["","background-color:rgba(0,255,157,0.12);color:#00ff9d;",
                       "background-color:rgba(255,107,53,0.06);color:#8a9ab0;"]
        elif v2 > v1:
            return ["","background-color:rgba(255,107,53,0.06);color:#8a9ab0;",
                       "background-color:rgba(0,212,255,0.12);color:#00d4ff;"]
        return ["","",""]

    st.dataframe(display.style.apply(highlight, axis=1),
                 width="stretch", height=480)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — EXPLORE PLAYER TYPES
# ─────────────────────────────────────────────────────────────────────────────
def page_clusters(ps_df, cl_df):
    st.markdown('<div class="cv-heading">🗺️ Explore Player Types</div>', unsafe_allow_html=True)
    st.markdown('<div class="cv-sub">Our AI grouped 7,000+ players into 8 playing styles using machine learning — explore them here</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Explanation
    st.markdown("""
    <div class="howto-banner">
        <b style="color:#00d4ff;">What is this?</b><br>
        <span style="font-size:0.85rem;color:#8a9ab0;">
        We analysed 24 statistics for every player and used a machine-learning technique
        called <b>KMeans clustering</b> to automatically group players by how they play.
        Players in the same group have similar statistical fingerprints — not the same team or nationality,
        just a similar <b>playing style</b>.<br><br>
        The chart below shows every player as a dot. <b>Dots close together = players who play similarly.</b>
        Hover over any dot to see the player name.
        </span>
    </div>""", unsafe_allow_html=True)

    # Controls row
    c1, c2, c3 = st.columns([3,2,2])
    with c1:
        st.markdown('<span class="cv-label">Highlight a specific player (optional)</span>',
                    unsafe_allow_html=True)
        eligible_ids = set(ps_df.loc[ps_df["minutes_played"] >= MIN_MINUTES, "player_id"])
        highlight_df = cl_df[cl_df["player_id"].isin(eligible_ids)].copy()
        all_names = ["— Don't highlight anyone —"] + sorted(highlight_df["display_name"].dropna().unique())
        hl_name = st.selectbox("hl", all_names, label_visibility="collapsed",
                               help="Search for a player to mark them with a gold star on the chart")
    with c2:
        st.markdown('<span class="cv-label">Filter to one player type</span>',
                    unsafe_allow_html=True)
        type_opts = ["Show all types"] + [f"{a['icon']} {a['name']}" for a in CLUSTER_ARCHETYPES.values()]
        type_filter = st.selectbox("tf", type_opts, label_visibility="collapsed")
    with c3:
        st.markdown('<span class="cv-label">Chart style</span>', unsafe_allow_html=True)
        proj = st.radio("proj", ["PCA (faster)", "t-SNE (richer)"],
                        horizontal=True, label_visibility="collapsed")

    # Resolve highlight
    hl_id = None
    if hl_name != "— Don't highlight anyone —":
        rows = highlight_df[highlight_df["display_name"] == hl_name]
        if not rows.empty:
            hl_id = int(rows.iloc[0]["player_id"])

    # Filter by type
    plot_df = cl_df.copy()
    if type_filter != "Show all types":
        target_name = type_filter.split(" ", 1)[1]  # strip emoji
        cid = next((k for k,v in CLUSTER_ARCHETYPES.items() if v["name"] == target_name), None)
        if cid is not None:
            plot_df = plot_df[plot_df["cluster"] == cid]

    mode = "tsne" if "t-SNE" in proj else "pca"
    st.plotly_chart(make_cluster_scatter(plot_df, hl_id, mode),
                    width="stretch", config={"displayModeBar": False})

    st.markdown('<hr style="border-color:#1e3050;">', unsafe_allow_html=True)

    # Archetype cards — in a 4-column grid
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;
         color:#e8f1ff;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
         THE 8 PLAYER TYPES EXPLAINED</div>""", unsafe_allow_html=True)

    for row_items in [list(CLUSTER_ARCHETYPES.items())[i:i+4] for i in range(0,8,4)]:
        cols = st.columns(4)
        for col, (cid, arch) in zip(cols, row_items):
            count = len(cl_df[cl_df["cluster"] == cid])
            with col:
                st.markdown(f"""
                <div class="cv-card" style="border-top:3px solid {arch['color']};min-height:170px;">
                    <div style="font-size:1.8rem;margin-bottom:4px;">{arch['icon']}</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;
                         font-weight:700;color:{arch['color']};margin-bottom:4px;">{arch['name']}</div>
                    <div style="font-size:0.76rem;color:{C['muted']};margin-bottom:8px;">
                        {count:,} players in this group</div>
                    <div style="font-size:0.82rem;color:{C['text']};line-height:1.5;">
                        {arch['desc']}</div>
                </div>""", unsafe_allow_html=True)

    # Top 10 players in selected type
    if type_filter != "Show all types":
        target_name = type_filter.split(" ", 1)[1]
        cid = next((k for k,v in CLUSTER_ARCHETYPES.items() if v["name"] == target_name), None)
        if cid is not None:
            st.markdown(f"""
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;
                 color:#e8f1ff;text-transform:uppercase;letter-spacing:1px;
                 margin:16px 0 10px;">Top Players — {type_filter}</div>""", unsafe_allow_html=True)
            pids = cl_df[cl_df["cluster"]==cid]["player_id"].tolist()
            top = (ps_df[ps_df["player_id"].isin(pids)]
                   .sort_values("minutes_played", ascending=False)
                   .drop_duplicates("display_name")
                   .head(15)[["display_name","team","competition","position",
                               "goals","assists","minutes_played"]])
            top.columns = ["Player","Team","Competition","Position","Goals","Assists","Minutes"]
            st.dataframe(top.reset_index(drop=True), width="stretch", height=420)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — FIND SIMILAR PLAYERS
# ─────────────────────────────────────────────────────────────────────────────
def page_similar(ps_df, pp_df, sim_df, cl_df):
    st.markdown('<div class="cv-heading">🔗 Find Similar Players</div>', unsafe_allow_html=True)
    st.markdown('<div class="cv-sub">Pick any player and we\'ll find the players who play most like them — great for scouting alternatives</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <div class="howto-banner">
        <b style="color:#00d4ff;">How this works</b><br>
        <span style="font-size:0.85rem;color:#8a9ab0;">
        We compare the player's statistical profile across 24 metrics against every other player
        using a technique called <b>cosine similarity</b>. A score of <b>100% = statistically identical</b>.
        A score of <b>80%+ = very similar playing style</b>.
        This doesn't mean they're equally talented — just that they play in a similar way.
        </span>
    </div>""", unsafe_allow_html=True)

    # Reference player picker
    player_row, per90_row, pid = player_picker(ps_df, pp_df, key_prefix="sim", show_howto=False)
    if player_row is None:
        return

    c1,c2 = st.columns([2,1])
    with c1:
        n_results = st.slider("How many similar players to show?", 5, 20, 10,
                              help="Slide to see more or fewer results")
    with c2:
        same_pos = st.toggle("Same position only",
                             help="Turn this ON to only see players in the same position")

    if st.button("🔍  Find Similar Players", width="content"):
        with st.spinner("Comparing against 7,000+ players…"):
            sim_rows = sim_df[sim_df["player_id"] == pid]
            if sim_rows.empty:
                st.error("Player not found in similarity dataset.")
                return
            X = sim_df[FEATURE_COLS_SIM].values
            idx_val = sim_rows.index[0]
            sims = cosine_similarity(X[idx_val].reshape(1,-1), X)[0]
            result = sim_df.copy(); result["similarity"] = sims
            result = result[result["player_id"] != pid]
            if same_pos:
                my_pos = sim_rows.iloc[0]["primary_position"]
                result = result[result["primary_position"] == my_pos]
            eligible_ids = set(ps_df.loc[ps_df["minutes_played"] >= MIN_MINUTES, "player_id"])
            result = result[result["player_id"].isin(eligible_ids)]
            result = result.nlargest(n_results, "similarity").reset_index(drop=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Reference card
        render_profile_header(player_row, cl_df)
        st.markdown(f"""
        <div style="font-size:0.82rem;color:{C['muted']};margin-bottom:16px;">
            Showing the <b style="color:{C['text']};">{len(result)}</b> most statistically similar players
            {'in the same position' if same_pos else 'from all positions'}
        </div>""", unsafe_allow_html=True)

        # Result cards in 2-column grid
        for i in range(0, len(result), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx2 = i+j
                if idx2 >= len(result): break
                row = result.iloc[idx2]
                score = int(row["similarity"] * 100)
                scol = (C["green"] if score>=90 else C["accent"] if score>=75 else C["gold"])

                # Archetype
                cl_rows = cl_df[cl_df["player_id"] == row["player_id"]] if cl_df is not None else pd.DataFrame()
                arch = {}
                if not cl_rows.empty:
                    try: arch = CLUSTER_ARCHETYPES.get(int(cl_rows.iloc[0]["cluster"]), {})
                    except Exception: pass

                # Plain English similarity label
                sim_label = ("🟢 Virtually identical" if score>=92 else
                             "✅ Very similar"        if score>=80 else
                             "🔵 Somewhat similar"   if score>=65 else
                             "⚪ Loosely similar")

                with col:
                    st.markdown(f"""
                    <div class="cv-card" style="border-left:3px solid {scol};">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                            <div>
                                <div style="font-family:'Barlow Condensed',sans-serif;
                                     font-size:1.1rem;font-weight:700;color:{C['text']};">
                                     #{idx2+1} &nbsp; {row['display_name']}</div>
                                <div style="font-size:0.8rem;color:{C['muted']};margin-top:2px;">
                                    {row['primary_position']} · {row['team_name']}</div>
                                <div style="font-size:0.76rem;color:{C['muted']};">
                                    {row['competition_name']}</div>
                            </div>
                            <div style="text-align:right;flex-shrink:0;margin-left:10px;">
                                <div style="font-family:'Barlow Condensed',sans-serif;
                                     font-size:2rem;font-weight:900;color:{scol};
                                     line-height:1;">{score}%</div>
                                <div style="font-size:0.68rem;color:{C['muted']};">SIMILARITY</div>
                            </div>
                        </div>
                        <div style="margin:8px 0 4px;">
                            <div style="background:#1e3050;border-radius:4px;height:6px;">
                                <div style="width:{score}%;height:6px;border-radius:4px;
                                     background:{scol};"></div>
                            </div>
                        </div>
                        <div style="font-size:0.8rem;color:{scol};margin-top:6px;">{sim_label}</div>
                        <div style="margin-top:6px;">
                            <span class="arch-chip"
                                  style="font-size:0.75rem;padding:3px 10px;
                                  background:rgba(255,255,255,0.05);
                                  color:{arch.get('color',C['muted'])};
                                  border:1px solid {arch.get('color',C['border'])};">
                                  {arch.get('icon','⚽')} {arch.get('name','—')}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center;padding:2.5rem;color:{C['muted']};font-size:0.95rem;">
            👆 Select a player above, then click <b style="color:{C['accent']};">Find Similar Players</b>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Loading player data…"):
        ps_df  = load_player_stats()
        pp_df  = load_per90()
        sim_df = load_similarity()
        cl_df  = load_clusters()

    page = render_sidebar()

    if "Search" in page:
        page_search(ps_df, pp_df, cl_df)
    elif "Compare" in page:
        page_compare(ps_df, pp_df, cl_df)
    elif "Explore" in page or "Types" in page:
        page_clusters(ps_df, cl_df)
    elif "Similar" in page:
        page_similar(ps_df, pp_df, sim_df, cl_df)


if __name__ == "__main__":
    main()