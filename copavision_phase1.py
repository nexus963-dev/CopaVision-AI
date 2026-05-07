"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CopaVision AI — Phase 1: Match Outcome Predictor                 ║
║           Senior ML Engineer Pipeline | International Football              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Pipeline Sections:
  1. Data Loading & Preprocessing
  2. Target Variable Engineering
  3. Feature Engineering (rolling/Elo — no data leakage)
  4. Model Training (Logistic Regression + Random Forest)
  5. Evaluation & Comparison
  6. Visualizations
  7. Model Export
  8. Summary & Next Steps
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, f1_score
)

# ─── Output directory ────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/mnt/user-data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FORM_MATCHES = 5       # rolling window for recent form
TRAIN_CUTOFF_YEAR = 2017 # matches before this → train; >= this → test

print("=" * 70)
print("  CopaVision AI  |  Phase 1: Match Outcome Predictor")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
# WHY: Clean, correctly-typed data is the foundation of any ML project.
# We parse dates so we can sort chronologically — critical for leak-free
# feature engineering. We filter to post-2000 because modern football
# (tactics, fitness, data quality) differs structurally from older eras.

def load_and_preprocess(path: str) -> pd.DataFrame:
    """Load raw CSV, enforce types, sort chronologically, filter to 2000+."""
    print("\n[1/7] Loading & preprocessing data...")

    df = pd.read_csv(path)

    # Convert date string → datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically — ESSENTIAL before any rolling operation
    df = df.sort_values("date").reset_index(drop=True)

    # Filter: modern football era only
    df = df[df["date"].dt.year >= 2000].reset_index(drop=True)

    # Cast neutral to int (True→1, False→0) for ML
    df["neutral"] = df["neutral"].astype(int)

    # Add a match_id column for traceability
    df["match_id"] = df.index

    print(f"  ✔ Loaded {len(df):,} matches  |  "
          f"{df['date'].dt.year.min()}–{df['date'].dt.year.max()}")
    print(f"  ✔ Teams: {df['home_team'].nunique()}  |  "
          f"Tournaments: {df['tournament'].nunique()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. TARGET VARIABLE
# ─────────────────────────────────────────────────────────────────────────────
# WHY: We frame prediction as a 3-class problem: Home Win (0), Away Win (1),
# Draw (2). This is standard in football betting/analytics literature and
# captures the full match outcome space.

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome column: 0=Home Win, 1=Away Win, 2=Draw."""
    print("\n[2/7] Creating target variable...")

    conditions = [
        df["home_score"] > df["away_score"],   # Home Win
        df["home_score"] < df["away_score"],   # Away Win
        df["home_score"] == df["away_score"],  # Draw
    ]
    df["outcome"] = np.select(conditions, [0, 1, 2])

    counts = df["outcome"].value_counts().sort_index()
    labels = {0: "Home Win", 1: "Away Win", 2: "Draw"}
    print("  Class distribution:")
    for cls, cnt in counts.items():
        pct = cnt / len(df) * 100
        print(f"    {labels[cls]:10s} → {cnt:,}  ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL DESIGN RULE — NO DATA LEAKAGE:
#   Every feature for match i must be computed using ONLY matches 0..(i-1).
#   We achieve this by iterating through the sorted dataframe and maintaining
#   per-team running histories — never "looking ahead".
#
# FEATURE RATIONALE:
#   • Recent Points (3/1/0): captures current form, the single best proxy
#     for short-term team performance in sports analytics.
#   • Avg Goals Scored/Conceded: offensive & defensive strength signals.
#   • Rolling Goal Difference: net attacking dominance over recent games.
#   • Neutral Venue: eliminates home advantage, shifts dynamics significantly.
#   • Tournament Type: FIFA World Cup ≠ friendly — context matters enormously.
#   • Elo Rating: globally recognised continuous team-strength estimator.
#     Invented for chess, widely used in football analytics. Accounts for
#     opponent quality and recency better than raw win/loss records.

# ── 3a. Tournament importance encoder ────────────────────────────────────────
TOURNAMENT_IMPORTANCE = {
    "FIFA World Cup": 5,
    "UEFA Euro": 5,
    "Copa America": 5,
    "AFC Asian Cup": 4,
    "African Cup of Nations": 4,
    "Gold Cup": 4,
    "FIFA World Cup qualification": 3,
    "UEFA Euro qualification": 3,
    "UEFA Nations League": 3,
    "Friendly": 1,
}

def get_tournament_importance(tournament: str) -> int:
    """Map tournament name to an importance tier (1–5)."""
    for key, val in TOURNAMENT_IMPORTANCE.items():
        if key.lower() in tournament.lower():
            return val
    return 2  # default for unrecognised tournaments


# ── 3b. Elo Rating System ─────────────────────────────────────────────────────
# Standard football Elo implementation:
#   • K-factor scales with match importance
#   • Expected score computed from rating difference via logistic curve
#   • Ratings updated after every match

ELO_DEFAULT = 1500.0   # starting rating for all teams
ELO_K_BASE  = 20.0     # base K-factor

def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for team A vs team B under Elo model."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating: float, expected: float, actual: float, k: float) -> float:
    """Return updated Elo rating after one match."""
    return rating + k * (actual - expected)


# ── 3c. Main feature-engineering function ─────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sequentially build all features for every match row.
    Processes matches in chronological order; for each match reads
    only past records from `team_history` — zero leakage guaranteed.
    """
    print("\n[3/7] Engineering features (sequential — no leakage)...")
    print(f"  Rolling window: last {N_FORM_MATCHES} matches per team")

    # Per-team history: list of dicts {goals_for, goals_against, points}
    team_history: dict[str, list[dict]] = {}
    # Per-team Elo rating (live, updates after each processed match)
    elo_ratings: dict[str, float] = {}

    rows = []  # accumulate feature dicts

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # ── Fetch histories (empty list if first appearance) ──────────────
        h_hist = team_history.get(home, [])
        a_hist = team_history.get(away, [])

        h_recent = h_hist[-N_FORM_MATCHES:]
        a_recent = a_hist[-N_FORM_MATCHES:]

        # ── Recent form points (3=W, 1=D, 0=L) ───────────────────────────
        h_pts  = np.mean([m["points"] for m in h_recent]) if h_recent else 1.0
        a_pts  = np.mean([m["points"] for m in a_recent]) if a_recent else 1.0

        # ── Avg goals scored / conceded ───────────────────────────────────
        h_scored    = np.mean([m["goals_for"]     for m in h_recent]) if h_recent else 1.0
        h_conceded  = np.mean([m["goals_against"] for m in h_recent]) if h_recent else 1.0
        a_scored    = np.mean([m["goals_for"]     for m in a_recent]) if a_recent else 1.0
        a_conceded  = np.mean([m["goals_against"] for m in a_recent]) if a_recent else 1.0

        # ── Rolling goal difference ───────────────────────────────────────
        h_gd = np.mean([m["goals_for"] - m["goals_against"] for m in h_recent]) if h_recent else 0.0
        a_gd = np.mean([m["goals_for"] - m["goals_against"] for m in a_recent]) if a_recent else 0.0

        # ── Elo ratings BEFORE this match ─────────────────────────────────
        h_elo = elo_ratings.get(home, ELO_DEFAULT)
        a_elo = elo_ratings.get(away, ELO_DEFAULT)
        elo_diff = h_elo - a_elo   # positive → home team is stronger

        # ── Tournament features ───────────────────────────────────────────
        tournament_importance = get_tournament_importance(row["tournament"])

        # ── Store this match's feature vector ─────────────────────────────
        rows.append({
            "match_id":                row["match_id"],
            "home_recent_points":      h_pts,
            "away_recent_points":      a_pts,
            "home_avg_goals_scored":   h_scored,
            "away_avg_goals_scored":   a_scored,
            "home_avg_goals_conceded": h_conceded,
            "away_avg_goals_conceded": a_conceded,
            "home_rolling_gd":         h_gd,
            "away_rolling_gd":         a_gd,
            "elo_diff":                elo_diff,
            "home_elo":                h_elo,
            "away_elo":                a_elo,
            "neutral_venue":           row["neutral"],
            "tournament_importance":   tournament_importance,
        })

        # ── NOW update histories and Elo with actual match result ─────────
        hs, as_ = row["home_score"], row["away_score"]

        if hs > as_:
            h_pts_earned, a_pts_earned = 3, 0
            h_actual, a_actual = 1.0, 0.0
        elif hs < as_:
            h_pts_earned, a_pts_earned = 0, 3
            h_actual, a_actual = 0.0, 1.0
        else:
            h_pts_earned, a_pts_earned = 1, 1
            h_actual, a_actual = 0.5, 0.5

        k = ELO_K_BASE * tournament_importance   # higher-stakes → larger update
        h_exp = elo_expected(h_elo, a_elo)
        a_exp = elo_expected(a_elo, h_elo)

        elo_ratings[home] = update_elo(h_elo, h_exp, h_actual, k)
        elo_ratings[away] = update_elo(a_elo, a_exp, a_actual, k)

        for team, gf, ga, pts in [
            (home, hs, as_, h_pts_earned),
            (away, as_, hs, a_pts_earned),
        ]:
            team_history.setdefault(team, []).append({
                "goals_for":     gf,
                "goals_against": ga,
                "points":        pts,
            })

    features_df = pd.DataFrame(rows)
    df = df.merge(features_df, on="match_id")

    print(f"  ✔ Features built for {len(df):,} matches")
    print(f"  ✔ Feature columns: {features_df.columns.drop('match_id').tolist()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (time-based — never random)
# ─────────────────────────────────────────────────────────────────────────────
# WHY TIME-BASED SPLIT:
#   A random split would let the model "see" 2019 matches while training and
#   predict 2010 matches — a form of leakage. In sports analytics the correct
#   split mirrors real deployment: train on the past, predict the future.

FEATURE_COLS = [
    "home_recent_points", "away_recent_points",
    "home_avg_goals_scored", "away_avg_goals_scored",
    "home_avg_goals_conceded", "away_avg_goals_conceded",
    "home_rolling_gd", "away_rolling_gd",
    "elo_diff", "home_elo", "away_elo",
    "neutral_venue", "tournament_importance",
]

def time_split(df: pd.DataFrame):
    """Split into train / test strictly by date (no shuffle)."""
    print(f"\n[4/7] Time-based train/test split (cutoff: {TRAIN_CUTOFF_YEAR})...")

    train = df[df["date"].dt.year < TRAIN_CUTOFF_YEAR]
    test  = df[df["date"].dt.year >= TRAIN_CUTOFF_YEAR]

    X_train = train[FEATURE_COLS].values
    y_train = train["outcome"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["outcome"].values

    print(f"  Train: {len(train):,} matches  "
          f"({train['date'].dt.year.min()}–{train['date'].dt.year.max()})")
    print(f"  Test : {len(test):,}  matches  "
          f"({test['date'].dt.year.min()}–{test['date'].dt.year.max()})")

    return X_train, X_test, y_train, y_test, train, test


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    """Train Logistic Regression (baseline) and Random Forest."""
    print("\n[5/7] Training models...")

    # Logistic Regression — fast, interpretable baseline.
    # C=0.1 provides mild L2 regularisation to prevent overfit on noisy labels.
    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    print("  ✔ Logistic Regression trained")

    # Random Forest — captures non-linear interactions (e.g., Elo × form).
    # 300 trees; class_weight='balanced' handles class imbalance automatically.
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print("  ✔ Random Forest trained")

    return lr, rf


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Home Win", "Away Win", "Draw"]

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Return accuracy, macro F1, confusion matrix, and class report."""
    y_pred = model.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print(f"\n  ── {model_name} ──")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Macro : {f1_mac:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # Probability predictions (for display purposes)
    probs = model.predict_proba(X_test)

    return {
        "name":     model_name,
        "model":    model,
        "y_pred":   y_pred,
        "probs":    probs,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "cm":       cm,
    }


def run_evaluation(lr, rf, X_test, y_test):
    print("\n[6/7] Evaluating models...")
    lr_res = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    rf_res = evaluate_model(rf, X_test, y_test, "Random Forest")
    return lr_res, rf_res


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "accent1": "#58a6ff",
    "accent2": "#3fb950",
    "accent3": "#f78166",
    "accent4": "#d2a8ff",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
}

def _apply_style(fig, axes_list):
    """Apply dark CopaVision theme to figure and axes."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes_list:
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])


def plot_class_distribution(df: pd.DataFrame, save_path: Path):
    """Bar chart showing Home Win / Away Win / Draw distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_style(fig, [ax])

    counts = df["outcome"].value_counts().sort_index()
    colors = [PALETTE["accent2"], PALETTE["accent3"], PALETTE["accent1"]]
    bars = ax.bar(CLASS_NAMES, counts.values, color=colors,
                  edgecolor=PALETTE["border"], linewidth=0.8, width=0.55)

    for bar, cnt in zip(bars, counts.values):
        pct = cnt / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{cnt:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                color=PALETTE["text"], fontsize=10, fontweight="bold")

    ax.set_title("Match Outcome Distribution  (2000–2020)", fontsize=13,
                 fontweight="bold", pad=14)
    ax.set_ylabel("Number of Matches")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", color=PALETTE["border"], linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  → Saved: {save_path.name}")


def plot_confusion_matrices(lr_res: dict, rf_res: dict, save_path: Path):
    """Side-by-side normalised confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _apply_style(fig, axes)

    for ax, res in zip(axes, [lr_res, rf_res]):
        cm_norm = res["cm"].astype(float) / res["cm"].sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            linewidths=0.5,
            linecolor=PALETTE["border"],
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(
            f"{res['name']}\nAcc: {res['accuracy']*100:.1f}%  |  "
            f"F1: {res['f1_macro']:.3f}",
            fontsize=11, fontweight="bold", color=PALETTE["text"]
        )
        ax.set_xlabel("Predicted", color=PALETTE["text"])
        ax.set_ylabel("Actual", color=PALETTE["text"])
        ax.tick_params(colors=PALETTE["text"])

    fig.suptitle("Confusion Matrices — CopaVision AI Phase 1",
                 fontsize=14, fontweight="bold", color=PALETTE["text"], y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  → Saved: {save_path.name}")


def plot_feature_importance(rf_model, save_path: Path):
    """Horizontal bar chart of Random Forest feature importances."""
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)
    features_sorted = [FEATURE_COLS[i] for i in idx]
    imp_sorted = importances[idx]

    colors = [PALETTE["accent1"] if "elo" in f
              else PALETTE["accent2"] if "points" in f
              else PALETTE["accent4"] if "goal" in f or "gd" in f
              else PALETTE["muted"]
              for f in features_sorted]

    fig, ax = plt.subplots(figsize=(9, 6))
    _apply_style(fig, [ax])

    bars = ax.barh(features_sorted, imp_sorted, color=colors,
                   edgecolor=PALETTE["border"], linewidth=0.6, height=0.65)

    for bar, val in zip(bars, imp_sorted):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=PALETTE["text"])

    ax.set_title("Feature Importances — Random Forest", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Importance (Gini)")
    ax.grid(axis="x", color=PALETTE["border"], linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["accent1"], label="Elo Rating"),
        Patch(facecolor=PALETTE["accent2"], label="Recent Form"),
        Patch(facecolor=PALETTE["accent4"], label="Goals / GD"),
        Patch(facecolor=PALETTE["muted"],   label="Context"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  → Saved: {save_path.name}")


def plot_model_comparison(lr_res: dict, rf_res: dict, save_path: Path):
    """Grouped bar chart comparing accuracy & F1 between models."""
    metrics   = ["Accuracy", "F1 Macro"]
    lr_vals   = [lr_res["accuracy"], lr_res["f1_macro"]]
    rf_vals   = [rf_res["accuracy"], rf_res["f1_macro"]]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    _apply_style(fig, [ax])

    b1 = ax.bar(x - width / 2, lr_vals, width, label="Logistic Regression",
                color=PALETTE["accent1"], edgecolor=PALETTE["border"], linewidth=0.8)
    b2 = ax.bar(x + width / 2, rf_vals, width, label="Random Forest",
                color=PALETTE["accent2"], edgecolor=PALETTE["border"], linewidth=0.8)

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.set_title("Model Comparison — CopaVision AI Phase 1",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"], fontsize=9)
    ax.grid(axis="y", color=PALETTE["border"], linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  → Saved: {save_path.name}")


def generate_visualizations(df, lr_res, rf_res, rf_model, output_dir: Path):
    print("\n[7/7] Generating visualizations...")
    plot_class_distribution(df,
        output_dir / "1_class_distribution.png")
    plot_confusion_matrices(lr_res, rf_res,
        output_dir / "2_confusion_matrices.png")
    plot_feature_importance(rf_model,
        output_dir / "3_feature_importance.png")
    plot_model_comparison(lr_res, rf_res,
        output_dir / "4_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_models(lr, rf, output_dir: Path):
    """Persist trained models with joblib for later inference."""
    joblib.dump(lr, output_dir / "copavision_lr.pkl")
    joblib.dump(rf, output_dir / "copavision_rf.pkl")
    print("\n  ✔ Models saved:")
    print(f"    {output_dir / 'copavision_lr.pkl'}")
    print(f"    {output_dir / 'copavision_rf.pkl'}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. PROBABILITY PREDICTION DEMO
# ─────────────────────────────────────────────────────────────────────────────

def predict_match_proba(model, feature_vector: list, model_name: str = "Model"):
    """
    Given a pre-built feature vector, return win/draw probabilities.
    In production this would be called via a REST API endpoint.

    feature_vector order matches FEATURE_COLS exactly.
    """
    X = np.array(feature_vector).reshape(1, -1)
    probs = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]

    label_map = {0: "Home Win", 1: "Away Win", 2: "Draw"}
    print(f"\n  [{model_name}] Probability Prediction:")
    for cls, prob in enumerate(probs):
        marker = " ◄ predicted" if cls == pred else ""
        print(f"    {label_map[cls]:10s}: {prob:.3f}  ({prob*100:.1f}%){marker}")
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    DATA_PATH = "/mnt/user-data/uploads/results.csv"

    # 1. Load & preprocess
    df = load_and_preprocess(DATA_PATH)

    # 2. Target variable
    df = create_target(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Train/test split
    X_train, X_test, y_train, y_test, train_df, test_df = time_split(df)

    # 5. Train models
    lr, rf = train_models(X_train, y_train)

    # 6. Evaluate
    lr_res, rf_res = run_evaluation(lr, rf, X_test, y_test)

    # 7. Visualizations
    generate_visualizations(df, lr_res, rf_res, rf, OUTPUT_DIR)

    # 8. Save models
    save_models(lr, rf, OUTPUT_DIR)

    # 9. Probability demo: Brazil vs Argentina on neutral ground
    print("\n──────────────────────────────────────────────")
    print("  DEMO: Brazil vs Argentina (neutral venue)")
    print("──────────────────────────────────────────────")
    # Feature vector: [h_pts, a_pts, h_scr, a_scr, h_con, a_con,
    #                  h_gd, a_gd, elo_diff, h_elo, a_elo, neutral, importance]
    demo_features = [2.2, 2.1, 1.9, 1.8, 0.9, 1.0,
                     0.8, 0.7, 50.0, 1900.0, 1850.0, 1, 5]
    predict_match_proba(rf, demo_features, "Random Forest")
    predict_match_proba(lr, demo_features, "Logistic Regression")

    # Final summary
    best = rf_res if rf_res["accuracy"] >= lr_res["accuracy"] else lr_res
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Best model   : {best['name']}")
    print(f"  Accuracy     : {best['accuracy']*100:.1f}%")
    print(f"  Macro F1     : {best['f1_macro']:.4f}")
    print(f"  Train period : 2000–{TRAIN_CUTOFF_YEAR - 1}")
    print(f"  Test period  : {TRAIN_CUTOFF_YEAR}–2020")
    print(f"  Outputs      : {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
