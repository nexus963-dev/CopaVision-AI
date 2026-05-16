"""
CopaVision AI — Phase 3
scripts/momentum_engine.py

Football sentiment momentum analysis:
  - rolling sentiment averages
  - spike detection
  - team/player aggregation
  - trending topic extraction
  - emotion timeline building
"""

import re
import logging
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING SENTIMENT & MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def build_sentiment_timeline(df: pd.DataFrame,
                              freq: str = "1h") -> pd.DataFrame:
    """
    Aggregate sentiment scores into a time-series timeline.

    Args:
        df:   Scored articles DataFrame from sentiment_pipeline.score_articles()
        freq: Pandas resample frequency — '30T'=30min, '1h'=hourly, '1d'=daily

    Returns:
        DataFrame with columns: timestamp, avg_compound, pos_pct, neg_pct,
                                 neu_pct, article_count, rolling_avg
    """
    if df.empty or "published_at" not in df.columns:
        return pd.DataFrame()

    freq = freq.strip().lower()

    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df = df.set_index("published_at").sort_index()

    agg = df["compound"].resample(freq).agg(
        avg_compound="mean",
        article_count="count",
    ).reset_index()

    agg.columns = ["timestamp", "avg_compound", "article_count"]

    # Label distribution per bucket
    def _label_pcts(sub_df, ts, f):
        bucket = df.loc[
            (df.index >= ts) & (df.index < ts + pd.tseries.frequencies.to_offset(f))
        ]
        if bucket.empty:
            return 0.0, 0.0, 1.0
        total = len(bucket)
        pos = (bucket["label"] == "Positive").sum() / total
        neg = (bucket["label"] == "Negative").sum() / total
        neu = 1 - pos - neg
        return pos, neg, neu

    agg["pos_pct"] = agg.apply(
        lambda r: _label_pcts(df, r["timestamp"], freq)[0], axis=1)
    agg["neg_pct"] = agg.apply(
        lambda r: _label_pcts(df, r["timestamp"], freq)[1], axis=1)
    agg["neu_pct"] = agg.apply(
        lambda r: _label_pcts(df, r["timestamp"], freq)[2], axis=1)

    # Rolling 3-period average momentum curve
    agg["rolling_avg"] = (agg["avg_compound"]
                          .rolling(window=3, min_periods=1)
                          .mean())

    agg["avg_compound"] = agg["avg_compound"].round(4)
    agg["rolling_avg"]  = agg["rolling_avg"].round(4)
    return agg.dropna(subset=["avg_compound"])


def detect_spikes(timeline: pd.DataFrame,
                  threshold: float = 0.25) -> pd.DataFrame:
    """
    Detect sentiment spikes — moments where sentiment shifts sharply.

    A spike is defined as a change > threshold from the previous period.

    Returns:
        DataFrame of spike events with columns:
        timestamp, compound, direction, magnitude, spike_type
    """
    if timeline.empty or len(timeline) < 2:
        return pd.DataFrame()

    tl = timeline.copy()
    tl["delta"] = tl["avg_compound"].diff().abs()
    tl["direction"] = tl["avg_compound"].diff().apply(
        lambda x: "📈 Positive Surge" if x > 0 else "📉 Negative Drop"
    )

    spikes = tl[tl["delta"] >= threshold].copy()
    spikes["magnitude"] = spikes["delta"].round(3)
    spikes["spike_type"] = spikes.apply(
        lambda r: "Goal / Big Win" if r["avg_compound"] > 0.3
        else "Red Card / Controversy" if r["avg_compound"] < -0.3
        else "Mixed Reaction", axis=1
    )

    return spikes[["timestamp", "avg_compound", "direction",
                   "magnitude", "spike_type"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TEAM & PLAYER AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_team_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment by team.

    Returns:
        DataFrame with columns: team, avg_compound, article_count,
                                 pos_pct, neg_pct, sentiment_label
    """
    if df.empty or "team" not in df.columns:
        return pd.DataFrame()

    grp = df.groupby("team").agg(
        avg_compound  = ("compound",  "mean"),
        article_count = ("compound",  "count"),
        pos_count     = ("label",     lambda x: (x == "Positive").sum()),
        neg_count     = ("label",     lambda x: (x == "Negative").sum()),
    ).reset_index()

    grp["pos_pct"] = grp["pos_count"] / grp["article_count"]
    grp["neg_pct"] = grp["neg_count"] / grp["article_count"]
    grp["sentiment_label"] = grp["avg_compound"].apply(
        lambda c: "Positive" if c >= 0.05 else "Negative" if c <= -0.05 else "Neutral"
    )
    grp = grp.drop(columns=["pos_count", "neg_count"])
    return grp.sort_values("article_count", ascending=False).reset_index(drop=True)


def aggregate_player_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment by player.

    Returns:
        DataFrame with player sentiment rankings.
    """
    if df.empty or "player" not in df.columns:
        return pd.DataFrame()

    pdf = df[df["player"] != "General"]
    if pdf.empty:
        return pd.DataFrame()

    grp = pdf.groupby("player").agg(
        avg_compound  = ("compound",  "mean"),
        article_count = ("compound",  "count"),
        pos_pct       = ("label",     lambda x: (x == "Positive").mean()),
        neg_pct       = ("label",     lambda x: (x == "Negative").mean()),
    ).reset_index()

    grp["sentiment_label"] = grp["avg_compound"].apply(
        lambda c: "⭐ Fan Favourite" if c >= 0.15
        else "⚠️ Under Pressure" if c <= -0.1
        else "📊 Mixed"
    )
    return grp.sort_values("article_count", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRENDING TOPICS
# ─────────────────────────────────────────────────────────────────────────────

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "that","this","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","dare","ought","used","it","its","their","they",
    "he","she","we","i","you","his","her","our","your","my","who","which",
    "from","as","by","about","into","through","after","before","during",
    "match","game","team","club","player","football","soccer","season",
}

def extract_trending_topics(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Extract trending keywords from article titles.

    Returns:
        DataFrame with columns: topic, mentions, avg_sentiment, trend_score
    """
    if df.empty or "title" not in df.columns:
        return pd.DataFrame()

    word_sentiments: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        title = str(row.get("title", "")).lower()
        words = re.findall(r"\b[a-z]{4,}\b", title)
        for word in words:
            if word not in STOP_WORDS:
                word_sentiments.setdefault(word, []).append(row.get("compound", 0))

    rows = []
    for word, sentiments in word_sentiments.items():
        count = len(sentiments)
        avg_s = float(np.mean(sentiments))
        # trend_score combines frequency + recency boost
        trend_score = count * (1 + abs(avg_s))
        rows.append({
            "topic":         word.title(),
            "mentions":      count,
            "avg_sentiment": round(avg_s, 3),
            "trend_score":   round(trend_score, 2),
        })

    if not rows:
        return pd.DataFrame()

    result = (pd.DataFrame(rows)
              .sort_values("trend_score", ascending=False)
              .head(top_n)
              .reset_index(drop=True))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_COLS = ["excitement", "celebration", "frustration",
                "anger", "disappointment", "shock"]

def build_emotion_timeline(df: pd.DataFrame,
                            freq: str = "1h") -> pd.DataFrame:
    """
    Build a time-series of average emotion scores per bucket.

    Returns:
        DataFrame with timestamp + one column per emotion.
    """
    if df.empty:
        return pd.DataFrame()

    freq = freq.strip().lower()

    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df = df.set_index("published_at").sort_index()

    avail = [c for c in EMOTION_COLS if c in df.columns]
    if not avail:
        return pd.DataFrame()

    agg = df[avail].resample(freq).mean().reset_index()
    agg.columns = ["timestamp"] + avail
    return agg.dropna(how="all", subset=avail)


def compute_live_stats(df: pd.DataFrame) -> dict:
    """
    Compute headline KPIs for the live dashboard.

    Returns:
        dict with keys: overall_sentiment, pos_pct, neg_pct, neu_pct,
                        total_articles, dominant_emotion,
                        most_mentioned_team, most_mentioned_player,
                        last_updated
    """
    if df.empty:
        return {
            "overall_sentiment": 0.0, "pos_pct": 33, "neg_pct": 33,
            "neu_pct": 34, "total_articles": 0,
            "dominant_emotion": "None", "most_mentioned_team": "—",
            "most_mentioned_player": "—",
            "last_updated": datetime.utcnow().strftime("%H:%M UTC"),
        }

    total = len(df)
    pos   = int((df["label"] == "Positive").sum() / total * 100)
    neg   = int((df["label"] == "Negative").sum() / total * 100)
    neu   = 100 - pos - neg

    emotion_avgs = {c: df[c].mean() for c in EMOTION_COLS if c in df.columns}
    dominant = max(emotion_avgs, key=emotion_avgs.get) if emotion_avgs else "neutral"

    team_counts  = df[df["team"]   != "General"]["team"].value_counts()
    plyr_counts  = df[df["player"] != "General"]["player"].value_counts()

    return {
        "overall_sentiment":   round(float(df["compound"].mean()), 3),
        "pos_pct":             pos,
        "neg_pct":             neg,
        "neu_pct":             neu,
        "total_articles":      total,
        "dominant_emotion":    dominant.title(),
        "most_mentioned_team": team_counts.index[0] if not team_counts.empty else "—",
        "most_mentioned_player": plyr_counts.index[0] if not plyr_counts.empty else "—",
        "last_updated":        datetime.utcnow().strftime("%H:%M UTC"),
    }
