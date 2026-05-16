"""
CopaVision AI — Phase 3
scripts/sentiment_pipeline.py

Sentiment analysis engine using VADER (Valence Aware Dictionary
and sEntiment Reasoner) — optimised for short social/news text.

VADER is ideal because:
  - No GPU needed (runs on Hugging Face free CPU tier)
  - Specifically built for social media + news language
  - Handles punctuation, caps, and emojis naturally
  - Instant inference (no model loading delay)
  - 100% offline after install

Install: pip install vaderSentiment
"""

import re
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ── Lazy-load VADER so import doesn't fail if not installed ──────────────────
_VADER = None

def _get_vader():
    global _VADER
    if _VADER is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _VADER = SentimentIntensityAnalyzer()
            log.info("VADER loaded successfully")
        except ImportError:
            log.warning("vaderSentiment not installed — using fallback scorer")
            _VADER = _FallbackScorer()
    return _VADER


class _FallbackScorer:
    """
    Pure-Python fallback scorer when VADER is not installed.
    Good enough for demo/testing; install vaderSentiment for production.
    """
    POS = {"goal","brilliant","incredible","amazing","superb","winner",
           "champion","glory","fantastic","great","excellent","magnificent",
           "hat-trick","magic","perfect","stunning","wonderful","hero",
           "celebration","victory","win","scored","beautiful","masterclass",
           "brace","comeback","unbeaten","record","promoted","title","trophy",
           "save","clean sheet","assist","clinical","dominant"}
    NEG = {"terrible","awful","disaster","miss","injury","red card","penalty",
           "relegated","defeated","lost","poor","weak","disappointing",
           "frustrating","shocking","worst","mistake","failure","crisis",
           "suspended","banned","controversy","riot","angry","protest",
           "sacked","heartbreak","foul","offside","controversial","blunder"}

    def polarity_scores(self, text: str) -> dict:
        import math
        t = text.lower()
        words = set(re.findall(r"\b\w+\b", t))
        pos = len(words & self.POS)
        neg = len(words & self.NEG)
        total = pos + neg + 0.001
        compound = (pos - neg) / math.sqrt(total + 4)
        compound = max(-1.0, min(1.0, compound))
        return {"compound": compound,
                "pos": pos / total,
                "neg": neg / total,
                "neu": max(0.0, 1.0 - pos / total - neg / total)}


# ── Football-specific emotion keyword lexicon ────────────────────────────────
EMOTION_LEXICON = {
    "excitement": {
        "goal", "incredible", "brilliant", "amazing", "unbelievable",
        "stunning", "spectacular", "wow", "hat trick", "brace", "magic",
        "masterclass", "world class", "screamer", "what a goal",
    },
    "celebration": {
        "champion", "title", "trophy", "glory", "winner", "victory",
        "triumph", "promoted", "unbeaten", "record", "legendary", "historic",
    },
    "frustration": {
        "poor", "weak", "disappointing", "terrible", "disgrace", "awful",
        "pathetic", "wasting", "miss", "chance", "offside", "frustrating",
    },
    "anger": {
        "red card", "referee", "var", "controversial", "foul", "penalty",
        "denied", "robbery", "disgraceful", "outrageous", "scandal", "cheating",
    },
    "disappointment": {
        "defeat", "loss", "relegated", "injury", "suspended", "miss",
        "heartbreak", "penalty miss", "shootout", "exit", "knocked out",
    },
    "shock": {
        "shock", "surprise", "unexpected", "unbelievable", "transfer",
        "sacked", "resigned", "crisis", "stunning", "sudden",
    },
}


def score_sentiment(text: str) -> dict:
    """
    Score a single piece of text for sentiment and emotion.

    Returns:
        dict with keys:
            compound    — overall score -1 to +1
            label       — 'Positive' | 'Negative' | 'Neutral'
            pos / neg / neu — component scores
            emotions    — dict of emotion → score (0–1)
            dominant_emotion — the strongest emotion found
    """
    if not text or not isinstance(text, str):
        return _empty_score()

    vader = _get_vader()
    scores = vader.polarity_scores(text)
    compound = scores["compound"]

    label = ("Positive" if compound >= 0.05
             else "Negative" if compound <= -0.05
             else "Neutral")

    # Emotion scoring
    lower = text.lower()
    emotions = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        hits = sum(1 for kw in keywords if kw in lower)
        emotions[emotion] = min(1.0, hits * 0.4)   # normalise

    dominant = max(emotions, key=emotions.get) if any(emotions.values()) else "neutral"

    return {
        "compound":        round(compound, 4),
        "label":           label,
        "pos":             round(scores["pos"], 3),
        "neg":             round(scores["neg"], 3),
        "neu":             round(scores["neu"], 3),
        "emotions":        emotions,
        "dominant_emotion": dominant,
    }


def _empty_score() -> dict:
    return {
        "compound": 0.0, "label": "Neutral",
        "pos": 0.0, "neg": 0.0, "neu": 1.0,
        "emotions": {k: 0.0 for k in EMOTION_LEXICON},
        "dominant_emotion": "neutral",
    }


def score_articles(articles: list[dict]) -> pd.DataFrame:
    """
    Score a list of article dicts and return a flat DataFrame.

    Input article keys: id, title, body, published_at, source, query
    Output columns add: compound, label, pos, neg, dominant_emotion,
                        excitement, celebration, frustration, anger,
                        disappointment, shock, team, player
    """
    rows = []
    for art in articles:
        text  = (art.get("title", "") + " " + art.get("body", "")).strip()
        score = score_sentiment(text)

        # Entity extraction
        team   = _extract_entity(text, _TEAMS)
        player = _extract_entity(text, _PLAYERS)

        # Parse timestamp
        raw_ts = art.get("published_at", "")
        try:
            ts = pd.to_datetime(raw_ts, utc=True)
        except Exception:
            ts = pd.Timestamp.now(tz="UTC")

        rows.append({
            "id":               art.get("id", ""),
            "title":            art.get("title", "")[:120],
            "source":           art.get("source", ""),
            "query":            art.get("query", ""),
            "published_at":     ts,
            "compound":         score["compound"],
            "label":            score["label"],
            "pos":              score["pos"],
            "neg":              score["neg"],
            "neu":              score["neu"],
            "dominant_emotion": score["dominant_emotion"],
            "excitement":       score["emotions"]["excitement"],
            "celebration":      score["emotions"]["celebration"],
            "frustration":      score["emotions"]["frustration"],
            "anger":            score["emotions"]["anger"],
            "disappointment":   score["emotions"]["disappointment"],
            "shock":            score["emotions"]["shock"],
            "team":             team,
            "player":           player,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df


def _extract_entity(text: str, entity_list: list[str]) -> str:
    lower = text.lower()
    for entity in entity_list:
        if entity.lower() in lower:
            return entity
    return "General"


# Entity lists for extraction
_TEAMS = [
    "Real Madrid", "Barcelona", "Manchester City", "Manchester United",
    "Liverpool", "Arsenal", "Chelsea", "Tottenham", "PSG", "Bayern Munich",
    "Juventus", "Inter Milan", "AC Milan", "Borussia Dortmund",
    "Atletico Madrid", "Ajax", "Brazil", "Argentina", "France", "England",
    "Germany", "Spain", "Italy", "Portugal",
]

_PLAYERS = [
    "Messi", "Ronaldo", "Mbappe", "Haaland", "Neymar", "Salah",
    "De Bruyne", "Bellingham", "Vinicius", "Pedri", "Gavi",
    "Kane", "Rashford", "Saka", "Odegaard", "Lewandowski",
    "Benzema", "Modric", "Alisson", "Courtois", "Valverde",
]
