"""
CopaVision AI — Phase 3
scripts/news_collector.py

Live football news collector using:
  - The Guardian API (free, no approval needed)
  - NewsAPI (free tier, 100 requests/day)

Both APIs are stable, reliable, and deployable on Hugging Face Spaces.

Usage:
    collector = NewsCollector(guardian_key="...", newsapi_key="...")
    articles  = collector.collect_all(query="Premier League", max_per_source=20)
"""

import os
import re
import time
import hashlib
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)


# ─── Football keyword sets ────────────────────────────────────────────────────
FOOTBALL_TEAMS = [
    "Manchester City", "Manchester United", "Liverpool", "Arsenal",
    "Chelsea", "Tottenham", "Barcelona", "Real Madrid", "Bayern Munich",
    "PSG", "Juventus", "Inter Milan", "AC Milan", "Borussia Dortmund",
    "Atletico Madrid", "Ajax", "Porto", "Benfica", "Celtic", "Rangers",
    "Brazil", "Argentina", "France", "England", "Germany", "Spain",
    "Italy", "Portugal", "Netherlands", "Belgium",
]

FOOTBALL_PLAYERS = [
    "Messi", "Ronaldo", "Mbappe", "Haaland", "Neymar", "Salah",
    "De Bruyne", "Bellingham", "Vinicius", "Pedri", "Gavi",
    "Kane", "Rashford", "Saka", "Odegaard", "Lewandowski",
    "Benzema", "Modric", "Kroos", "Alisson", "Courtois",
]

SENTIMENT_KEYWORDS = {
    "attacking": ["goal", "scored", "hat-trick", "brace", "penalty",
                  "free kick", "header", "volley", "assist", "strike"],
    "defending": ["clean sheet", "save", "tackle", "interception",
                  "clearance", "block", "offside trap"],
    "negative":  ["red card", "injury", "suspended", "banned", "crisis",
                  "defeat", "relegated", "sacked", "controversy"],
    "positive":  ["win", "victory", "champion", "title", "trophy",
                  "unbeaten", "record", "comeback", "promoted"],
}


def _article_id(url: str) -> str:
    """Generate a stable dedup key from URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


class GuardianCollector:
    """
    Collects football articles from The Guardian's free API.
    Sign up at: https://open-platform.theguardian.com/access/
    """

    BASE_URL = "https://content.guardianapis.com/search"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._seen: set[str] = set()

    def fetch(self, query: str,
              days_back: int = 3,
              max_results: int = 30) -> list[dict]:
        """
        Fetch football articles from The Guardian.

        Args:
            query:       Search query (e.g. 'Premier League goals')
            days_back:   How many days back to search
            max_results: Maximum articles to return

        Returns:
            List of article dicts with keys:
            id, title, body, url, published_at, source, query
        """
        from_date = (datetime.utcnow() - timedelta(days=days_back)
                     ).strftime("%Y-%m-%d")

        params = {
            "q":           query,
            "section":     "football",
            "from-date":   from_date,
            "api-key":     self.api_key,
            "show-fields": "bodyText,trailText",
            "page-size":   min(max_results, 50),
            "order-by":    "newest",
        }

        articles = []
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("response", {}).get("results", []):
                url = item.get("webUrl", "")
                aid = _article_id(url)
                if aid in self._seen:
                    continue
                self._seen.add(aid)

                fields = item.get("fields", {})
                body   = fields.get("bodyText", "") or \
                         fields.get("trailText", "")

                articles.append({
                    "id":           aid,
                    "title":        item.get("webTitle", ""),
                    "body":         body[:1000],   # cap at 1000 chars
                    "url":          url,
                    "published_at": item.get("webPublicationDate", ""),
                    "source":       "The Guardian",
                    "query":        query,
                })

        except requests.RequestException as e:
            log.warning(f"Guardian API error for '{query}': {e}")

        return articles


class NewsAPICollector:
    """
    Collects football articles from NewsAPI.org (free tier).
    Sign up at: https://newsapi.org/register
    Free tier: 100 requests/day, articles up to 1 month old.
    """

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._seen: set[str] = set()

    def fetch(self, query: str,
              days_back: int = 2,
              max_results: int = 20) -> list[dict]:
        """
        Fetch football articles from NewsAPI.

        Returns:
            List of article dicts (same schema as GuardianCollector)
        """
        from_date = (datetime.utcnow() - timedelta(days=days_back)
                     ).strftime("%Y-%m-%dT%H:%M:%S")

        params = {
            "q":        f"{query} football",
            "from":     from_date,
            "language": "en",
            "sortBy":   "publishedAt",
            "apiKey":   self.api_key,
            "pageSize": min(max_results, 100),
        }

        articles = []
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("articles", []):
                url = item.get("url", "")
                aid = _article_id(url)
                if aid in self._seen:
                    continue
                self._seen.add(aid)

                articles.append({
                    "id":           aid,
                    "title":        item.get("title", ""),
                    "body":         (item.get("description") or "") + " " +
                                    (item.get("content") or ""),
                    "url":          url,
                    "published_at": item.get("publishedAt", ""),
                    "source":       item.get("source", {}).get("name", "NewsAPI"),
                    "query":        query,
                })

        except requests.RequestException as e:
            log.warning(f"NewsAPI error for '{query}': {e}")

        return articles


class NewsCollector:
    """
    Unified collector that aggregates from Guardian + NewsAPI.
    Falls back gracefully if either API key is missing or fails.
    """

    # Default football queries to track
    DEFAULT_QUERIES = [
        "Premier League",
        "Champions League",
        "La Liga",
        "World Cup football",
        "football transfer",
        "Messi Ronaldo Mbappe Haaland",
        "football goal injury",
    ]

    def __init__(self,
                 guardian_key: Optional[str] = None,
                 newsapi_key: Optional[str] = None):
        self.guardian = GuardianCollector(guardian_key) \
                        if guardian_key else None
        self.newsapi  = NewsAPICollector(newsapi_key) \
                        if newsapi_key else None
        self._global_seen: set[str] = set()

    def collect_all(self,
                    queries: Optional[list[str]] = None,
                    max_per_source: int = 20,
                    days_back: int = 3) -> list[dict]:
        """
        Run all queries across all available sources.

        Returns:
            Deduplicated list of article dicts, newest first.
        """
        queries = queries or self.DEFAULT_QUERIES
        all_articles = []

        for query in queries:
            if self.guardian:
                arts = self.guardian.fetch(query, days_back, max_per_source)
                for a in arts:
                    if a["id"] not in self._global_seen:
                        self._global_seen.add(a["id"])
                        all_articles.append(a)
                time.sleep(0.3)   # polite delay

            if self.newsapi:
                arts = self.newsapi.fetch(query, days_back, max_per_source)
                for a in arts:
                    if a["id"] not in self._global_seen:
                        self._global_seen.add(a["id"])
                        all_articles.append(a)
                time.sleep(0.3)

        # Sort newest first
        all_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return all_articles

    def collect_demo(self, n: int = 80) -> list[dict]:
        """
        Generate realistic demo data when no API keys are configured.
        Used for UI testing and Hugging Face demo deployment.
        """
        import random
        from datetime import timezone

        headlines = [
            ("Mbappe scores stunning hat-trick as Real Madrid crush Barcelona 4-1",
             "real madrid barcelona laliga"),
            ("Premier League title race: Arsenal close gap on Manchester City with vital win",
             "arsenal manchester city premier league"),
            ("Champions League: Liverpool's incredible comeback stuns Bayern Munich",
             "liverpool bayern champions league"),
            ("Haaland breaks another scoring record — 50 goals in a single season",
             "haaland manchester city goals record"),
            ("Controversy as VAR decision denies Chelsea late equaliser",
             "chelsea var controversy referee"),
            ("Messi leads Argentina to World Cup glory in unforgettable final",
             "messi argentina world cup champion"),
            ("Ronaldo scores free kick winner as Al Nassr reach Asian Champions League final",
             "ronaldo al nassr goals"),
            ("Vinicius Junior wins Ballon d'Or after magical Champions League campaign",
             "vinicius real madrid ballon dor"),
            ("Injury blow for England as Bellingham ruled out for six weeks",
             "bellingham england injury"),
            ("Saka signs new long-term contract with Arsenal amid transfer interest",
             "saka arsenal transfer contract"),
            ("Referee controversy mars El Clasico — Barcelona furious after red card",
             "barcelona real madrid red card controversy referee"),
            ("De Bruyne masterclass guides Manchester City to title triumph",
             "de bruyne manchester city premier league title"),
            ("PSG struggle without Mbappe as Ligue 1 title slips away",
             "psg mbappe ligue1 struggle"),
            ("Tactical breakdown: How Guardiola's pressing system dominates Europe",
             "guardiola manchester city tactics pressing"),
            ("Transfer window: Haaland to Real Madrid — the move that shook football",
             "haaland real madrid transfer record"),
            ("Penalty shootout heartbreak as England exit World Cup quarter-finals",
             "england world cup penalty shootout disappointed"),
            ("Sensational Salah brace rescues Liverpool in Champions League thriller",
             "salah liverpool champions league goal"),
            ("Rising star Pedri shines in Barcelona's 3-0 demolition of Atletico",
             "pedri barcelona atletico goals"),
            ("Goalkeeper Alisson saves penalty to keep Liverpool's title hopes alive",
             "alisson liverpool penalty save"),
            ("Goal of the season contender: Rashford's 40-yard wonder strike stuns crowd",
             "rashford manchester united goal wonder strike"),
        ]

        now = datetime.now(timezone.utc)
        articles = []
        for i in range(n):
            h, q = random.choice(headlines)
            mins_ago = random.randint(5, 60 * 24 * days_back)
            pub = now - timedelta(minutes=mins_ago)
            articles.append({
                "id":           f"demo_{i:04d}",
                "title":        h,
                "body":         h + ". " + h.lower().replace(".", ""),
                "url":          f"https://example.com/article/{i}",
                "published_at": pub.isoformat(),
                "source":       random.choice(["The Guardian", "BBC Sport",
                                               "Sky Sports", "ESPN FC"]),
                "query":        q,
            })

        days_back = 3
        articles.sort(key=lambda x: x["published_at"], reverse=True)
        return articles
