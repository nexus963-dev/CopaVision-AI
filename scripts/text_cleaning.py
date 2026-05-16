"""
CopaVision AI — Phase 3
scripts/text_cleaning.py

Football-oriented text preprocessing pipeline.
Cleans news article text and headlines for sentiment analysis.
"""

import re
import string


# ─── Football slang normalization map ────────────────────────────────────────
FOOTBALL_SLANG = {
    "pen":       "penalty",
    "pens":      "penalties",
    "gk":        "goalkeeper",
    "cb":        "centre back",
    "rb":        "right back",
    "lb":        "left back",
    "cm":        "midfielder",
    "cf":        "striker",
    "lw":        "winger",
    "rw":        "winger",
    "brace":     "two goals",
    "treble":    "hat trick",
    "ol":        "own goal",
    "og":        "own goal",
    "ucl":       "champions league",
    "epl":       "premier league",
    "pl":        "premier league",
    "cl":        "champions league",
    "wc":        "world cup",
    "var":       "video assistant referee",
    "motm":      "man of the match",
    "pog":       "player of the game",
    "utd":       "united",
    "fc":        "football club",
    "afc":       "football club",
    "cfc":       "football club",
    "vs":        "versus",
    "v":         "versus",
    "ft":        "full time",
    "ht":        "half time",
    "et":        "extra time",
    "aet":       "after extra time",
}

# ─── Noise patterns to remove ────────────────────────────────────────────────
URL_PATTERN      = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN     = re.compile(r"<[^>]+>")
MENTION_PATTERN  = re.compile(r"@\w+")
HASHTAG_PATTERN  = re.compile(r"#(\w+)")   # keep word, remove #
REPEAT_PATTERN   = re.compile(r"(.)\1{2,}")  # e.g. "goooal" → "goal"
WHITESPACE       = re.compile(r"\s+")
PUNCT_ONLY       = re.compile(r"^[^\w]+$")


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)

def remove_html(text: str) -> str:
    return HTML_PATTERN.sub(" ", text)

def remove_mentions(text: str) -> str:
    return MENTION_PATTERN.sub(" ", text)

def normalize_hashtags(text: str) -> str:
    """#WorldCup → WorldCup (keeps the word, removes the symbol)"""
    return HASHTAG_PATTERN.sub(r"\1", text)

def normalize_repeats(text: str) -> str:
    """goooooal → goal, yesss → yes"""
    return REPEAT_PATTERN.sub(r"\1\1", text)

def normalize_slang(text: str) -> str:
    """Replace common football abbreviations with full words."""
    words = text.lower().split()
    return " ".join(FOOTBALL_SLANG.get(w, w) for w in words)

def remove_punctuation_noise(text: str) -> str:
    """Keep alphanumeric and basic punctuation, remove symbol noise."""
    return re.sub(r"[^\w\s\.\,\!\?\-\']", " ", text)

def normalize_whitespace(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def clean_text(text: str, slang: bool = True) -> str:
    """
    Full cleaning pipeline for a single piece of text.
    
    Args:
        text:  Raw text string
        slang: Whether to normalize football slang
    
    Returns:
        Cleaned text string ready for sentiment analysis
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = remove_html(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_hashtags(text)
    text = normalize_repeats(text)
    text = remove_punctuation_noise(text)

    if slang:
        text = normalize_slang(text)

    text = normalize_whitespace(text)
    return text


def extract_entities(text: str,
                     teams: list[str],
                     players: list[str]) -> dict:
    """
    Extract mentioned football entities from cleaned text.

    Returns:
        dict with keys 'teams' and 'players' — lists of matches found
    """
    lower = text.lower()
    found_teams   = [t for t in teams   if t.lower() in lower]
    found_players = [p for p in players if p.lower() in lower]
    return {"teams": found_teams, "players": found_players}


def is_football_relevant(text: str,
                         threshold: int = 1) -> bool:
    """
    Quick filter: is this text actually about football?
    Returns True if at least `threshold` football keywords are found.
    """
    FOOTBALL_KEYWORDS = {
        "football", "soccer", "goal", "match", "player", "club", "league",
        "premier", "champions", "world cup", "transfer", "manager", "coach",
        "stadium", "referee", "penalty", "offside", "tackle", "dribble",
        "striker", "midfielder", "defender", "goalkeeper", "winger",
        "messi", "ronaldo", "mbappe", "haaland", "neymar",
        "barcelona", "madrid", "manchester", "liverpool", "arsenal",
        "chelsea", "juventus", "psg", "bayern", "dortmund",
    }
    lower = text.lower()
    hits = sum(1 for kw in FOOTBALL_KEYWORDS if kw in lower)
    return hits >= threshold
