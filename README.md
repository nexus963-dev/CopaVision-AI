# CopaVision AI

AI-powered football analytics platform built with Streamlit, scikit-learn, Plotly, and StatsBomb-derived player datasets.

## Current Modules

| Phase | Module | Status |
|---|---|---|
| 1 | Match Outcome Predictor | Complete |
| 2 | Player Performance Intelligence Dashboard | Complete |
| 3 | Sentiment Tracker | Planned |
| 4 | Live Match Feed | Planned |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit automatically discovers the Phase 2 module from the `pages/` directory.

## Folder Structure

```text
copavision-ai/
├── app.py
├── copavision_phase1.py
├── team_stats.json
├── requirements.txt
├── README.md
├── models/
│   ├── copavision_rf.pkl
│   └── copavision_lr.pkl
├── data/
│   ├── results.csv
│   ├── player_stats.csv
│   ├── player_per90.csv
│   ├── similarity_features.csv
│   └── clustered_player_features.csv
└── pages/
    └── 02_Player_Dashboard.py
```

## Phase 1: Match Outcome Predictor

The main `app.py` module predicts match outcomes using precomputed Elo/form features and trained scikit-learn models.

Expected assets:

- `models/copavision_rf.pkl`
- `models/copavision_lr.pkl`
- `team_stats.json`

## Phase 2: Player Intelligence Dashboard

The `pages/02_Player_Dashboard.py` module uses only processed CSV files. Raw StatsBomb JSON files are not required in the app.

Features:

- Player search and profile dashboard
- Filters for position, competition, team, nationality, and age when available
- Percentile rankings and strengths/weaknesses
- Side-by-side player comparison
- Radar charts and percentile comparison charts
- KMeans player archetype explorer with PCA and t-SNE projections
- Similar player engine powered by cosine similarity

Required data files:

- `data/player_stats.csv`
- `data/player_per90.csv`
- `data/similarity_features.csv`
- `data/clustered_player_features.csv`

## Deployment

For Streamlit Cloud:

1. Commit `app.py`, `pages/`, `data/`, `models/`, `team_stats.json`, `requirements.txt`, and `README.md`.
2. Create a Streamlit Cloud app with `app.py` as the entry file.
3. Make sure all four Phase 2 CSVs are present in `data/`.
4. Streamlit will show the Player Dashboard automatically in the sidebar.

## Extending the Platform

Phase 3 can plug into the same architecture as another Streamlit page, for example:

```text
pages/
├── 02_Player_Dashboard.py
├── 03_Sentiment_Tracker.py
└── 04_Live_Feed.py
```

Good future upgrades:

- Position-specific percentile models
- Transfer shortlist scoring
- Player development trajectory views across seasons
- Team fit recommendation engine
- Injury and availability context
- Sentiment/news integration for Phase 3
