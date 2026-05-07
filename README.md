# ⚽ CopaVision AI — Phase 1: Match Outcome Predictor

A production-grade football analytics dashboard built with Streamlit, scikit-learn, and Plotly.

## 🚀 Quick Start

```bash
# 1. Clone / download the project
git clone https://github.com/YOUR_USERNAME/copavision-ai.git
cd copavision-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure your file structure looks like this:
#   app.py
#   requirements.txt
#   team_stats.json
#   models/
#       copavision_rf.pkl
#       copavision_lr.pkl

# 4. Run
streamlit run app.py
```

## 📁 File Structure

```
copavision-ai/
│
├── app.py                  ← Main Streamlit dashboard
├── requirements.txt
├── team_stats.json         ← Pre-computed Elo + form data (298 teams)
├── README.md
│
├── models/
│   ├── copavision_rf.pkl   ← Trained Random Forest
│   └── copavision_lr.pkl   ← Trained Logistic Regression
│
├── data/
│   └── results.csv         ← Raw historical match data
│
└── copavision_phase1.py    ← Phase 1 training pipeline (run this first)
```

## ☁️ Deploy on Streamlit Cloud

1. Push this folder to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New App** → select your repo, branch (`main`), and set main file to `app.py`
4. Click **Deploy** — live in ~2 minutes

> **Note:** The `.pkl` model files and `team_stats.json` must be committed to GitHub (they're small enough). Add them with `git add models/ team_stats.json`.

## 🗺️ Phase Roadmap

| Phase | Module | Status |
|---|---|---|
| 1 | Match Outcome Predictor | ✅ Complete |
| 2 | Player Dashboard | 🔲 Planned |
| 3 | Sentiment Tracker | 🔲 Planned |
| 4 | Live Match Feed | 🔲 Planned |

## 🔌 Connecting Future Modules

Streamlit's **multi-page** feature makes adding modules trivial:

```
copavision-ai/
├── app.py                  ← Home / Match Predictor
└── pages/
    ├── 02_Player_Dashboard.py    ← Phase 2
    ├── 03_Sentiment_Tracker.py   ← Phase 3
    └── 04_Live_Feed.py           ← Phase 4
```

Each file in `pages/` automatically appears in the sidebar navigation.

## 📊 Model Performance

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 57.7% | 0.433 |
| Random Forest | 54.5% | 0.510 |

Trained on 16,113 matches (2000–2016). Tested on 3,025 matches (2017–2020).
