# 🏀 Data Dunk

**A full-stack machine learning web app that predicts NBA game outcomes and player stats — and explains *why*, using SHAP.**

Most sports prediction tools give you a number and nothing else. Data Dunk shows a win probability, then breaks down exactly which stats drove that prediction, in plain English and interactive charts. 

---

## 🎥 Demo

A full walkthrough is available in `DATA_DUNK-DEMO.mp4` in this repo. GitHub doesn't render local video files inline in a README — to embed it as a playable clip, open a new GitHub Issue in your repo, drag the `.mp4` into the comment box, copy the `https://github.com/user-attachments/...` link it generates, then paste that link here (or as a plain link: `[▶ Watch the demo](your-link-here)`). Cancel the issue afterwards without submitting — the uploaded asset link still works.

---

## 🎯 What it does

- **Game Prediction** — pick any two of the 30 NBA teams, get a win probability, predicted final score, and each team's key player, from two competing models (XGBoost and Logistic Regression) that you can switch between or compare side by side.
- **Game Insights (Explainable AI)** — a dedicated page powered by **SHAP** that shows *why* the model favours a team: a feature-importance bar chart, a prediction waterfall, and a full breakdown table with plain-English descriptions like *"BOS's superior net rating strongly increases their win probability."*
- **Player Lab** — head-to-head player comparison with radar charts, and a "Teammate Chemistry" tool that scores how well two players on the same team fit together (synergy score, play style, badges).

<img width="1440" height="900" alt="predictions_home" src="https://github.com/user-attachments/assets/6fdaa547-1454-4c01-ae8c-ae54ac0348ca" />


---

## 🧠 Why it's interesting

Anyone can train a classifier and print an accuracy score. The point of this project was to make the model **transparent**:

- Two models trained side by side (Logistic Regression vs. XGBoost) so you can compare *how* they disagree, not just their headline accuracy.
- Every prediction ships with its own SHAP explanation, computed live via a `TreeExplainer`, not just a static global importance chart.
- The system is honest about its limits — the report documents where XGBoost overfits, where player-stat prediction is genuinely hard (~15% R², because per-game scoring is noisy), and why Logistic Regression actually wins on generalisation for the game-outcome task.

<img width="1440" height="900" alt="game_prediction_result" src="https://github.com/user-attachments/assets/914d289b-d1c1-482f-97d5-2b9a6cc004b9" />


---

## 🔍 Game Insights — the SHAP explainability page

This is the core feature. For any matchup, the app shows a confidence meter, a feature-importance chart (which stats pushed the prediction toward the home or away team), a prediction waterfall, and global feature importance across the whole training set.

<img width="585" height="811" alt="game_insights_shap_top" src="https://github.com/user-attachments/assets/d738dbcf-8008-43cc-bdb7-f43b31c616cd" />


Below that, a full feature breakdown table gives the raw stat value, its SHAP impact, and a plain-English explanation for every feature — so a user with zero ML background can still follow the reasoning.

<img width="601" height="625" alt="game_insights_feature_breakdown" src="https://github.com/user-attachments/assets/2f592bb7-c25d-4a17-b036-6b5fbd092d2d" />


---

## ⚡ Player Lab

**Head-to-head comparison** — pick any two players and compare them across five dimensions (scoring, rebounding, playmaking, endurance, usage) on an overlaid radar chart, with a category-by-category verdict.

<img width="570" height="618" alt="player_lab_head_to_head" src="https://github.com/user-attachments/assets/29c470f7-92c1-4211-92cf-baa84e37b7ab" />


**Teammate chemistry** — pick a team and see how its two key players' styles combine: a synergy score (role complementarity, usage balance, playmaker/finisher fit), combined stats, and descriptive badges like *"Elite Scoring"* or *"Playmaking Duo."*

<img width="459" height="658" alt="player_lab_teammate_chemistry" src="https://github.com/user-attachments/assets/affdabb8-3b7c-4b10-98c6-54b88ddf7388" />


---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| **Frontend** | HTML, CSS, TypeScript, Chart.js |
| **Backend** | Python, FastAPI, Uvicorn |
| **ML** | scikit-learn (Logistic Regression, Linear Regression), XGBoost |
| **Explainability** | SHAP (TreeExplainer) |
| **Data** | Synthetic NBA data (6 seasons, 30 teams, 60 players) generated with pandas/NumPy |
| **Testing** | pytest, httpx (unit + integration), Locust (load testing) |
| **Deployment** | Docker, Render.com |

### Why synthetic data?
Rather than depending on a live API that could change, rate-limit, or disappear, I generated six full seasons (2020–21 to 2025–26) of realistic NBA statistics myself — win percentages, offensive/defensive ratings, pace, and player stats — with year-to-year drift and per-game noise. This keeps the whole pipeline **fully reproducible**: anyone who clones the repo gets identical data, identical models, and identical results.

---

## 📊 Model Performance

**Game outcome (win/loss), 5-fold cross-validation:**

| Model | Train Acc | Test Acc | CV Accuracy |
|---|---|---|---|
| Logistic Regression | 91.8% | 92.0% | **90.9%** (±2.8%) |
| XGBoost Classifier | 100.0% | 89.1% | 87.6% (±2.2%) |

Logistic Regression generalises better here — XGBoost's 100% train accuracy vs. its lower CV score is a textbook overfitting signal, which the report discusses and partially corrects for with regularisation (`max_depth`, `min_child_weight`, `reg_alpha`, `subsample`).

**Player points prediction:**

| Model | Train R² | Test R² |
|---|---|---|
| Linear Regression | 12.0% | 12.2% |
| XGBoost Regressor | 21.7% | **14.6%** |

Low R² here isn't a bug — predicting a single player's point total in one specific game is genuinely hard; foul trouble, game flow, and shooting variance dominate. The report treats this as an honest finding rather than hiding it.

---

## 🏗️ Architecture

Three-tier design: a static TypeScript/HTML frontend talks to a FastAPI backend over REST, which loads pre-trained models and CSV data into memory at startup.

```
Browser (TS/HTML/Chart.js)
        │  fetch() → JSON
        ▼
FastAPI backend (main.py)
├── /predict-game        → win probability, score, key players
├── /explain-game        → SHAP values + plain-English explanations
├── /predict-player       → player stat prediction (opponent-aware)
├── /compare-players      → head-to-head radar data
├── /team-chemistry       → synergy score + badges
├── /team-trend/{team}    → per-season win% and net rating
└── /health               → status + loaded models
        │
        ▼
CSV data + pickled models (scikit-learn / XGBoost / SHAP background)
```

---

## 📂 Repo Structure

```
data-dunk/
├── backend/
│   ├── main.py                  # FastAPI app, all endpoints
│   ├── data/
│   │   ├── fetch_stats.py       # synthetic data generator
│   │   ├── team_stats.csv
│   │   └── player_stats.csv
│   ├── models/
│   │   ├── train.py             # trains all 4 models + SHAP
│   │   ├── game_model.pkl / xgb_game_model.pkl
│   │   ├── player_model.pkl / xgb_player_model.pkl
│   │   └── shap_game_data.pkl
│   ├── tests/                   # pytest unit + integration tests
│   ├── locustfile.py            # load testing
│   ├── run_pipeline.sh          # one-command: install → generate → train → serve
│   ├── Dockerfile
│   └── requirements.txt
│
└── my-website/
    ├── index.html
    ├── src/
    │   ├── index.html               # Predictions page
    │   ├── game-insights.html       # SHAP explainability page
    │   ├── player-compare.html      # Player Lab
    │   ├── scripts/                 # main.ts, insights.ts, player-compare.ts
    │   └── styles/
    └── docs/                        # math breakdown, project definition
```

---

## 🚀 Running Locally

**Backend:**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash run_pipeline.sh
# → generates data, trains all models, starts the API on http://localhost:8000
# → interactive docs at http://localhost:8000/docs
```

**Frontend:**
```bash
cd my-website
npm install
npm start
# → served on http://localhost:3000 (or via Live Server on the src/ folder)
```

**With Docker:**
```bash
docker build -t data-dunk ./backend
docker run -p 8000:8000 data-dunk
```

---

## ✅ Testing

- **14 unit tests** (pytest) — rolling averages, rest-day calculation, feature differencing, SHAP description formatting
- **8 integration tests** (httpx + FastAPI TestClient) — every endpoint, valid and invalid input
- **12 manual end-to-end scenarios** — desktop + mobile, all passed
- **Load testing (Locust)** — 100 concurrent users, 3,696 requests, zero failures; median 142ms for predictions, 847ms for SHAP explanations

---

## 🔮 Future Work

- Replace synthetic data with real historical NBA data
- Add injury reports as a model feature
- Cache SHAP explanations for repeated matchups (biggest current performance bottleneck)
- Compare more than two players at once in Player Lab
- Extend the same pipeline (data → features → model → SHAP) to other sports

---

## 👤 Author
**Jamiu Idowu**
