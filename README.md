**Access Application here**  : https://fhxhvod8eyabcgedaojjns.streamlit.app/


# ML Learning Lab

An interactive Streamlit app that teaches the three main ways machines learn from data — **Supervised Learning**, **Unsupervised Learning**, and **RLHF** — with plain-language "Why?" explainers next to every metric so results are easy to understand.

## What's inside

- **Introduction** — quick overview of the three paradigms and when to use each
- **Supervised Learning** — Classification (churn prediction) and Regression (house prices) with live model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **Unsupervised Learning** — Clustering (KMeans, DBSCAN), Dimensionality Reduction (PCA, t-SNE), and Anomaly Detection
- **RLHF Simulation** — rank responses and watch a reward model learn human preferences
- **Compare All Three** — side-by-side summary
- **Quiz Mode** — test what you learned

Every score (accuracy, R², silhouette, SHAP, etc.) comes with a kid-friendly "Why?" explainer that reacts to the actual value.

## How to use

1. **Pick a section** in the sidebar on the left.
2. **Open Settings** in the sidebar to tweak data size, noise, model complexity, or test split — charts update live.
3. **Click any expander** (e.g. "Dataset Preview", "How to read the metrics") to reveal more detail.
4. **Hover over charts** for exact values. They are interactive — zoom, pan, and download.
5. Read the **"Why this score?"** callouts after each result to understand *why* you got that number.

**Suggested path:** Introduction → Supervised → Unsupervised → RLHF → Compare → Quiz.

## Run locally

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly shap xgboost
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy

**Vercel is not supported** — Vercel is built for frontend/serverless workloads, but Streamlit needs a long-running Python server. Use one of these instead:

### Streamlit Community Cloud (recommended, free)
1. Push this repo to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, select this repo, set the main file to `app.py`, deploy.

### Hugging Face Spaces (free)
1. Create a new Space, pick the **Streamlit** template.
2. Add `app.py` and a `requirements.txt` with the dependencies above.

### Render / Railway (free tier)
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
shap
xgboost
```
