# Requirements:
# pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly shap xgboost

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.3rem; }
    h3 { font-size: 1.05rem; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, css_class=""):
    st.metric(label, value)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("ML Learning Lab")

section = st.sidebar.radio(
    "Section",
    ["Introduction", "Supervised Learning", "Unsupervised Learning",
     "RLHF Simulation", "Compare All Three", "Quiz Mode"],
)

with st.sidebar.expander("Settings (optional)", expanded=False):
    st.caption("Tweak these to see how data and models react.")
    random_seed = st.slider("Random Seed", 0, 100, 42)
    dataset_size = st.slider("Dataset Size", 100, 2000, 500, step=100)
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.2, step=0.05)
    model_complexity = st.slider("Model Complexity", 1, 10, 5)
    test_split = st.slider("Test Split %", 10, 50, 20, step=5)

np.random.seed(random_seed)


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_classification_data(n, noise, seed):
    rng = np.random.RandomState(seed)
    age = rng.randint(18, 70, n)
    tenure = rng.randint(1, 72, n)
    monthly_spend = rng.uniform(20, 200, n)
    usage_freq = rng.randint(1, 30, n)
    support_tickets = rng.poisson(2, n)
    plan_type = rng.choice(["Basic", "Standard", "Premium"], n)

    churn_score = (
        -0.02 * age + 0.03 * support_tickets - 0.01 * tenure
        - 0.005 * monthly_spend - 0.02 * usage_freq
        + 0.5 * (plan_type == "Basic").astype(float)
        + noise * rng.randn(n)
    )
    churn = (churn_score > np.median(churn_score)).astype(int)

    df = pd.DataFrame({
        "Age": age, "Tenure": tenure, "MonthlySpend": monthly_spend,
        "UsageFrequency": usage_freq, "SupportTickets": support_tickets,
        "PlanType": plan_type, "Churn": churn,
    })
    return df


@st.cache_data
def generate_regression_data(n, noise, seed):
    rng = np.random.RandomState(seed)
    area = rng.uniform(500, 5000, n)
    rooms = rng.randint(1, 8, n)
    location_score = rng.uniform(1, 10, n)
    age_property = rng.randint(0, 50, n)
    distance_city = rng.uniform(0.5, 30, n)
    amenities = rng.uniform(1, 10, n)

    price = (
        50 * area + 20000 * rooms + 15000 * location_score
        - 2000 * age_property - 5000 * distance_city + 10000 * amenities
        + noise * 50000 * rng.randn(n)
    )
    price = np.clip(price, 10000, None)

    return pd.DataFrame({
        "Area": area, "Rooms": rooms, "LocationScore": location_score,
        "PropertyAge": age_property, "DistanceToCity": distance_city,
        "AmenitiesScore": amenities, "Price": price,
    })


@st.cache_data
def generate_cluster_data(n, seed):
    rng = np.random.RandomState(seed)
    centers = [[2, 2], [8, 3], [5, 8]]
    labels = rng.choice(3, n)
    data = np.array(centers)[labels] + rng.randn(n, 2) * 1.0
    return pd.DataFrame(data, columns=["Feature1", "Feature2"])


@st.cache_data
def generate_high_dim_data(n, seed):
    rng = np.random.RandomState(seed)
    n_features = 10
    n_groups = 3
    labels = rng.choice(n_groups, n)
    means = rng.randn(n_groups, n_features) * 3
    data = means[labels] + rng.randn(n, n_features)
    cols = [f"Feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    df["TrueGroup"] = labels
    return df


@st.cache_data
def generate_anomaly_data(n, seed):
    rng = np.random.RandomState(seed)
    n_normal = int(n * 0.95)
    n_outlier = n - n_normal
    normal = rng.randn(n_normal, 2) * 1.5 + [5, 5]
    outliers = rng.uniform(0, 12, (n_outlier, 2))
    data = np.vstack([normal, outliers])
    labels = np.array([0] * n_normal + [1] * n_outlier)
    df = pd.DataFrame(data, columns=["X1", "X2"])
    df["TrueAnomaly"] = labels
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_classifier(name, complexity):
    if name == "Logistic Regression":
        return LogisticRegression(C=complexity, max_iter=1000, random_state=random_seed)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=complexity, random_state=random_seed)
    elif name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=complexity * 10, max_depth=complexity,
            random_state=random_seed, n_jobs=-1,
        )
    elif name == "XGBoost" and HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=complexity * 10, max_depth=complexity,
            random_state=random_seed, use_label_encoder=False,
            eval_metric="logloss",
        )
    return LogisticRegression(max_iter=1000, random_state=random_seed)


def train_and_evaluate_classifier(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba)
    return model, y_pred, y_proba, metrics


def train_and_evaluate_regressor(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R-squared": r2_score(y_test, y_pred),
    }
    return model, y_pred, metrics


def _select_positive_class(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[..., 1] if arr.shape[-1] > 1 else arr[..., 0]
    return arr


def run_shap_explanation(model, X_train, X_test, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])
        return explainer, _select_positive_class(shap_values)
    except Exception:
        try:
            bg = shap.sample(X_train, min(50, len(X_train)))
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X_test[:20])
            return explainer, _select_positive_class(shap_values)
        except Exception:
            return None, None


# ══════════════════════════════════════════════════════════════════════════════
# RLHF SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

RLHF_PROMPTS = {
    "How do I reset my password?": [
        "Go to Settings > Security > Reset Password. Click the link in your email to confirm. Contact support if you need help.",
        "You can try the settings page.",
        "idk lol just make a new account",
    ],
    "Write an apology email to a client": [
        "Dear [Client], I sincerely apologize for the inconvenience caused. We've identified the issue and implemented steps to prevent recurrence. Please let me know how I can make this right.",
        "Sorry for the trouble. We'll try to fix it.",
        "Not our fault but ok sorry I guess.",
    ],
    "Explain taxes simply": [
        "Taxes are payments you make to the government based on your income. The government uses this money to fund public services like roads, schools, and healthcare. Your tax rate depends on how much you earn.",
        "Taxes are money you pay the government.",
        "Taxes are complicated. Hire an accountant.",
    ],
    "Recommend a product for dry skin": [
        "I'd recommend a gentle, fragrance-free moisturizer with hyaluronic acid and ceramides. Apply after washing your face while skin is still damp. CeraVe and Cetaphil are well-regarded options.",
        "Use moisturizer.",
        "Just drink more water lmao.",
    ],
}


def simulate_rlhf(preferences, n_rounds=10):
    rng = np.random.RandomState(random_seed)
    reward_scores = np.array([0.3, 0.5, 0.7])  # initial: poor, neutral, helpful
    history = [reward_scores.copy()]

    for pref in preferences:
        if pref is not None:
            reward_scores[pref] += 0.15
            for j in range(3):
                if j != pref:
                    reward_scores[j] -= 0.05
            reward_scores = np.clip(reward_scores, 0, 1)

    rounds_data = []
    policy_quality = 0.3
    for r in range(n_rounds):
        improvement = 0.06 * (1 - policy_quality) + rng.randn() * 0.01
        policy_quality = min(policy_quality + improvement, 0.98)
        rounds_data.append({"Round": r + 1, "PolicyQuality": policy_quality})

    win_rates = []
    for r in range(n_rounds):
        wr = 0.5 + 0.4 * (r / n_rounds) + rng.randn() * 0.03
        win_rates.append({"Round": r + 1, "WinRate": min(wr, 0.99)})

    return reward_scores, rounds_data, win_rates


# ══════════════════════════════════════════════════════════════════════════════
# SECTIONS
# ══════════════════════════════════════════════════════════════════════════════

if section == "Introduction":
    st.title("ML Learning Lab")
    st.markdown("A simple, visual guide to the three main ways machines learn.")

    with st.expander("How to use this app", expanded=False):
        st.markdown("""
        Pick a section from the sidebar. Open **Settings** in the sidebar to tweak data or models.

        **Key terms:**
        - **Dataset** — a table of data
        - **Features** — the input columns
        - **Label** — what the model predicts
        - **Training** — the model learns from data
        - **Testing** — checking the model on new data
        """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Supervised")
        st.caption("Labeled data")
        st.markdown("Learns from examples with correct answers. Predicts a label or number.")
    with col2:
        st.subheader("Unsupervised")
        st.caption("No labels")
        st.markdown("Finds hidden groups or patterns in data on its own.")
    with col3:
        st.subheader("RLHF")
        st.caption("Human feedback")
        st.markdown("Learns preferred behavior from human rankings. Used to align AI assistants.")

    st.markdown("---")
    st.subheader("Side-by-side")

    comparison_df = pd.DataFrame({
        "Aspect": [
            "Input Data", "Output Goal", "Use Cases", "Key Algorithms",
            "Primary Metrics", "Business Examples",
        ],
        "Supervised": [
            "Labeled (X, y pairs)", "Predict label/value",
            "Classification, Regression",
            "Logistic Reg, Decision Trees, Random Forest, XGBoost",
            "Accuracy, F1, RMSE, R-squared",
            "Churn prediction, Price forecasting, Spam detection",
        ],
        "Unsupervised": [
            "Unlabeled (X only)", "Find structure/patterns",
            "Clustering, Dimensionality Reduction, Anomaly Detection",
            "KMeans, DBSCAN, PCA, t-SNE, Isolation Forest",
            "Silhouette, Explained Variance, Outlier Score",
            "Customer segmentation, Fraud detection, Data visualization",
        ],
        "RLHF": [
            "Prompts + human rankings", "Align behavior to preferences",
            "LLM alignment, Chatbot training, Content generation",
            "Reward Model + PPO/DPO policy optimization",
            "Win rate, Reward score, Human preference",
            "ChatGPT training, Content moderation, Safe AI assistants",
        ],
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Which one do I need?")
    st.markdown("""
    - **Have labeled data and want to predict something?** → Supervised
    - **Have data but no labels, want to find patterns?** → Unsupervised
    - **Training an AI assistant with human preferences?** → RLHF
    """)


# ── SUPERVISED LEARNING ──────────────────────────────────────────────────────
elif section == "Supervised Learning":
    st.title("Supervised Learning")
    st.markdown("Learn from labeled examples to predict outcomes.")

    with st.expander("What is Supervised Learning? (Read this first)", expanded=False):
        st.markdown("""
        **Supervised learning** is like learning with a teacher. The model is given **examples with correct answers** and learns to predict answers for new, unseen data.

        **Two types:**
        - **Classification** = predicting a **category** (e.g., "Will this customer leave? Yes or No")
        - **Regression** = predicting a **number** (e.g., "What will this house sell for?")

        **How to use this section:**
        1. Pick the **Classification** or **Regression** tab
        2. Choose a model from the dropdown -- each model learns differently
        3. Explore the data, charts, and metrics below
        4. Use the sidebar controls to change data size or noise and see how the model is affected

        **Models explained (simple version):**
        - **Logistic Regression** -- draws a straight line to separate categories. Simple but fast.
        - **Decision Tree** -- asks yes/no questions about features, like a flowchart. Easy to understand.
        - **Random Forest** -- many decision trees that vote together. More accurate but harder to explain.
        - **XGBoost** -- builds trees one after another, each fixing previous mistakes. Often the most accurate.
        - **Linear Regression** -- draws a straight line through data to predict numbers.
        """)

    tab_cls, tab_reg = st.tabs(["Classification", "Regression"])

    # ── Classification ───────────────────────────────────────────────────────
    with tab_cls:
        st.subheader("Customer Churn Prediction")
        st.markdown("""
        > **Scenario:** You work at a telecom company. You want to predict which customers will **cancel their subscription (churn)**
        > so your team can reach out and retain them before they leave. The model learns from past customer data where you already know who churned.
        """)
        df_cls = generate_classification_data(dataset_size, noise_level, random_seed)

        model_options = ["Logistic Regression", "Decision Tree", "Random Forest"]
        if HAS_XGBOOST:
            model_options.append("XGBoost")
        model_name = st.selectbox("Select Model", model_options, key="cls_model")

        with st.expander("Dataset Preview", expanded=False):
            st.dataframe(df_cls.head(20), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Shape:** {df_cls.shape[0]} rows x {df_cls.shape[1]} cols")
            with c2:
                churn_pct = df_cls["Churn"].mean() * 100
                st.markdown(f"**Churn Rate:** {churn_pct:.1f}%")

        with st.expander("Feature Distributions"):
            st.markdown("*These histograms show how each feature's values are spread out. Look for skewed distributions (most values on one side) or unusual shapes.*")
            num_cols = ["Age", "Tenure", "MonthlySpend", "UsageFrequency", "SupportTickets"]
            fig = make_subplots(rows=1, cols=len(num_cols), subplot_titles=num_cols)
            for i, col in enumerate(num_cols):
                fig.add_trace(
                    go.Histogram(x=df_cls[col], name=col, marker_color="#667eea", opacity=0.75),
                    row=1, col=i + 1,
                )
            fig.update_layout(height=300, showlegend=False, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Correlation Heatmap"):
            st.markdown("""
            *This heatmap shows how strongly each pair of features is related. Values range from -1 to +1:*
            - **+1 (dark red)** = as one goes up, the other always goes up (strong positive relationship)
            - **-1 (dark blue)** = as one goes up, the other always goes down (strong negative relationship)
            - **0 (white)** = no relationship
            - *Look at the "Churn" row/column to see which features most relate to customer churn.*
            """)
            df_encoded = df_cls.copy()
            df_encoded["PlanType"] = LabelEncoder().fit_transform(df_encoded["PlanType"])
            corr = df_encoded.corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                aspect="auto", title="Feature Correlations",
            )
            fig_corr.update_layout(height=450)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Prepare data
        df_model = df_cls.copy()
        df_model["PlanType"] = LabelEncoder().fit_transform(df_model["PlanType"])
        X = df_model.drop("Churn", axis=1)
        y = df_model["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split / 100, random_state=random_seed,
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        clf = get_classifier(model_name, model_complexity)
        if model_name == "Logistic Regression":
            clf, y_pred, y_proba, metrics = train_and_evaluate_classifier(
                clf, X_train_sc, X_test_sc, y_train, y_test
            )
        else:
            clf, y_pred, y_proba, metrics = train_and_evaluate_classifier(
                clf, X_train, X_test, y_train, y_test
            )

        # Metrics row
        st.markdown("### Model Performance")
        with st.expander("How to read these metrics"):
            st.markdown("""
            All metrics range from **0 to 1**. Higher is generally better. Here's what each one tells you:

            | Metric | What it measures | Plain English |
            |--------|-----------------|---------------|
            | **Accuracy** | % of all predictions that were correct | "How often is the model right overall?" |
            | **Precision** | Of those predicted as churn, how many actually churned? | "When the model says someone will leave, how often is it right?" |
            | **Recall** | Of those who actually churned, how many did the model catch? | "Of all customers who left, how many did we spot in advance?" |
            | **F1** | Balance between Precision and Recall | "A single number that balances both concerns" |
            | **ROC-AUC** | Model's ability to distinguish churners from non-churners | "How good is the model at ranking -- putting churners above non-churners?" |

            **Quick rule of thumb:** Above 0.8 is good, above 0.9 is excellent, below 0.6 needs improvement.
            """)
        cols = st.columns(5)
        card_classes = ["metric-green", "metric-blue", "metric-orange", "metric-red", ""]
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i]:
                metric_card(k, f"{v:.3f}", card_classes[i % len(card_classes)])

        acc = metrics.get("Accuracy", 0)
        if acc >= 0.85:
            why_msg = f"**Why so high?** The model found clear clues in the data (like tenure and support tickets). More clues = easier to guess right → accuracy went up to {acc:.2f}."
        elif acc >= 0.7:
            why_msg = f"**Why around here?** Some clues are clear, some are fuzzy. The model guesses right most of the time, but not always → accuracy landed at {acc:.2f}."
        else:
            why_msg = f"**Why so low?** The data is noisy or the model is too simple, so the clues look confusing. Confusing clues = more wrong guesses → accuracy dropped to {acc:.2f}. Try lowering **Noise Level** in sidebar Settings."
        st.info(f"Why this score? {why_msg}")

        # Confusion matrix + ROC
        c1, c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
                x=["No Churn", "Churn"], y=["No Churn", "Churn"],
                title="Confusion Matrix",
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
            with st.expander("How to read the Confusion Matrix"):
                st.markdown(f"""
                This 2x2 grid shows the model's predictions vs reality:

                - **Top-left ({cm[0][0]}):** Correctly predicted "No Churn" (True Negatives) -- customers who stayed, and the model said they'd stay.
                - **Top-right ({cm[0][1]}):** Incorrectly predicted "Churn" (False Positives) -- customers who stayed, but the model wrongly flagged them as leaving.
                - **Bottom-left ({cm[1][0]}):** Missed churners (False Negatives) -- customers who left, but the model didn't catch them. **These are the costly mistakes.**
                - **Bottom-right ({cm[1][1]}):** Correctly predicted "Churn" (True Positives) -- customers who left, and the model caught them in time.

                **Goal:** Maximize the diagonal (top-left and bottom-right) and minimize the off-diagonal.
                """)
            st.info(
                f"**Why these numbers?** The model looked at each customer's data and made a best guess. "
                f"It got the diagonal ({cm[0][0] + cm[1][1]}) right because the clues were clear. "
                f"It got the off-diagonal ({cm[0][1] + cm[1][0]}) wrong because those customers looked like the other group — "
                f"their clues were misleading."
            )

        with c2:
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#667eea", width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
                fig_roc.update_layout(
                    title=f"ROC Curve (AUC={metrics.get('ROC-AUC', 0):.3f})",
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    height=400,
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                with st.expander("How to read the ROC Curve"):
                    st.markdown(f"""
                    The ROC curve shows how well the model separates churners from non-churners at every possible threshold.

                    - **Blue line** = your model's performance. The more it hugs the **top-left corner**, the better.
                    - **Gray dashed line** = random guessing (50/50 coin flip). Your model should be well above this.
                    - **AUC (Area Under Curve) = {metrics.get('ROC-AUC', 0):.3f}** -- this is the area under the blue curve.
                      - AUC = 1.0 means perfect separation
                      - AUC = 0.5 means no better than random
                      - AUC > 0.8 is generally considered good
                    """)
                auc_v = metrics.get("ROC-AUC", 0)
                st.info(
                    f"**Why AUC is {auc_v:.2f}?** Imagine lining up all customers from 'least likely to churn' to 'most likely'. "
                    f"AUC tells us how often the real churners end up higher in the line than non-churners. "
                    f"A score of {auc_v:.2f} means {auc_v*100:.0f}% of the time the model puts them in the right order."
                )

        # Precision-Recall curve + threshold slider
        if y_proba is not None:
            st.markdown("### Precision-Recall Trade-off")
            st.markdown("""
            > **Try it:** Drag the threshold slider below. Watch how Precision and Recall change in opposite directions.
            > A **low threshold** catches more churners (high recall) but also flags many non-churners (low precision).
            > A **high threshold** only flags customers the model is very sure about (high precision) but misses others (low recall).
            """)
            threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01, key="thresh")
            y_pred_thresh = (y_proba >= threshold).astype(int)
            p_t = precision_score(y_test, y_pred_thresh, zero_division=0)
            r_t = recall_score(y_test, y_pred_thresh, zero_division=0)
            f1_t = f1_score(y_test, y_pred_thresh, zero_division=0)

            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                metric_card("Precision @ threshold", f"{p_t:.3f}", "metric-green")
            with tc2:
                metric_card("Recall @ threshold", f"{r_t:.3f}", "metric-blue")
            with tc3:
                metric_card("F1 @ threshold", f"{f1_t:.3f}", "metric-orange")

            if threshold >= 0.7:
                st.info(
                    f"**Why did Precision go up and Recall go down?** You set a very strict rule (threshold {threshold:.2f}). "
                    "The model only says 'churn' when it's really sure → it rarely makes a false alarm (high precision), "
                    "but it misses the unsure cases (low recall). Strict rule = fewer catches, but the catches are correct."
                )
            elif threshold <= 0.3:
                st.info(
                    f"**Why did Recall go up and Precision go down?** You set a very loose rule (threshold {threshold:.2f}). "
                    "The model says 'churn' even when only a little suspicious → it catches almost everyone who leaves (high recall), "
                    "but also flags many who stay (low precision). Loose rule = more catches, but many false alarms."
                )
            else:
                st.info(
                    f"**Why are these balanced?** A middle threshold ({threshold:.2f}) is like a moderate judge — "
                    "not too strict, not too lenient. Precision and recall stay roughly even because the model isn't leaning hard either way."
                )

            prec_vals, rec_vals, thresholds_pr = precision_recall_curve(y_test, y_proba)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec_vals, y=prec_vals, mode="lines", name="PR Curve", line=dict(color="#38ef7d", width=3)))
            fig_pr.add_vline(x=r_t, line_dash="dash", line_color="red", annotation_text=f"Threshold={threshold}")
            fig_pr.update_layout(
                title="Precision-Recall Curve", xaxis_title="Recall",
                yaxis_title="Precision", height=400,
            )
            st.plotly_chart(fig_pr, use_container_width=True)

            with st.expander("Business Interpretation"):
                st.markdown("""
                - **High Recall** = catches more churning customers (fewer false negatives). Use when the cost of missing a churner is high.
                - **High Precision** = fewer false alarms, so retention offers go to truly at-risk customers. Use when retention budget is limited.
                - **Threshold tuning** lets you balance these trade-offs based on business priorities.
                """)

        # SHAP
        st.markdown("### SHAP Explainability")
        with st.expander("What is SHAP and why does it matter?"):
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** answers the question: **"Why did the model make this prediction?"**

            Instead of treating the model as a black box, SHAP shows you:
            - **Which features matter most** across all predictions (Feature Importance chart)
            - **Why the model made a specific prediction** for an individual customer (Single Prediction Explanation)

            **How to read the charts:**
            - **Feature Importance:** Taller bars = that feature has more influence on predictions overall.
            - **Single Prediction:** Bars pointing **right (positive)** push the prediction toward "Churn." Bars pointing **left (negative)** push toward "No Churn."

            **Example:** If "SupportTickets" has a large positive SHAP value for a customer, it means their many support tickets are a key reason the model predicts they'll churn.

            **Why this matters for business:** You can explain to stakeholders *why* the model flagged a specific customer, not just *that* it flagged them.
            """)
        with st.spinner("Computing SHAP values..."):
            if model_name == "Logistic Regression":
                explainer, shap_vals = run_shap_explanation(clf, X_train_sc, X_test_sc, X.columns.tolist())
            else:
                explainer, shap_vals = run_shap_explanation(clf, X_train, X_test, X.columns.tolist())

        if shap_vals is not None:
            shap_importance = np.abs(shap_vals).mean(axis=0)
            feat_imp_df = pd.DataFrame({
                "Feature": X.columns, "Importance": shap_importance,
            }).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                feat_imp_df, x="Importance", y="Feature", orientation="h",
                title="SHAP Feature Importance", color="Importance",
                color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(height=400)
            st.plotly_chart(fig_imp, use_container_width=True)
            top_feat = feat_imp_df.iloc[-1]["Feature"]
            st.info(
                f"**Why is '{top_feat}' on top?** Think of features as clues. "
                f"'{top_feat}' is the clue the model looks at the most when deciding. "
                "It's like a detective saying: *'of all the evidence, this one tells me the most.'* "
                "Features lower in the chart had weaker clues, so the model relied on them less."
            )

            with st.expander("Single Prediction Explanation"):
                idx = st.number_input("Test sample index", 0, min(len(shap_vals) - 1, 99), 0)
                sample_shap = pd.DataFrame({
                    "Feature": X.columns,
                    "SHAP Value": shap_vals[idx],
                }).sort_values("SHAP Value")
                fig_s = px.bar(
                    sample_shap, x="SHAP Value", y="Feature", orientation="h",
                    title=f"SHAP Values for Sample {idx}",
                    color="SHAP Value", color_continuous_scale="RdBu_r",
                )
                fig_s.update_layout(height=350)
                st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("SHAP computation not available for this model configuration.")

    # ── Regression ───────────────────────────────────────────────────────────
    with tab_reg:
        st.subheader("House Price Prediction")
        st.markdown("""
        > **Scenario:** You're a real estate analyst predicting **house prices** based on property features.
        > Unlike classification (yes/no), regression predicts a **continuous number** (the price).
        """)
        df_reg = generate_regression_data(dataset_size, noise_level, random_seed)

        reg_model_name = st.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Random Forest Regressor"],
            key="reg_model",
        )

        with st.expander("Dataset Preview", expanded=False):
            st.dataframe(df_reg.head(20), use_container_width=True)

        X_r = df_reg.drop("Price", axis=1)
        y_r = df_reg["Price"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_r, y_r, test_size=test_split / 100, random_state=random_seed,
        )

        if reg_model_name == "Linear Regression":
            reg = LinearRegression()
        else:
            reg = RandomForestRegressor(
                n_estimators=model_complexity * 10, max_depth=model_complexity + 3,
                random_state=random_seed, n_jobs=-1,
            )

        reg, y_pred_r, reg_metrics = train_and_evaluate_regressor(reg, X_tr, X_te, y_tr, y_te)

        with st.expander("How to read regression metrics"):
            st.markdown("""
            | Metric | What it means | How to interpret |
            |--------|--------------|-----------------|
            | **MAE** (Mean Absolute Error) | Average difference between predicted and actual price | "On average, the model's predictions are off by this dollar amount." Lower is better. |
            | **RMSE** (Root Mean Squared Error) | Like MAE but penalizes large errors more | "Similar to MAE, but big mistakes count extra." Lower is better. |
            | **R-squared** | How much of the price variation the model explains (0 to 1) | "R-squared = 0.85 means the model explains 85% of why prices differ." Closer to 1.0 is better. |
            """)
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            metric_card("MAE", f"${reg_metrics['MAE']:,.0f}", "metric-green")
        with rc2:
            metric_card("RMSE", f"${reg_metrics['RMSE']:,.0f}", "metric-blue")
        with rc3:
            metric_card("R-squared", f"{reg_metrics['R-squared']:.3f}", "metric-orange")

        r2 = reg_metrics["R-squared"]
        mae = reg_metrics["MAE"]
        if r2 >= 0.8:
            st.info(
                f"**Why R² is {r2:.2f}?** Think of R² as a school grade. "
                f"The model 'understood' about {r2*100:.0f}% of what makes house prices different. "
                f"It's off by about ${mae:,.0f} on average — not bad because the clues (area, rooms, location) are strong."
            )
        elif r2 >= 0.5:
            st.info(
                f"**Why R² is {r2:.2f}?** The model got about {r2*100:.0f}% of the picture — it caught the big patterns "
                f"(bigger houses cost more) but missed finer details. That's why predictions are off by ${mae:,.0f} on average."
            )
        else:
            st.info(
                f"**Why R² is so low ({r2:.2f})?** The data is too noisy or the model is too simple to find the pattern → "
                f"predictions miss by ${mae:,.0f} on average. Try lowering **Noise Level** or picking Random Forest in sidebar Settings."
            )

        c1, c2 = st.columns(2)
        with c1:
            fig_pa = go.Figure()
            fig_pa.add_trace(go.Scatter(
                x=y_te, y=y_pred_r, mode="markers",
                marker=dict(color="#667eea", opacity=0.6), name="Predictions",
            ))
            mn, mx = min(y_te.min(), y_pred_r.min()), max(y_te.max(), y_pred_r.max())
            fig_pa.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines",
                line=dict(dash="dash", color="red"), name="Perfect",
            ))
            fig_pa.update_layout(
                title="Predicted vs Actual", xaxis_title="Actual Price",
                yaxis_title="Predicted Price", height=450,
            )
            st.plotly_chart(fig_pa, use_container_width=True)
            with st.expander("How to read this chart"):
                st.markdown("""
                Each dot is a house. The **red dashed line** shows perfect prediction (predicted = actual).
                - Dots **on the line** = model got it exactly right
                - Dots **above the line** = model overestimated the price
                - Dots **below the line** = model underestimated the price
                - The **tighter** dots cluster around the line, the better the model.
                """)

        with c2:
            residuals = y_te - y_pred_r
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=y_pred_r, y=residuals, mode="markers",
                marker=dict(color="#38ef7d", opacity=0.6),
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(
                title="Residual Plot", xaxis_title="Predicted Price",
                yaxis_title="Residual", height=450,
            )
            st.plotly_chart(fig_res, use_container_width=True)
            with st.expander("How to read the Residual Plot"):
                st.markdown("""
                A **residual** = actual price - predicted price (how much the model was "off" by).
                - Dots near the **red zero line** = small errors (good!)
                - Dots far from zero = large errors
                - **What to look for:** Dots should be randomly scattered around zero with no pattern.
                  If you see a pattern (e.g., errors get bigger for expensive houses), the model is missing something.
                """)

        with st.expander("Use Cases"):
            st.markdown("""
            - **Real estate pricing:** Predict property values from features.
            - **Revenue forecasting:** Estimate future revenue from historical trends.
            - **Demand prediction:** Forecast product demand for inventory planning.
            """)


# ── UNSUPERVISED LEARNING ────────────────────────────────────────────────────
elif section == "Unsupervised Learning":
    st.title("Unsupervised Learning")
    st.markdown("Discover hidden patterns without labels.")

    with st.expander("What is Unsupervised Learning? (Read this first)", expanded=False):
        st.markdown("""
        **Unsupervised learning** is like exploring data without a teacher. There are **no correct answers** provided --
        the model finds patterns, groups, and structure on its own.

        **Three techniques you'll explore here:**

        | Tab | What it does | Real-world analogy |
        |-----|-------------|-------------------|
        | **Clustering** | Groups similar data points together | Sorting a pile of photos into albums by similarity -- you don't know the categories in advance |
        | **Dimensionality Reduction** | Simplifies complex data so you can visualize it | Like seeing a 3D object's shadow on a wall -- you lose some detail but can see the overall shape |
        | **Anomaly Detection** | Finds unusual data points that don't fit the pattern | Like a security guard spotting someone acting differently from everyone else in a crowd |

        **How to use:** Click each tab, adjust the algorithm settings, and watch how the results change.
        """)

    tab_clust, tab_dim, tab_anom = st.tabs(["Clustering", "Dimensionality Reduction", "Anomaly Detection"])

    with tab_clust:
        st.subheader("Customer Segmentation")
        st.markdown("""
        > **Scenario:** You have customer data but **no labels**. You want to discover natural groups
        > (e.g., "budget shoppers" vs "premium buyers") so your marketing team can target each group differently.
        """)
        df_clust = generate_cluster_data(dataset_size, random_seed)

        clust_algo = st.selectbox("Algorithm", ["KMeans", "DBSCAN"], key="clust_algo")
        with st.expander("KMeans vs DBSCAN -- which to pick?"):
            st.markdown("""
            - **KMeans:** You tell it how many groups to find (use the slider). It creates round, evenly-sized clusters.
              Best when you have a rough idea of how many segments exist.
            - **DBSCAN:** It figures out the number of groups automatically based on density. It can find odd-shaped clusters
              and marks scattered points as noise (-1). Use the **Epsilon** slider to control how close points must be to form a group,
              and **Min Samples** for how many points make a valid cluster.

            **Try it:** Switch between them and see how the clusters differ!
            """)

        if clust_algo == "KMeans":
            n_clusters = st.slider("Number of Clusters", 2, 8, 3, key="n_clust")
            model_c = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
            labels = model_c.fit_predict(df_clust)
            centroids = model_c.cluster_centers_
        else:
            eps = st.slider("Epsilon (DBSCAN)", 0.1, 3.0, 0.8, 0.1, key="eps")
            min_samples = st.slider("Min Samples", 2, 20, 5, key="min_s")
            model_c = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model_c.fit_predict(df_clust)
            centroids = None

        df_clust["Cluster"] = labels.astype(str)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_cl = px.scatter(
                df_clust, x="Feature1", y="Feature2", color="Cluster",
                title="Cluster Visualization",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            if centroids is not None:
                fig_cl.add_trace(go.Scatter(
                    x=centroids[:, 0], y=centroids[:, 1], mode="markers",
                    marker=dict(size=15, color="red", symbol="x"), name="Centroids",
                ))
            fig_cl.update_layout(height=500)
            st.plotly_chart(fig_cl, use_container_width=True)

        with c2:
            cluster_counts = pd.Series(labels).value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            fig_bar = px.bar(
                cluster_counts, x="Cluster", y="Count", title="Cluster Sizes",
                color="Cluster", color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)

            unique_labels = set(labels)
            valid_labels = unique_labels - {-1}
            if len(valid_labels) >= 2:
                mask = labels != -1
                if mask.sum() > len(valid_labels):
                    sil = silhouette_score(df_clust[["Feature1", "Feature2"]][mask], labels[mask])
                    metric_card("Silhouette Score", f"{sil:.3f}", "metric-green")
                    st.caption("Ranges from -1 to 1. Above 0.5 = good clusters. Below 0.25 = overlapping/weak clusters.")
                    if sil >= 0.5:
                        st.info(
                            f"**Why {sil:.2f}?** Imagine kids in a playground forming friend groups. "
                            "Each kid is close to their own group and far from others → the groups are clearly separate. "
                            "That's why the score is high."
                        )
                    elif sil >= 0.25:
                        st.info(
                            f"**Why {sil:.2f}?** The groups exist, but some kids are standing between groups — not clearly in one. "
                            "So the clusters are real but a bit blurry at the edges."
                        )
                    else:
                        st.info(
                            f"**Why so low ({sil:.2f})?** The 'groups' overlap a lot — kids are mixed together. "
                            "Either there aren't real groups in this data, or you asked for the wrong number of groups. "
                            "Try changing the number of clusters."
                        )

        with st.expander("Business Applications"):
            st.markdown("""
            - **Customer segmentation:** Group customers by behavior for targeted marketing.
            - **Fraud grouping:** Identify clusters of suspicious transactions.
            - **Behavior discovery:** Find natural user segments in product usage data.
            """)

    with tab_dim:
        st.subheader("Dimensionality Reduction")
        st.markdown("""
        > **The problem:** Your data has 10 features (columns). You can't plot 10 dimensions on a screen.
        > Dimensionality reduction **compresses** the data into 2 dimensions so you can **see** the patterns.
        > Think of it like taking a photo of a 3D object -- you lose some info but gain visibility.
        """)
        df_hd = generate_high_dim_data(dataset_size, random_seed)

        dr_method = st.selectbox("Method", ["PCA", "t-SNE"], key="dr_method")

        features = df_hd.drop("TrueGroup", axis=1)
        scaled = StandardScaler().fit_transform(features)

        if dr_method == "PCA":
            pca = PCA(n_components=2, random_state=random_seed)
            proj = pca.fit_transform(scaled)
            df_proj = pd.DataFrame(proj, columns=["PC1", "PC2"])
            df_proj["Group"] = df_hd["TrueGroup"].astype(str)

            c1, c2 = st.columns(2)
            with c1:
                fig_pca = px.scatter(
                    df_proj, x="PC1", y="PC2", color="Group",
                    title="PCA 2D Projection",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_pca.update_layout(height=450)
                st.plotly_chart(fig_pca, use_container_width=True)
            with c2:
                pca_full = PCA(random_state=random_seed).fit(scaled)
                ev = pca_full.explained_variance_ratio_
                fig_ev = go.Figure()
                fig_ev.add_trace(go.Bar(
                    x=[f"PC{i+1}" for i in range(len(ev))], y=ev,
                    marker_color="#667eea", name="Individual",
                ))
                fig_ev.add_trace(go.Scatter(
                    x=[f"PC{i+1}" for i in range(len(ev))], y=np.cumsum(ev),
                    mode="lines+markers", name="Cumulative", line=dict(color="#38ef7d", width=3),
                ))
                fig_ev.update_layout(
                    title="Explained Variance", yaxis_title="Variance Ratio", height=450,
                )
                st.plotly_chart(fig_ev, use_container_width=True)
        else:
            perplexity = st.slider("Perplexity", 5, 50, 30, key="perp")
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_seed, n_iter=1000)
            proj = tsne.fit_transform(scaled)
            df_proj = pd.DataFrame(proj, columns=["Dim1", "Dim2"])
            df_proj["Group"] = df_hd["TrueGroup"].astype(str)

            fig_tsne = px.scatter(
                df_proj, x="Dim1", y="Dim2", color="Group",
                title="t-SNE 2D Projection",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_tsne.update_layout(height=500)
            st.plotly_chart(fig_tsne, use_container_width=True)

        with st.expander("How to interpret these charts"):
            st.markdown("""
            **Scatter plot:** Each dot is a data point that originally had 10 features, now compressed to 2 axes.
            Colors show the true groups. If the algorithm works well, **same-colored dots should cluster together**.

            **PCA Explained Variance chart:** Shows how much information each principal component captures.
            - The **bars** show individual components. PC1 usually captures the most.
            - The **green line** shows cumulative total. If the first 2 components cover 80%+, your 2D plot captures most of the structure.

            **PCA vs t-SNE:**
            - **PCA** finds directions of maximum variance and projects data onto them. It preserves global structure and is fast.
            - **t-SNE** preserves local neighborhood relationships, making clusters visually distinct. Better for visualization but slower.
            - **Perplexity** (t-SNE slider): Controls how many neighbors to consider. Low = tight small clusters. High = broader structure. Try different values!
            """)

    with tab_anom:
        st.subheader("Anomaly Detection")
        st.markdown("""
        > **Scenario:** You're monitoring transactions and want to automatically flag **suspicious activity** that doesn't
        > match normal behavior. The model learns what "normal" looks like, then flags anything that deviates.
        """)
        df_anom = generate_anomaly_data(dataset_size, random_seed)

        anom_algo = st.selectbox("Algorithm", ["Isolation Forest", "Local Outlier Factor"], key="anom_algo")
        with st.expander("Algorithm guide"):
            st.markdown("""
            - **Isolation Forest:** Randomly splits data. Anomalies are isolated quickly (few splits needed) because they're different from everything else.
              The **Contamination** slider tells the model what % of data you expect to be anomalous (e.g., 0.05 = 5%).
            - **Local Outlier Factor (LOF):** Compares how dense each point's neighborhood is to its neighbors. Points in sparse areas (far from others) are anomalies.
              The **Neighbors** slider controls how many nearby points to compare against.
            """)

        features_a = df_anom[["X1", "X2"]]
        if anom_algo == "Isolation Forest":
            contamination = st.slider("Contamination", 0.01, 0.2, 0.05, 0.01, key="contam")
            iso = IsolationForest(contamination=contamination, random_state=random_seed)
            preds = iso.fit_predict(features_a)
            scores = iso.score_samples(features_a)
        else:
            n_neighbors = st.slider("Neighbors", 5, 50, 20, key="lof_n")
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
            preds = lof.fit_predict(features_a)
            scores = lof.negative_outlier_factor_

        df_anom["Predicted"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
        df_anom["Score"] = scores

        c1, c2 = st.columns(2)
        with c1:
            fig_an = px.scatter(
                df_anom, x="X1", y="X2", color="Predicted",
                color_discrete_map={"Normal": "#38ef7d", "Anomaly": "#ef473a"},
                title="Anomaly Detection Results",
            )
            fig_an.update_layout(height=450)
            st.plotly_chart(fig_an, use_container_width=True)

        with c2:
            fig_sc = px.histogram(
                df_anom, x="Score", color="Predicted", nbins=50,
                color_discrete_map={"Normal": "#38ef7d", "Anomaly": "#ef473a"},
                title="Outlier Score Distribution", barmode="overlay", opacity=0.7,
            )
            fig_sc.update_layout(height=450)
            st.plotly_chart(fig_sc, use_container_width=True)

        n_detected = (preds == -1).sum()
        n_true = df_anom["TrueAnomaly"].sum()
        mc1, mc2 = st.columns(2)
        with mc1:
            metric_card("Detected Anomalies", str(n_detected), "metric-red")
        with mc2:
            metric_card("True Anomalies", str(n_true), "metric-orange")

        if n_detected > n_true * 1.2:
            st.info(
                f"**Why detected ({n_detected}) > true ({n_true})?** You told the model to expect lots of weirdos, "
                "so it got suspicious of normal points too. Like a guard dog that barks at every leaf. "
                "Lower the **Contamination** slider to make it pickier."
            )
        elif n_detected < n_true * 0.8:
            st.info(
                f"**Why detected ({n_detected}) < true ({n_true})?** You told the model to expect very few weirdos, "
                "so it let some real weirdos through. Like a guard dog that only barks at burglars with giant flashlights. "
                "Raise the **Contamination** slider so it flags more."
            )
        else:
            st.info(
                f"**Why the numbers match?** The Contamination setting matches the actual weirdo rate in the data, "
                "so the model's expectation lines up with reality."
            )

        with st.expander("How to interpret these results"):
            st.markdown(f"""
            **Scatter plot:** Green dots = normal, Red dots = flagged anomalies. Anomalies are typically points far from the main cluster.

            **Score distribution:** Shows the "anomaly score" for each point. Normal points cluster together on one side; anomalies are on the other.
            The less overlap between the two colors, the better the algorithm is at separating normal from unusual.

            **Detected ({n_detected}) vs True ({n_true}):**
            - If detected > true: the model is **over-flagging** (too many false alarms)
            - If detected < true: the model is **missing anomalies** (too lenient)
            - If they're close: the model is well-calibrated

            **Try:** Adjust the contamination/neighbors slider and watch how the detection count changes.
            """)

        with st.expander("Use Cases"):
            st.markdown("""
            - **Fraud detection:** Flag unusual transactions for review.
            - **System monitoring:** Detect server anomalies before outages.
            - **Quality control:** Identify defective items in manufacturing.
            """)


# ── RLHF SIMULATION ─────────────────────────────────────────────────────────
elif section == "RLHF Simulation":
    st.title("RLHF Simulation")
    st.warning("**Educational simulation** -- not a full production RLHF pipeline.")
    st.markdown("Rank AI responses to see how human feedback improves model behavior.")

    with st.expander("What is RLHF and how does this simulation work? (Read this first)", expanded=False):
        st.markdown("""
        **RLHF (Reinforcement Learning from Human Feedback)** is how AI companies like Anthropic and OpenAI train
        chatbots (like Claude and ChatGPT) to give **helpful, safe, and honest** responses.

        **The problem it solves:** A language model can generate text, but it doesn't know which responses
        humans actually prefer. RLHF teaches it by using human rankings.

        **How this simulation works (3 steps):**

        1. **You rank responses:** Below, you'll see AI-generated answers to prompts. Pick the best one for each.
           This is exactly what human raters do at AI companies.
        2. **A reward model learns your preferences:** Your rankings teach a "reward model" to score responses.
           Responses you liked get higher scores.
        3. **The AI policy improves:** Using the reward model, the AI adjusts its behavior over multiple rounds
           to produce responses more like the ones you preferred.

        **What to watch for in the charts:**
        - The reward model should give your preferred response type the **highest score**
        - Policy quality should **increase** over rounds
        - Win rate should **climb** toward 1.0
        """)

    st.markdown("---")
    st.subheader("Step 1: Rank Responses")
    st.markdown("*Read each prompt and its three responses. Pick the one you think is best. There are no wrong answers -- this is about your preference.*")

    preferences = []
    prompts = list(RLHF_PROMPTS.keys())

    for i, prompt in enumerate(prompts):
        responses = RLHF_PROMPTS[prompt]
        st.markdown(f"**Prompt:** *{prompt}*")
        labels = ["Helpful", "Neutral", "Poor"]
        cols = st.columns(3)
        for j, (col, resp, label) in enumerate(zip(cols, responses, labels)):
            with col:
                st.markdown(f"**{label}**")
                st.info(resp)

        rank = st.radio(
            f"Which response is best for \"{prompt}\"?",
            ["Helpful (A)", "Neutral (B)", "Poor (C)"],
            key=f"rank_{i}", horizontal=True,
        )
        pref_map = {"Helpful (A)": 0, "Neutral (B)": 1, "Poor (C)": 2}
        preferences.append(pref_map[rank])
        st.markdown("---")

    st.subheader("Step 2: Simulation Results")
    st.markdown("*Based on your rankings, here's how the AI learns and improves. Each chart shows a different aspect of the training process.*")

    reward_scores, rounds_data, win_rates = simulate_rlhf(preferences)

    c1, c2 = st.columns(2)
    with c1:
        labels_r = ["Helpful", "Neutral", "Poor"]
        colors_r = ["#38ef7d", "#6dd5ed", "#ef473a"]
        fig_rw = go.Figure(go.Bar(
            x=labels_r, y=reward_scores, marker_color=colors_r,
            text=[f"{s:.2f}" for s in reward_scores], textposition="auto",
        ))
        fig_rw.update_layout(title="Learned Reward Scores", yaxis_title="Score", height=400)
        st.plotly_chart(fig_rw, use_container_width=True)
        st.caption("The reward model learned to score responses based on your rankings. Higher = more preferred.")

    with c2:
        df_rounds = pd.DataFrame(rounds_data)
        fig_pol = px.line(
            df_rounds, x="Round", y="PolicyQuality",
            title="Policy Improvement Over Rounds",
            markers=True,
        )
        fig_pol.update_traces(line_color="#667eea", line_width=3)
        fig_pol.update_layout(yaxis_title="Policy Quality", height=400)
        st.plotly_chart(fig_pol, use_container_width=True)
        st.caption("Each round, the AI adjusts its behavior to produce better responses. The curve should rise and plateau.")

    c3, c4 = st.columns(2)
    with c3:
        df_wr = pd.DataFrame(win_rates)
        fig_wr = px.area(
            df_wr, x="Round", y="WinRate",
            title="Win Rate of Preferred Response",
        )
        fig_wr.update_traces(line_color="#38ef7d", fillcolor="rgba(56,239,125,0.2)")
        fig_wr.update_layout(yaxis_title="Win Rate", height=400)
        st.plotly_chart(fig_wr, use_container_width=True)
        st.caption("Win rate = how often the AI's preferred response beats alternatives. Rising = AI is learning your preferences.")

    with c4:
        before = [0.3, 0.5, 0.2]
        after = [reward_scores[0], reward_scores[1], reward_scores[2]]
        fig_ba = go.Figure()
        fig_ba.add_trace(go.Bar(name="Before RLHF", x=labels_r, y=before, marker_color="#ef473a", opacity=0.7))
        fig_ba.add_trace(go.Bar(name="After RLHF", x=labels_r, y=after, marker_color="#38ef7d", opacity=0.7))
        fig_ba.update_layout(title="Before vs After Quality", barmode="group", height=400)
        st.plotly_chart(fig_ba, use_container_width=True)
        st.caption("Compares response quality before and after RLHF training. Green bars should be higher for preferred responses.")

    with st.expander("How RLHF Works (detailed explanation)"):
        st.markdown("""
        **RLHF (Reinforcement Learning from Human Feedback)** has three stages:

        1. **Supervised Fine-Tuning (SFT):** Train a base model on high-quality demonstrations.
        2. **Reward Model Training:** Collect human rankings of model outputs and train a reward model to predict human preferences.
        3. **Policy Optimization (PPO/DPO):** Use the reward model to fine-tune the language model via reinforcement learning, maximizing the reward signal.

        **Why RLHF matters for LLMs:**
        - Aligns model behavior with human values
        - Reduces harmful or unhelpful outputs
        - Makes models more useful and safe
        - Powers systems like ChatGPT, Claude, and Gemini
        """)


# ── COMPARE ALL THREE ────────────────────────────────────────────────────────
elif section == "Compare All Three":
    st.title("Compare: Supervised vs Unsupervised vs RLHF")
    st.markdown("""
    This section puts all three learning paradigms side by side so you can see the differences at a glance.
    Use the **comparison table** as a quick reference, and the **decision guide** to figure out which approach fits a given problem.
    """)

    comparison = pd.DataFrame({
        "Aspect": [
            "Training Data", "Labels Needed?", "Feedback Loop",
            "Output Type", "Common Metrics", "Explainability",
            "Business Use Cases", "Strengths", "Weaknesses",
        ],
        "Supervised": [
            "Labeled dataset (X, y)", "Yes -- required",
            "None (static labels)", "Predictions (class or value)",
            "Accuracy, F1, RMSE, AUC", "SHAP, feature importance",
            "Churn, fraud, pricing, spam", "High accuracy with good labels",
            "Needs labeled data; can overfit",
        ],
        "Unsupervised": [
            "Unlabeled dataset (X only)", "No",
            "None", "Clusters, projections, anomaly flags",
            "Silhouette, explained variance", "Cluster profiles, projections",
            "Segmentation, anomaly detection", "No labels needed; discovers patterns",
            "Hard to evaluate; subjective results",
        ],
        "RLHF": [
            "Prompts + human rankings", "Preferences (relative)",
            "Iterative human feedback loop", "Improved behavior/policy",
            "Win rate, reward score", "Reward decomposition",
            "LLM alignment, chatbot tuning", "Aligns with human values",
            "Expensive; reward hacking risk",
        ],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Decision Guide")

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+delta", value=1,
        title={"text": "Do you have labeled data?"},
        domain={"x": [0, 0.33], "y": [0.5, 1]},
    ))
    st.markdown("""
    ```
    START
      |
      v
    Do you have labeled data?
      |           |
     YES          NO
      |           |
      v           v
    Supervised   Do you need behavior alignment?
                   |           |
                  YES          NO
                   |           |
                   v           v
                 RLHF      Unsupervised
    ```
    """)

    st.markdown("---")
    st.subheader("When to Use Each")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Supervised")
        st.success("""
        - You have labeled examples
        - Clear input-output mapping
        - Need precise predictions
        - Enough training data available
        """)
    with c2:
        st.markdown("#### Unsupervised")
        st.info("""
        - No labels available
        - Want to explore data structure
        - Need to find anomalies
        - Customer segmentation tasks
        """)
    with c3:
        st.markdown("#### RLHF")
        st.warning("""
        - Aligning AI with human preferences
        - Subjective quality judgments
        - Improving language model outputs
        - Safety and helpfulness tuning
        """)


# ── QUIZ MODE ────────────────────────────────────────────────────────────────
elif section == "Quiz Mode":
    st.title("Quiz Mode")
    st.markdown("Test your understanding of ML concepts!")
    st.markdown("""
    > **How it works:** Answer all 10 questions, then click **Submit Quiz** at the bottom.
    > You'll see your score and a detailed explanation for every question -- even the ones you got right.
    > Change the **Random Seed** in the sidebar to get a different question order. You can retake the quiz as many times as you want!
    """)

    QUIZ = [
        {
            "q": "Which type of learning uses labeled data?",
            "opts": ["Unsupervised", "Supervised", "RLHF", "None of the above"],
            "ans": 1,
            "exp": "Supervised learning requires labeled data (input-output pairs) to learn the mapping from features to targets.",
        },
        {
            "q": "Which metric is most important when false negatives are costly?",
            "opts": ["Precision", "Accuracy", "Recall", "F1 Score"],
            "ans": 2,
            "exp": "Recall measures how many actual positives are correctly identified. High recall minimizes false negatives.",
        },
        {
            "q": "What does PCA (Principal Component Analysis) do?",
            "opts": [
                "Clusters data into groups",
                "Reduces dimensionality by finding directions of maximum variance",
                "Detects anomalies",
                "Trains a reward model",
            ],
            "ans": 1,
            "exp": "PCA projects data onto principal components (directions of maximum variance), reducing dimensionality while preserving the most information.",
        },
        {
            "q": "Why is RLHF important for large language models?",
            "opts": [
                "It makes models faster",
                "It reduces model size",
                "It aligns model behavior with human preferences",
                "It removes the need for training data",
            ],
            "ans": 2,
            "exp": "RLHF uses human feedback to fine-tune models so they produce helpful, harmless, and honest outputs aligned with human values.",
        },
        {
            "q": "What does the Silhouette Score measure?",
            "opts": [
                "Classification accuracy",
                "Regression error",
                "How well-separated clusters are",
                "Feature importance",
            ],
            "ans": 2,
            "exp": "Silhouette Score ranges from -1 to 1 and measures how similar an object is to its own cluster vs. other clusters. Higher is better.",
        },
        {
            "q": "Which algorithm is commonly used for anomaly detection?",
            "opts": ["Linear Regression", "KMeans", "Isolation Forest", "PCA"],
            "ans": 2,
            "exp": "Isolation Forest isolates anomalies by randomly partitioning data. Anomalies are easier to isolate and require fewer splits.",
        },
        {
            "q": "In a confusion matrix, what does a false positive mean?",
            "opts": [
                "Model correctly predicted positive",
                "Model predicted positive but actual was negative",
                "Model predicted negative but actual was positive",
                "Model correctly predicted negative",
            ],
            "ans": 1,
            "exp": "A false positive (Type I error) means the model incorrectly predicted the positive class when the true label was negative.",
        },
        {
            "q": "What is the main difference between KMeans and DBSCAN?",
            "opts": [
                "KMeans needs the number of clusters specified; DBSCAN does not",
                "DBSCAN only works on 2D data",
                "KMeans can find anomalies but DBSCAN cannot",
                "They are identical algorithms",
            ],
            "ans": 0,
            "exp": "KMeans requires you to specify k (number of clusters). DBSCAN discovers clusters based on density and can find arbitrarily shaped clusters.",
        },
        {
            "q": "What does R-squared (R^2) tell you in regression?",
            "opts": [
                "The number of features used",
                "The proportion of variance in the target explained by the model",
                "The classification accuracy",
                "The learning rate",
            ],
            "ans": 1,
            "exp": "R-squared indicates how much of the target variable's variance is explained by the model. R^2=1.0 means perfect prediction; R^2=0 means the model is no better than predicting the mean.",
        },
        {
            "q": "In RLHF, what is the role of the reward model?",
            "opts": [
                "To generate training data",
                "To predict which output a human would prefer",
                "To compress the model",
                "To split data into train/test",
            ],
            "ans": 1,
            "exp": "The reward model is trained on human preference data to predict scores for model outputs. It guides the policy model toward generating preferred responses.",
        },
    ]

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    rng_quiz = np.random.RandomState(random_seed)
    indices = list(range(len(QUIZ)))
    rng_quiz.shuffle(indices)
    selected = indices[:10]

    for idx, qi in enumerate(selected):
        item = QUIZ[qi]
        st.markdown(f"**Q{idx + 1}.** {item['q']}")
        answer = st.radio(
            f"Select answer for Q{idx + 1}",
            item["opts"],
            key=f"quiz_{qi}",
            label_visibility="collapsed",
        )
        st.session_state.quiz_answers[qi] = item["opts"].index(answer)
        st.markdown("---")

    if st.button("Submit Quiz", type="primary"):
        st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted:
        score = 0
        for qi in selected:
            item = QUIZ[qi]
            user_ans = st.session_state.quiz_answers.get(qi, -1)
            if user_ans == item["ans"]:
                score += 1

        st.markdown(f"### Your Score: {score} / {len(selected)}")
        pct = score / len(selected) * 100
        st.progress(score / len(selected))

        if pct >= 80:
            st.success(f"Excellent! {pct:.0f}% -- You have a strong understanding of ML concepts!")
        elif pct >= 50:
            st.warning(f"Good effort! {pct:.0f}% -- Review the sections you missed.")
        else:
            st.error(f"Keep learning! {pct:.0f}% -- Explore each section to deepen your understanding.")

        st.markdown("### Explanations")
        for idx, qi in enumerate(selected):
            item = QUIZ[qi]
            user_ans = st.session_state.quiz_answers.get(qi, -1)
            correct = user_ans == item["ans"]
            icon = "+" if correct else "-"
            with st.expander(f"Q{idx + 1}: {'Correct' if correct else 'Incorrect'} -- {item['q']}"):
                st.markdown(f"**Your answer:** {item['opts'][user_ans]}")
                st.markdown(f"**Correct answer:** {item['opts'][item['ans']]}")
                st.markdown(f"**Explanation:** {item['exp']}")

# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>ML Learning Lab v1.0<br>Built with Streamlit + Scikit-learn + Plotly + SHAP</small>",
    unsafe_allow_html=True,
)
