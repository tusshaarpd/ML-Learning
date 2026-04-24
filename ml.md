```text id="codex_pro_ml_learning_app"
Build a production quality single file Streamlit application in Python that can be used as a portfolio grade interactive learning product called:

ML Learning Lab: Supervised vs Unsupervised vs RLHF

Objective
Create a visually impressive and highly educational app that helps users deeply understand the difference between supervised learning, unsupervised learning, and RLHF through synthetic datasets, model training, explainability, metrics, and interactive simulations.

Audience
Beginners, product managers, interview candidates, students, and business leaders who want intuitive understanding of AI concepts.

Tech stack
Python
Streamlit
Pandas
NumPy
Scikit learn
Matplotlib
Seaborn
Plotly
SHAP

Use only synthetic or dummy data. No external datasets.

App structure
Use wide layout.
Use polished dashboard styling.
Use sidebar navigation.
Use tabs inside each section.
Use clean spacing and readable typography.

Main sidebar sections
1 Introduction
2 Supervised Learning
3 Unsupervised Learning
4 RLHF Simulation
5 Compare All Three
6 Quiz Mode

Global sidebar controls
Random seed
Dataset size
Noise level
Model complexity
Train test split
Theme selector if possible

Section 1 Introduction
Create a clean overview page that explains:

What is machine learning
Difference between:
Supervised learning = learns from labeled examples
Unsupervised learning = finds hidden patterns
RLHF = learns behavior using human preferences

Use a comparison table:
Input data type
Output goal
Use cases
Algorithms
Metrics
Business examples

Add a visual flow diagram.

Section 2 Supervised Learning
Create two tabs:
A Classification
B Regression

Classification tab
Generate synthetic customer churn style dataset with:
Age
Tenure
Monthly spend
Usage frequency
Support tickets
Plan type categorical

Target:
Churn yes or no

Allow model selection:
Logistic Regression
Decision Tree
Random Forest
XGBoost if available else skip

Show:
Dataset preview
Feature distributions
Correlation heatmap
Train test split
Prediction probabilities
Confusion matrix
Precision
Recall
F1 score
Accuracy
ROC AUC
Precision Recall curve

Add threshold slider to show precision recall tradeoff.

Explain business meaning:
High recall catches more churn users
High precision reduces wasted retention offers

SHAP Explainability
Use SHAP for tree models or KernelExplainer fallback.
Show:
Feature importance
Single prediction explanation
Summary plot

Regression tab
Generate synthetic house pricing or revenue forecasting dataset.

Features:
Area
Rooms
Location score
Age of property
Distance to city
Amenities score

Target:
Price

Allow model selection:
Linear Regression
Random Forest Regressor

Show:
Predicted vs Actual chart
Residual plot
MAE
RMSE
R squared

Explain use cases.

Section 3 Unsupervised Learning
Create three tabs:
A Clustering
B Dimensionality Reduction
C Anomaly Detection

Clustering
Generate unlabeled customer segments dataset.

Use:
KMeans
DBSCAN

Show:
2D scatter clusters
Centroids
Cluster sizes
Silhouette score if applicable

Explain examples:
Customer segmentation
Fraud grouping
Behavior discovery

Dimensionality Reduction
Generate high dimensional data with 8 to 12 features.

Use:
PCA
tSNE

Show:
2D projection chart
Explained variance for PCA
How hidden structure appears visually

Anomaly Detection
Use Isolation Forest or Local Outlier Factor.

Show:
Normal points vs anomalies
Outlier score distribution

Explain fraud detection and monitoring use cases.

Section 4 RLHF Simulation
This must be conceptual but interactive.

Build a simulated chatbot response ranking system.

Input:
Dummy user prompt examples:
How do I reset password
Write apology email
Explain taxes simply
Recommend product

Generate 3 dummy responses:
Helpful
Neutral
Poor

Let user rank responses manually.

Then simulate:
Preference dataset creation
Reward model score learning
Policy improvement over iterations

Visuals:
Bar chart of reward scores
Policy improvement over rounds
Win rate of preferred response
Before vs after quality chart

Explain:
Supervised fine tuning
Reward model
Reinforcement optimization
Why RLHF is useful for LLMs

Clearly label:
Educational simulation, not full production RLHF.

Section 5 Compare All Three
Create a polished comparison dashboard.

Rows:
Training data
Labels needed
Feedback loop
Output type
Common metrics
Explainability
Business use cases
Strengths
Weaknesses

Include decision tree:
If labeled data use supervised
If no labels use unsupervised
If behavior alignment needed use RLHF

Section 6 Quiz Mode
Create 10 random MCQ questions.

Examples:
Which learning type uses labels
Which metric is best for false negatives
What does PCA do
Why RLHF matters

Show score and explanations.

Visual excellence requirements
Use Plotly interactive charts wherever possible.
Use metric cards.
Use progress bars.
Use expanders for explanations.
Use color coded insights.
Use responsive layout.

Engineering quality
Use clean modular functions:
generate_data()
train_model()
evaluate_model()
plot_metrics()
run_shap()
simulate_rlhf()

Cache expensive operations.
Use reproducible random seeds.
Gracefully handle errors.
No dead code.

Code quality
Single Python file only.
Must run with:
streamlit run app.py

At top include requirements comment.

Deliverable
Return complete final code only.
No extra explanation.
```
