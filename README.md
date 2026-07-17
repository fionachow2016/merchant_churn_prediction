# Merchant Churn Prediction 🏪

Predicting which merchants are likely to churn — and translating the model into a prioritized action list for the account team.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fionachow2016/merchant_churn_prediction/blob/main/Project1_Merchant_Churn_Prediction.ipynb)

**Tech stack:** Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · XGBoost

---

## 📌 Problem

Merchant churn is expensive: losing an active, revenue-generating merchant costs far more than the effort of retaining one. But account teams can't watch every account equally, so the real question is not just *"who churned?"* — it's ***"which still-active merchants are most at risk, and what should we do about them?"***

This project builds a churn model that (1) predicts churn probability for every merchant and (2) ranks active merchants by risk so the account team can focus its outreach where it matters most.

*Context: I spent years as a Technical Account Manager partnering with Sales to manage top-tier, high-volume merchants. This project applies data science to a problem I previously owned by hand.*

---

## 📊 Data

A merchant-level dataset of **3,000 merchants** with **19 features** and a binary `churned` target. **Baseline churn rate: 38.7%.**

> Note: the dataset is synthetic, generated to realistically mirror merchant/payments behavior. The focus of this project is the **method and the business framing**, not the specific figures.

| Category | Features |
|---|---|
| **Firmographics** | `industry`, `tier` (Small/Medium/Enterprise), `region`, `plan`, `tenure_months` |
| **Activity & revenue** | `avg_monthly_txn`, `avg_ticket_size`, `monthly_revenue`, `days_since_last_txn`, `revenue_trend_pct` |
| **Support** | `support_tickets_90d`, `avg_resolution_hours`, `satisfaction_score`, `failed_payments_90d` |
| **Engagement** | `account_manager_calls_90d`, `features_adopted`, `integration_active` |
| **Target** | `churned` (1 = churned, 0 = retained) |

---

## 🔍 Approach

1. **Data checks** — verified shape, dtypes, and confirmed no missing values.
2. **Exploratory analysis (EDA)** — churn rate by tier and industry, boxplots of key numeric signals for churned vs retained merchants, and a correlation heatmap.
3. **Feature preparation** — one-hot encoded categorical variables; stratified 75/25 train-test split.
4. **Modeling** — a **Logistic Regression baseline** (always establish a simple bar first), then a tuned **XGBoost** classifier.
5. **Evaluation** — compared models on precision, recall, and ROC-AUC; reviewed the confusion matrix.
6. **Interpretation & action** — ranked feature importance and converted the top drivers into a prioritized at-risk merchant list for the account team.

---

## 📈 Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression (baseline) | **0.75** |
| XGBoost | 0.71 |

**Key finding on modeling:** the simple logistic baseline slightly outperformed XGBoost on this dataset — a useful reminder that a more complex model isn't automatically better, and that a strong baseline is worth establishing before reaching for gradient boosting.

The model also produces a **ranked list of at-risk active merchants**, which is the deliverable an account team would actually use — the highest-probability accounts to call first.

---

## 💡 Key Business Takeaways

1. **Critical churn signal — idle + declining revenue = act now.** Merchants combining rising `days_since_last_txn` (>30 days) with negative `revenue_trend_pct` churn at **63%** vs. the 38.7% baseline — yet they're only **~6% of merchants (≈190 accounts)**, a small, high-value target list. *Action: trigger automated outreach the moment a merchant crosses 30 days idle while revenue declines.*

2. **Tier-based retention — protect small merchants.** Small-tier merchants churn at **43%** vs. 34% (medium) and 30% (enterprise). *Action: build a lightweight, automated onboarding and check-in cadence for small merchants, where 1:1 account management isn't cost-effective.*

3. **Drive product adoption — the most controllable lever.** Merchants with an **active integration churn at 36% vs 45%** without one, and **heavy feature adopters churn at 30% vs 45%** for light adopters. *Action: make integration setup and feature adoption a core part of onboarding and QBRs.*

4. **Proactive contact works — but confirm the cause.** Churn falls steadily with account-manager contact: **0 calls → 47% churn, 4 calls → 20%.** *Action: prioritize outreach to the high-risk list. Caveat: feature importance shows correlation, not proven causation — calls may be partly reverse-causal (at-risk merchants already get more calls), so confirm with a small controlled test before scaling.*

---

## 🚀 Run it yourself

**Option A — Google Colab (no setup):** click the *Open in Colab* badge above, then upload `merchant_churn.csv` via the folder icon and run all cells.

**Option B — locally:**

```bash
git clone https://github.com/fionachow2016/merchant_churn_prediction.git
cd merchant_churn_prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook Project1_Merchant_Churn_Prediction.ipynb
```

---

## 📁 Repository contents

| File | Description |
|---|---|
| `Project1_Merchant_Churn_Prediction.ipynb` | Full analysis: EDA, modeling, evaluation, and business takeaways |
| `merchant_churn.csv` | The merchant dataset (3,000 rows, 19 features) |
| `README.md` | This file |

---

## 🔭 Roadmap

The current notebook is an end-to-end *analysis*. The roadmap below turns it into a *repeatable, production-style workflow* — from a better model, to scoring and alerting, to a scheduled pipeline that retrains itself. Items are grouped by theme and ordered roughly by how I'd tackle them.

### 1. Modeling & evaluation
- **Tune XGBoost** with `GridSearchCV` / `RandomizedSearchCV` and add `scale_pos_weight` to address the class imbalance.
- **Evaluate on precision-recall curves**, not just ROC-AUC — PR curves are more informative for the imbalanced retention use case.
- **Calibrate probabilities** (Platt scaling / isotonic) so a "70% churn risk" score is trustworthy for prioritization and thresholds.
- **Explainability** with SHAP values to show *per-merchant* churn drivers, giving the account team a reason to call, not just a score.
- **Cross-validation** with stratified k-folds to report performance ranges rather than a single split.

### 2. Productionization / MLOps
- **Automate churn risk scoring for new merchants** — refactor the notebook into a reusable `score.py` that loads the saved model and outputs a churn probability for any new/updated merchant record. *(Your point.)*
- **Set up an automated email alert for high-risk merchants** — a job that flags merchants crossing a risk threshold (or the idle + declining-revenue trigger) and emails the account team a prioritized list. *(Your point.)*
- **Build a dashboard to visualize churn scores** — an interactive view (Streamlit / Plotly Dash / Power BI) of the ranked at-risk list, filterable by tier, region, and industry, as the "Merchant Health Scorecard." *(Your point.)*
- **Schedule the retraining script to run weekly** — wrap training in a `train.py` and schedule it (cron / GitHub Actions / Airflow) so the model refreshes on the latest data. *(Your point.)*
- **Integrate the weekly schedule with GitHub Actions** — a scheduled workflow (`.github/workflows/retrain.yml` on a `cron`) that runs retraining, versions the model artifact, and posts a run summary. *(Your point.)*
- **Model registry & versioning** — track models, params, and metrics with MLflow (or DVC) so each retrain is reproducible and comparable.
- **Persist artifacts** — save the trained model, encoder, and feature schema (e.g., `joblib` / `pickle`) so scoring and training share one source of truth.

### 3. Deployment & serving
- **Package as a REST API** — expose scoring via FastAPI/Flask so other systems (CRM, outreach tools) can request risk scores on demand.
- **Containerize with Docker** for consistent, portable deployment across environments.
- **Batch vs. real-time scoring** — support both a nightly batch score of the full book and on-demand scoring for a single merchant.
- **CRM integration** — push scores and the ranked list into the account team's existing workflow (e.g., Salesforce) instead of a standalone tool.

### 4. Monitoring & reliability
- **Data & concept drift monitoring** — track feature distributions and churn-rate shifts over time; alert when the model needs attention.
- **Performance monitoring** — log live predictions vs. realized outcomes to watch precision/recall degrade before it hurts.
- **Data validation** — schema and range checks (e.g., Great Expectations / `pandera`) at the top of the pipeline to catch bad inputs early.

### 5. Engineering & CI/CD
- **Refactor notebook → modular package** — split into `data`, `features`, `train`, `score`, and `evaluate` modules with a clear config.
- **Unit & integration tests** (`pytest`) for feature engineering and scoring, run automatically on every push.
- **CI pipeline** — GitHub Actions to lint, test, and validate the model on pull requests.
- **Reproducible environment** — pin dependencies (`requirements.txt` / `environment.yml`) and document a one-command setup.
- **Documentation** — a short model card covering intended use, features, metrics, and known limitations.

---

## 👤 About

Built by **Fiona Chow** — former Data Analyst and Technical Account Manager (Sales partnership, top-tier high-volume merchants), building a data & analytics portfolio focused on the merchant/payments domain.
