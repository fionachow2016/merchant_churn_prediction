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

## 🔭 Next steps

- Tune XGBoost hyperparameters (`GridSearchCV`) and add `scale_pos_weight` to address class balance.
- Evaluate with precision-recall curves, not just ROC-AUC, given the retention use case.
- Deploy the model as a **Merchant Health Scorecard** web app (Streamlit) — the next project in this portfolio.

---

## 👤 About

Built by **Fiona Chow** — former Data Analyst and Technical Account Manager (Sales partnership, top-tier high-volume merchants), building a data & analytics portfolio focused on the merchant/payments domain.
