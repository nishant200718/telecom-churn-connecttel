# ConnectTel Production Deployment & Operational Architecture Guide

This document details the production architecture, reproducibility verification, inference protocols, and monitoring lifecycle for the ConnectTel Customer Churn Machine Learning System.

---

## 1. Production Architecture Overview

The ConnectTel Churn Engine is architected as an enterprise-grade microservice supporting both **real-time synchronous inference** (CRM screen-pops and customer service calls) and **scheduled batch scoring** (monthly retention marketing campaigns).

```
                      ┌──────────────────────────────────────┐
                      │    ConnectTel Enterprise Sources     │
                      │  (Billing System / CRM Data Lake)   │
                      └──────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
      [Real-Time CRM Inquiries]                      [Daily/Monthly Batch Jobs]
                 │                                               │
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    REST API (/predict)    │                   │   CLI Batch (predict.py)  │
   │  Flask / Gunicorn Service │                   │  High-Throughput Scoring  │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    Serialized Pipeline Artifact      │
                      │       (churn_pipeline.joblib)        │
                      │  - Data Cleaning & Feature Alignment │
                      │  - Standard Scaling Transform        │
                      │  - XGBoost Probabilistic Scorer      │
                      │  - Business Policy Decision Engine   │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │           Actionable Output          │
                      │  - Churn Probability (0.0 to 1.0)    │
                      │  - Classification (Threshold: 0.28)  │
                      │  - Risk Tier (Critical, Mod, Low)    │
                      │  - Prescribed Retention Action       │
                      └──────────────────────────────────────┘
```

---

## 2. Model Performance Across Decision Thresholds (Zero-Leakage Pipeline)

Trained with zero data leakage using `sklearn.pipeline.Pipeline` with `StandardScaler` inside 5-Fold Stratified Cross-Validation:
- **5-Fold Cross-Validation ROC-AUC:** `0.8485`
- **Holdout Test-Set ROC-AUC:** `0.8457`

### Decision Threshold Trade-Off Analysis

| Operating Threshold | Accuracy | Precision | Recall (Catch Rate) | F1-Score | Strategic Business Application |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Default ($\tau = 0.50$)** | **79.99%** | **66.08%** | 50.53% | 0.5727 | Conservative outreach; minimizes false alarms. |
| **F1-Optimal ($\tau = 0.35$)** | 77.22% | 56.33% | 73.80% | **0.6389** | Balanced operations with peak harmonic accuracy. |
| **Retention-Optimal ($\tau = 0.28$)** | 74.38% | 51.12% | **79.14% (~80%)** | 0.6212 | **Recommended Production Standard**: Maximizes churner detection prior to cancellation. |

> **Economic Justification:** Missing a churned customer costs **$893.30/year** in lost subscription revenue. In contrast, extending a retention loyalty discount costs **$10–$15/month ($120–$180/year)** only when accepted. Therefore, operating at $\tau = 0.28$ yields the highest Net Economic Value for ConnectTel.

---

## 3. Verified SHAP Feature Attribution vs. Business Conclusions

Our model interpretations are mathematically derived directly from the tree structure using `shap.TreeExplainer`:

| Feature Rank | Feature Name | Mean \|SHAP\| Impact | Direction of Effect | Verified Business Conclusion |
| :---: | :--- | :---: | :---: | :--- |
| **#1** | `tenure` | **0.5134** | Higher $\rightarrow$ Lower Churn | Brand longevity forms loyalty; first 12 months require highest intervention. |
| **#2** | `Contract_Two year` | **0.4709** | True $\rightarrow$ Strong Retention | Contract commitment is the #1 structural churn barrier. Month-to-month users lack this anchor. |
| **#3** | `InternetService_Fiber optic` | **0.3403** | True $\rightarrow$ Higher Churn | Fiber optic users suffer from high prices and setup friction; largest revenue loss vector. |
| **#4** | `Contract_One year` | **0.2052** | True $\rightarrow$ Strong Retention | Even 1-year agreements reduce churn risk by over 73%. |
| **#5** | `PaymentMethod_Electronic check` | **0.1876** | True $\rightarrow$ Higher Churn | Friction in manual payments correlates with dissatisfaction; auto-pay drives retention. |
| **#6** | `TotalChargesPerTenure` | **0.0987** | Higher $\rightarrow$ Higher Churn | Engineered spend-velocity metric detects bill shock before cancellation. |
| **#7** | `OnlineSecurity_Yes` | **0.0983** | True $\rightarrow$ Lower Churn | Add-on security bundles significantly deepen switching costs. |
| **#8** | `TechSupport_Yes` | **0.0789** | True $\rightarrow$ Lower Churn | Access to tech support resolves dissatisfaction early. |

---

## 4. Financial ROI & Revenue Protection Calculation Model

### Mathematical Formulation
$$\text{Net Revenue Saved} = \left( N_{\text{churn}} \times \text{Recall} \times \text{Retention Success Rate} \times \text{ARR}_{\text{churner}} \right) - \left( N_{\text{saved}} \times \text{Annual Incentive Cost} \right)$$

> **Important Assumption & Methodology Note:**  
> The $5.68M revenue protection figure is an **Estimated Enterprise Scenario Projection**, based on an extrapolated enterprise base of 100,000 active subscribers, a modeled 26.54% churn rate, an empirical 79.14% model recall, a standard 35% retention campaign acceptance rate, and a $10/month loyalty credit ($120/year). It represents a target projection for scenario planning, not a guaranteed return. Actual realized savings depend on customer contact rates, retention offer elasticity, and operational execution.

### 1. Analysis of Dataset Cohort (7,043 Active Subscribers)
- Total Churned Subscribers: **1,869** (26.54% churn rate)
- Average Monthly Charge per Churned Subscriber: **$74.44**
- Average Annual Recurring Revenue (ARR) per Churned Subscriber: **$893.30** ($74.44 \times 12)
- Total Annual Revenue Lost to Churn in Sample: **$1,669,570**
- Flagged at $\tau = 0.28$ (79.14% Recall): **1,479 churners identified**
- Conservative Retention Success Rate: **35.0%**
- Number of Subscribers Saved: $1,479 \times 0.35 =$ **517 subscribers**
- Gross Revenue Protected: $517 \times \$893.30 =$ **$461,834**
- Cost of Retention Incentive ($10/mo credit for 12 mo = $120/yr): $517 \times \$120 =$ **$62,040**
- **Net Annual Revenue Saved (Cohort):** $\$461,834 - \$62,040 =$ **$399,794**

### 2. Enterprise Scenario Projection (100,000 ConnectTel Subscriber Base)
- Annual Churn Pool (26.54%): **26,535 subscribers**
- Annual Revenue at Risk: $26,535 \times \$893.30 =$ **$23,703,715 ($23.7M)**
- Churners Flagged by Engine (79.14% Recall): **21,000 subscribers**
- Subscribers Retained (35% Acceptance): **7,350 subscribers**
- Gross Annual Revenue Protected: $7,350 \times \$893.30 =$ **$6,565,755 ($6.57M)**
- Retention Program Budget ($120 incentive/subscriber): $7,350 \times \$120 =$ **$882,000**
- **Estimated Net Annual Revenue Protection:** $\$6,565,755 - \$882,000 =$ **$5,683,755 (~$5.68 Million Projection)**

### Sensitivity Matrix (Net Projected Revenue on 100k Base)

| Retention Success Rate | $10/mo Credit ($120/yr) | $15/mo Credit ($180/yr) | $20/mo Credit ($240/yr) |
| :---: | :---: | :---: | :---: |
| **25% (Conservative)** | $4,060,000 | $3,745,000 | $3,430,000 |
| **35% (Baseline Expected)** | **$5,684,000** | **$5,243,000** | **$4,802,000** |
| **50% (High Engagement)** | $8,120,000 | $7,490,000 | $6,860,000 |

---

## 5. Verification & Testing Instructions

The production codebase includes an automated test harness covering artifact integrity, deterministic prediction bounds, inference latency (<50ms SLA), and API endpoints.

```bash
# Run automated test suites (runnable from root or production/)
python test_production.py
python test_api.py
```

Expected Output:
```
=================================================================
  CONNECTTEL ML PRODUCTION PIPELINE VERIFICATION SUITE
=================================================================
Test 1: Verifying production artifacts on disk...
[PASS] All 7 production artifacts verified successfully.

Test 2: Verifying pipeline loading and scoring accuracy...
[PASS] High-Risk Customer correctly scored: Prob=73.15%, Tier=CRITICAL RISK
[PASS] Low-Risk Customer correctly scored: Prob=1.54%, Tier=LOW RISK

Test 3: Benchmarking inference latency (Production SLA < 50ms)...
[PASS] Average Inference Latency: 21.22 ms
[PASS] 95th Percentile Latency: 22.33 ms
[PASS] Production SLA (<50ms) satisfied.

Test 4: Verifying Flask REST API endpoints with test client...
[PASS] Health Check: healthy (Model AUC: 0.8457)
[PASS] /predict API response valid (Latency: 22.70 ms, Prob: 73.15%)
[PASS] /batch_predict API response valid (2 records scored)
=================================================================
  ALL 4 PRODUCTION VERIFICATION TESTS PASSED (100% SUCCESS)
=================================================================
```

And for REST API Verification Suite (`test_api.py`):
```
=================================================================
  CONNECTTEL REST API AUTOMATED VERIFICATION SUITE
=================================================================
[PASS] Test 01: /health endpoint verified (HTTP 200)
[PASS] Test 02: Root / endpoint verified (HTTP 200)
[PASS] Test 03: High-risk scoring verified (Prob: 73.15%, Tier: CRITICAL RISK)
[PASS] Test 04: Low-risk scoring verified (Prob: 1.54%, Tier=LOW RISK)
[PASS] Test 05: Wrapped payload format verified (HTTP 200)
[PASS] Test 06: List payload scoring verified (2 scored)
[PASS] Test 07: /batch_predict endpoint verified (2 scored)
[PASS] Test 08: Error handling for empty payload verified (HTTP 400)
[PASS] Test 09: GET /predict documentation verified (HTTP 200)
[PASS] Test 10: Live network socket HTTP test verified (HTTP 200)

Ran 10 tests in 0.507s - OK (100% SUCCESS)
=================================================================
```

---

## 6. How to Deploy

### Option A: Direct Python Execution
```bash
python predict.py --input sample_customers.json
python app.py
```

### Option B: Docker Container Deployment
```bash
# Build Docker image
docker build -t connecttel-churn-api:v1 .

# Run containerized microservice on port 8080
docker run -d -p 8080:8080 --name connecttel-service connecttel-churn-api:v1
```
