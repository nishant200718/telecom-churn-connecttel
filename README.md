# ConnectTel - Predictive Customer Churn Machine Learning System

An enterprise-grade, leakage-free Machine Learning classification and business interpretability system for **ConnectTel Telecommunications**, designed to proactively predict and prevent customer churn.

---

## 🌐 Live Working Project & Deliverables

| Deliverable | File / Access Link | Description |
| :--- | :--- | :--- |
| **Live Working Web App** | [**Click to Open Live Project**](https://nobody-document-marion-assistance.trycloudflare.com/ui) | Full interactive Churn Engine with multi-currency & batch CSV predictor |
| **Executive Presentation (PPT)** | [`ConnectTel_Customer_Churn_Retention_Strategy.pptx`](file:///./ConnectTel_Customer_Churn_Retention_Strategy.pptx) | 10-slide executive corporate presentation deck |
| **Executive Report (PDF)** | [`ConnectTel_Business_Retention_Report.pdf`](file:///./ConnectTel_Business_Retention_Report.pdf) | Comprehensive 1.3 MB technical & financial ROI retention report |
| **Standalone Web App** | [`web_live/index.html`](file:///./web_live/index.html) | Zero-server offline/online client-side compiled XGBoost suite |
| **Source Code & API** | [`production/`](file:///./production/) | Production pipeline, dual-engine REST API, vectorized scoring, test suite |
| **Master Notebook** | [`ConnectTel_Customer_Churn_Prediction.ipynb`](file:///./ConnectTel_Customer_Churn_Prediction.ipynb) | Executed end-to-end Jupyter Notebook (EDA, ML, SHAP, cross-validation) |

---

## 📁 Project Structure
```
telecom-churn-connecttel/
│
├── ConnectTel_Customer_Churn_Prediction.ipynb   # Master executed Jupyter Notebook (Leakage-Free)
├── ConnectTel_Business_Retention_Report.pdf     # Executive corporate report (PDF)
├── ConnectTel_Business_Retention_Report.html    # Interactive HTML report
├── CONNECTTEL_CHURN_REPORT.md                   # Plain-text executive summary
├── Telco-Customer-Churn.csv                     # Telco subscriber dataset
├── DEPLOYMENT.md                                # Architecture, monitoring & deployment manual
├── README.md                                    # Project overview and run guide
│
├── production/                                  # Production deployment microservice & CLI
│   ├── churn_pipeline.joblib                   # Serialized inference pipeline artifact
│   ├── pipeline.py                             # Core pipeline class definition
│   ├── predict.py                              # Standalone CLI batch scoring tool
│   ├── app.py                                  # Flask REST API microservice (/health, /predict)
│   ├── test_production.py                      # Automated test suite (SLA latency & API)
│   ├── test_api.py                             # Automated API verification test suite (10 unit/integration tests)
│   ├── sample_customers.json                   # Verified customer payload for testing
│   ├── requirements.txt                        # Pinned dependencies
│   └── Dockerfile                              # Multi-worker container specification
│
└── images/                                      # Standalone high-resolution visualization exports
    ├── eda_churn_distribution.png
    ├── eda_hypothesis_fiber_vs_dsl.png
    ├── eda_hypothesis_partner.png
    ├── eda_demographics_services.png
    ├── eda_financial_tenure.png
    ├── model_confusion_matrices.png
    ├── model_roc_curves.png
    ├── shap_summary_plot.png
    └── shap_feature_importance.png
```

---

## 🎯 Key Client Checklist Items Resolved

1. **API Tests & Flask Setup:** Automated test suite `production/test_production.py` passes 100% of integration, SLA latency (12.98 ms), and API endpoint tests.
2. **Zero Preprocessing Leakage:** `StandardScaler` is wrapped inside `sklearn.pipeline.Pipeline` within `GridSearchCV` and 5-fold cross-validation. Scaling fits strictly on training folds.
3. **Reconciled Financial Figures:** Cohort net annual savings is verified at **$399,794** across the notebook, PDF report, and documentation.
4. **Explicit Model Metric Labels:**
   - **5-Fold Cross-Validation ROC-AUC:** `0.8485`
   - **Holdout Test-Set ROC-AUC:** `0.8457`
5. **Clean Relative Paths:** All documentation and notebooks use relative paths without any machine-specific user paths.
6. **Transparent Assumptions:** The **$5.68M** enterprise protection figure is clearly defined as an **Estimated Enterprise Scenario Projection** (based on a 100k customer base, 35% campaign acceptance rate, and $10/mo discount), not guaranteed savings.

---

## 🚀 Quick Start & Verification Instructions

### 1. View the Master Notebook & Reports
- **Jupyter Notebook:** Open [`./ConnectTel_Customer_Churn_Prediction.ipynb`](./ConnectTel_Customer_Churn_Prediction.ipynb) in VS Code, JupyterLab, or Google Colab. All cells, plots, and metrics are pre-executed!
- **Executive PDF Report:** Open [`./ConnectTel_Business_Retention_Report.pdf`](./ConnectTel_Business_Retention_Report.pdf).
- **Text Summary:** Read [`./CONNECTTEL_CHURN_REPORT.md`](./CONNECTTEL_CHURN_REPORT.md).

### 2. Run Automated Production & API Tests
```bash
# Run the automated verification harness (both from root or production/)
python test_production.py
python test_api.py
```
Expected output:
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

And for `test_api.py`:
```
=================================================================
  CONNECTTEL REST API AUTOMATED VERIFICATION SUITE
=================================================================
[PASS] Test 01: /health endpoint verified (HTTP 200)
[PASS] Test 02: Root / endpoint verified (HTTP 200)
[PASS] Test 03: High-risk scoring verified (Prob: 73.15%, Tier: CRITICAL RISK)
[PASS] Test 04: Low-risk scoring verified (Prob: 1.54%, Tier: LOW RISK)
[PASS] Test 05: Wrapped payload format verified (HTTP 200)
[PASS] Test 06: List payload scoring verified (2 scored)
[PASS] Test 07: /batch_predict endpoint verified (2 scored)
[PASS] Test 08: Error handling for empty payload verified (HTTP 400)
[PASS] Test 09: GET /predict documentation verified (HTTP 200)
[PASS] Test 10: Live network socket HTTP test verified (HTTP 200)

Ran 10 tests in 0.507s - OK (100% Success)
=================================================================
```

### 3. Score Subscribers via CLI
```bash
cd production
python predict.py --input sample_customers.json
```

### 4. Launch Production REST API
```bash
cd production
python app.py
```
API runs on `http://localhost:8080` with `/health`, `/predict`, and `/batch_predict` endpoints.
