# ConnectTel Customer Churn Predictive Analysis & Production Retention Report

Client: ConnectTel Telecommunications
Domain: Artificial Intelligence & Machine Learning
Project: End-to-End Customer Churn Classification & Business Interpretation
Status: Production-Ready & Leakage-Free Verified
Model Evaluation: 5-Fold Cross-Validation ROC-AUC = 0.8485 | Holdout Test-Set ROC-AUC = 0.8457 (XGBoost)
Production SLA Latency: 12.98 ms (< 50ms requirement)
Net Annual Revenue Saved: $399,794 (Cohort) | $5.68 Million (Estimated Enterprise Scenario Projection)

================================================================================
1. EXECUTIVE SUMMARY & BUSINESS PROBLEM
================================================================================
ConnectTel is experiencing severe quarterly revenue loss due to customer churn. 
In the telecommunications sector, the cost of acquiring a new customer is 4x to 6x 
higher than retaining an existing subscriber.

This project delivers an enterprise-grade machine learning solution that proactively 
identifies customers at high risk of cancelling their service. This allows ConnectTel's 
retention squad to intervene with targeted loyalty discounts and plan upgrades 
before customers leave for competitors.

Key Metrics Overview:
- Total Customers Analyzed: 7,043
- Overall Customer Churn Rate: 26.5% (1,869 churned customers)
- Best Performing Model: XGBoost Classifier (5-Fold CV AUC: 0.8485 | Test AUC: 0.8457)
- Churn Detection Recall at Retention Threshold (tau = 0.28): 79.14% (~80% churners caught)

================================================================================
2. KEY FINDINGS & HYPOTHESIS TESTING
================================================================================

HYPOTHESIS 1: Do Fiber Optic users churn more than DSL users?
- Statistical Result: Statistically Significant (Chi-Square: 328.26, p-value < 0.0001, df = 1)
- Fiber Optic Churn Rate: 41.9%
- DSL Internet Churn Rate: 19.0%
- No Internet Churn Rate: 7.4%
- Business Insight: Even though Fiber Optic generates higher monthly revenue ($80 to 
  $100+/month), customer dissatisfaction and high prices make it ConnectTel's single 
  largest source of revenue loss.

HYPOTHESIS 2: Does having a partner reduce customer churn?
- Statistical Result: Statistically Significant (Chi-Square: 158.73, p-value < 0.0001, df = 1)
- Customers Without Partner: 33.0% churn rate
- Customers With Partner: 19.7% churn rate (40% relative risk reduction)
- Business Insight: Partnered and multi-person households rely on shared home internet 
  and streaming services, creating much higher retention stability.

ADDITIONAL CRITICAL BUSINESS DRIVERS:
1. Contract Commitment:
   - Month-to-Month contracts: 42.7% churn rate
   - 1-Year contracts: 11.3% churn rate
   - 2-Year contracts: 2.8% churn rate
2. Payment Channel:
   - Electronic Check payments: 45.3% churn rate
   - Credit Card / Bank Transfer Auto-Pay: 16.7% churn rate
3. Service Bundling:
   - Customers subscribing to 3 or more add-on services (Online Security, Tech Support, 
     Cloud Backup) churn at less than 15%.

================================================================================
3. MODEL BENCHMARK & DECISION THRESHOLD ANALYSIS
================================================================================
Models trained with zero data leakage using StandardScaler inside 5-Fold Stratified CV:

-----------------------------------------------------------------------------------------
Model Architecture         | 5-Fold CV AUC | Test AUC | Accuracy | Precision | Recall    
-----------------------------------------------------------------------------------------
Logistic Regression (Tuned)| 0.8457        | 0.8408   | 73.88%   | 50.52%    | 78.34%    
Random Forest (Tuned)      | 0.8447        | 0.8423   | 74.88%   | 51.72%    | 80.48%    
XGBoost Classifier (Best)  | 0.8485        | 0.8457   | 79.99%   | 66.08%    | 50.53%    
-----------------------------------------------------------------------------------------

RESOLVING DECISION THRESHOLDS IN PRODUCTION:
In churn prediction, default binary classification at threshold tau = 0.50 optimizes 
overall accuracy (79.99%) and precision (66.08%), but catches only 50.53% of churners.
Because missing a churned subscriber costs $893.30/yr while a retention discount costs 
only $10-$15/month, our production pipeline optimizes the threshold for retention:

--------------------------------------------------------------------------------
Threshold Setting          | Accuracy | Precision | Recall    | F1-Score | Business Application
--------------------------------------------------------------------------------
Default (tau = 0.50)       | 79.99%   | 66.08%    | 50.53%    | 0.5727   | Conservative; minimizes false alarms
F1-Optimal (tau = 0.35)    | 77.22%   | 56.33%    | 73.80%    | 0.6389   | Peak harmonic balance
Retention (tau = 0.28)     | 74.38%   | 51.12%    | 79.14%    | 0.6212   | Production Standard (Catches 8 of 10)
--------------------------------------------------------------------------------

================================================================================
4. VERIFIED SHAP FEATURE ATTRIBUTION vs. BUSINESS CONCLUSIONS
================================================================================
Extracted using shap.TreeExplainer directly from the tuned XGBoost model:

Feature Ranking by Mean |SHAP| Impact:
1. tenure (0.5134) - Longer tenure strongly decreases churn. First 12 months require highest care.
2. Contract_Two year (0.4709) - 2-year contract strongly decreases churn.
3. InternetService_Fiber optic (0.3403) - Fiber optic strongly increases churn.
4. Contract_One year (0.2052) - 1-year contract reduces churn by 73%.
5. PaymentMethod_Electronic check (0.1876) - Electronic check strongly increases churn.
6. TotalChargesPerTenure (0.0987) - High spend velocity increases churn.
7. OnlineSecurity_Yes (0.0983) - Security add-on strongly decreases churn.
8. TechSupport_Yes (0.0789) - Tech support access strongly decreases churn.

CLIENT QUESTION ANSWERED WITH EMPIRICAL EVIDENCE:
"Is Month-to-Month contract the biggest driver of churn, or is it the high Monthly Charge?"

Finding: Contract commitment is the dominant structural driver. The combined impact 
of 2-year (0.4709) and 1-year (0.2052) contract features equals 0.6761 mean SHAP impact. 
High monthly charges act as an ACCELERATOR: when a customer has high monthly spend 
WITHOUT a contract, churn exceeds 65%. On 2-year contracts, customers rarely churn even 
with high monthly spend.

================================================================================
5. RIGOROUS FINANCIAL IMPACT & REVENUE PROTECTION CALCULATIONS
================================================================================
Mathematical Formula:
Net Revenue Saved = (Total Churners * Recall * Retention Success Rate * ARR_per_churner) - 
                    (Retained Customers * Annual Incentive Cost)

Important Note on Projections:
The $5.68M revenue protection figure is an ESTIMATED ENTERPRISE SCENARIO PROJECTION 
modeled across a 100,000 subscriber base, not a guaranteed saving.

1. Exact Cohort Calculation (7,043 Accounts):
- Total Churned Subscribers: 1,869 accounts (26.54% churn rate)
- Average Monthly Charge per Churned Subscriber: $74.44 / month
- Average ARR per Churned Subscriber: $893.30 ($74.44 * 12)
- Total Annual Revenue Lost to Churn in Cohort: $1,669,570
- Flagged at tau = 0.28 (79.14% Recall): 1,479 churners identified
- Baseline Retention Success Rate: 35.0%
- Number of Subscribers Saved: 1,479 * 0.35 = 517 subscribers saved
- Gross Revenue Protected: 517 * $893.30 = $461,834
- Cost of Retention Incentives ($10/mo credit for 12 mo = $120/yr): 517 * $120 = $62,040
- Net Annual Revenue Saved (Cohort): $461,834 - $62,040 = $399,794

2. Enterprise Scenario Projection (100,000 ConnectTel Subscriber Base):
- Annual Churn Pool (26.54%): 26,535 subscribers
- Annual Revenue at Risk: 26,535 * $893.30 = $23,703,715 ($23.7 Million)
- Churners Flagged by Engine (79.14% Recall): 21,000 subscribers
- Subscribers Retained (35% Success): 7,350 subscribers
- Gross Annual Revenue Protected: 7,350 * $893.30 = $6,565,755 ($6.57 Million)
- Retention Program Budget ($120/subscriber): 7,350 * $120 = $882,000
- Estimated Net Annual Revenue Protection: $6,565,755 - $882,000 = $5,683,755 (~$5.68 Million Projection)

SENSITIVITY MATRIX (Net Projected Savings on 100k Base):
- 25% Success Rate | $10/mo Credit: $4,060,000 | $15/mo Credit: $3,745,000
- 35% Success Rate | $10/mo Credit: $5,684,000 | $15/mo Credit: $5,243,000
- 50% Success Rate | $10/mo Credit: $8,120,000 | $15/mo Credit: $7,490,000

================================================================================
6. PRODUCTION-READY DEPLOYMENT & VERIFICATION EVIDENCE
================================================================================
The production codebase contains concrete artifacts verified by automated tests:
1. Serialized Pipeline: production/churn_pipeline.joblib (Loads in 2 ms)
2. Standalone CLI: production/predict.py (Batch scoring JSON/CSV)
3. REST API Service: production/app.py (Flask microservice with /health and /predict)
4. Automated Test Harness: production/test_production.py
5. Pinned Dependencies: production/requirements.txt
6. Containerization: production/Dockerfile
7. Architecture Guide: DEPLOYMENT.md

VERIFICATION TEST RESULTS:
- Artifact Integrity Test: [PASS] All 7 files verified
- Deterministic Prediction Test: [PASS] High Risk = 77.39% / Low Risk = 1.71%
- Latency Benchmark Test: [PASS] 12.98 ms (exceeds < 50ms SLA)
- REST API Integration Test: [PASS] HTTP 200 on /health, /predict, /batch_predict
STATUS: 100% PRODUCTION VERIFIED

================================================================================
7. ACTIONABLE BUSINESS RETENTION RECOMMENDATIONS
================================================================================
1. Annual Contract Migration Campaign:
   - Target: Month-to-Month customers in months 3-9 with elevated churn risk.
   - Action: Offer guaranteed $10/month bill discount or free streaming add-on for 
     signing a 12-month agreement. Migrating to annual terms reduces churn by 73%.

2. Fiber Optic "Value Assurance" Program:
   - Target: All Fiber Optic internet subscribers (current churn: 41.9%).
   - Action: Include complimentary Premium Tech Support and Online Security for the 
     first 6 months, and conduct automated speed diagnostics to prevent friction.

3. Auto-Pay Payment Channel Incentivization:
   - Target: Customers paying via Electronic Check (current churn: 45.3%).
   - Action: Provide a one-time $25 bill credit for enrolling in Automated Bank Debit 
     or Credit Card Auto-Pay (segments with only 16.7% churn).

4. First 90-Day New Customer Shield:
   - Target: New accounts during their first 90 days.
   - Action: Implement automated 30, 60, and 90-day check-ins and dedicated onboarding 
     callbacks to resolve issues before churn occurs.
