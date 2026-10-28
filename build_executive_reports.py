import os
import base64
import subprocess
import shutil

WORKSPACE_DIR = r"C:\Users\NAMAN\.gemini\antigravity\scratch\telecom-churn-connecttel"
DESKTOP_DIR = r"C:\Users\NAMAN\OneDrive\Desktop\telecom-churn-connecttel"
IMAGES_DIR = os.path.join(WORKSPACE_DIR, "images")

def get_base64_image(image_name):
    path = os.path.join(IMAGES_DIR, image_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

img_churn = get_base64_image("eda_churn_distribution.png")
img_fiber = get_base64_image("eda_hypothesis_fiber_vs_dsl.png")
img_partner = get_base64_image("eda_hypothesis_partner.png")
img_services = get_base64_image("eda_demographics_services.png")
img_fin = get_base64_image("eda_financial_tenure.png")
img_cm = get_base64_image("model_confusion_matrices.png")
img_roc = get_base64_image("model_roc_curves.png")
img_shap_sum = get_base64_image("shap_summary_plot.png")
img_shap_imp = get_base64_image("shap_feature_importance.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ConnectTel - Customer Churn Predictive Analysis & Production Retention Report</title>
<style>
  @page {{
    size: A4;
    margin: 16mm 14mm 16mm 14mm;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #1f2937;
    background-color: #ffffff;
    line-height: 1.55;
    margin: 0;
    padding: 24px;
  }}
  .header {{
    border-bottom: 3px solid #1e40af;
    padding-bottom: 14px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  .title-area h1 {{
    font-size: 24px;
    color: #1e3a8a;
    margin: 0 0 4px 0;
    font-weight: 800;
    letter-spacing: -0.5px;
  }}
  .title-area h2 {{
    font-size: 14px;
    color: #4b5563;
    margin: 0;
    font-weight: 500;
  }}
  .badge-area {{
    text-align: right;
  }}
  .badge {{
    background: #059669;
    color: #ffffff;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 0.5px;
  }}
  .meta-box {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 20px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }}
  .meta-item .meta-label {{
    font-size: 10.5px;
    text-transform: uppercase;
    color: #64748b;
    font-weight: 700;
    letter-spacing: 0.5px;
  }}
  .meta-item .meta-val {{
    font-size: 13px;
    color: #0f172a;
    font-weight: 600;
    margin-top: 2px;
  }}
  h2.section-title {{
    font-size: 17px;
    color: #1e3a8a;
    border-left: 4px solid #2563eb;
    padding-left: 10px;
    margin: 24px 0 12px 0;
    font-weight: 700;
  }}
  h3.sub-title {{
    font-size: 14px;
    color: #334155;
    margin: 16px 0 6px 0;
    font-weight: 600;
  }}
  p {{
    font-size: 13px;
    color: #334155;
    margin: 0 0 10px 0;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 14px 0 20px 0;
  }}
  .kpi-card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #2563eb;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .kpi-card.danger {{
    border-top-color: #ef4444;
  }}
  .kpi-card.success {{
    border-top-color: #10b981;
  }}
  .kpi-card.warning {{
    border-top-color: #f59e0b;
  }}
  .kpi-val {{
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
    margin: 3px 0;
  }}
  .kpi-desc {{
    font-size: 11px;
    color: #64748b;
    font-weight: 500;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 20px 0;
    font-size: 12.5px;
  }}
  th, td {{
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
  }}
  th {{
    background-color: #f1f5f9;
    color: #1e293b;
    font-weight: 700;
    border-top: 1px solid #cbd5e1;
  }}
  tr:hover {{
    background-color: #f8fafc;
  }}
  .highlight-cell {{
    font-weight: 700;
    color: #1e40af;
    background-color: #eff6ff;
  }}
  .img-container {{
    margin: 16px 0;
    text-align: center;
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px;
    page-break-inside: avoid;
  }}
  .img-container img {{
    max-width: 100%;
    height: auto;
    border-radius: 4px;
  }}
  .img-caption {{
    font-size: 11.5px;
    color: #6b7280;
    margin-top: 6px;
    font-weight: 500;
  }}
  .recommendation-card {{
    background: #f8fafc;
    border-left: 4px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    page-break-inside: avoid;
  }}
  .recommendation-card h4 {{
    margin: 0 0 4px 0;
    font-size: 14px;
    color: #065f46;
    font-weight: 700;
  }}
  .recommendation-card p {{
    margin: 0;
    font-size: 12.5px;
    color: #334155;
  }}
  .callout {{
    background-color: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 12px 14px;
    border-radius: 0 6px 6px 0;
    margin: 12px 0;
    font-size: 12.5px;
    color: #92400e;
  }}
  .callout-success {{
    background-color: #ecfdf5;
    border-left: 4px solid #10b981;
    padding: 12px 14px;
    border-radius: 0 6px 6px 0;
    margin: 12px 0;
    font-size: 12.5px;
    color: #065f46;
  }}
  .code-badge {{
    background: #f1f5f9;
    color: #0f172a;
    font-family: monospace;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11.5px;
    border: 1px solid #cbd5e1;
  }}
  .footer {{
    margin-top: 30px;
    border-top: 1px solid #e2e8f0;
    padding-top: 12px;
    text-align: center;
    font-size: 11px;
    color: #94a3b8;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="title-area">
    <h1>ConnectTel Telecommunications</h1>
    <h2>Predictive Customer Churn Model & Production Retention Strategy Report</h2>
  </div>
  <div class="badge-area">
    <span class="badge">Production Verified</span>
  </div>
</div>

<div class="meta-box">
  <div class="meta-item">
    <div class="meta-label">Client</div>
    <div class="meta-val">ConnectTel Inc.</div>
  </div>
  <div class="meta-item">
    <div class="meta-label">Domain</div>
    <div class="meta-val">AI & Machine Learning</div>
  </div>
  <div class="meta-item">
    <div class="meta-label">Model Evaluation</div>
    <div class="meta-val">CV: 0.8485 | Test: 0.8457</div>
  </div>
  <div class="meta-item">
    <div class="meta-label">Verified SLA Latency</div>
    <div class="meta-val">12.98 ms (&lt; 50ms)</div>
  </div>
</div>

<div class="kpi-grid">
  <div class="kpi-card danger">
    <div class="kpi-desc">Overall Churn Rate</div>
    <div class="kpi-val">26.5%</div>
    <div class="kpi-desc">1,869 churners in cohort</div>
  </div>
  <div class="kpi-card danger">
    <div class="kpi-desc">Fiber Optic Churn</div>
    <div class="kpi-val">41.9%</div>
    <div class="kpi-desc">vs 19.0% for DSL</div>
  </div>
  <div class="kpi-card success">
    <div class="kpi-desc">Holdout Test ROC-AUC</div>
    <div class="kpi-val">0.8457</div>
    <div class="kpi-desc">5-Fold CV AUC: 0.8485</div>
  </div>
  <div class="kpi-card success">
    <div class="kpi-desc">Retention Recall (&tau;=0.28)</div>
    <div class="kpi-val">79.14%</div>
    <div class="kpi-desc">Catches 8 of 10 churners</div>
  </div>
</div>

<h2 class="section-title">1. Executive Summary & Business Objective</h2>
<p>
<strong>ConnectTel</strong> is experiencing significant quarterly revenue erosion due to customer churn. In the telecommunications sector, acquiring a new subscriber costs <strong>4x to 6x more</strong> than retaining an existing one. 
</p>
<p>
This project delivers a complete, production-ready machine learning system that identifies subscribers at imminent risk of cancelling. By flagging high-risk customers before they cancel, ConnectTel's retention task force can deploy targeted loyalty offers (annual discounts, complimentary tech support, payment method upgrades) to preserve long-term Customer Lifetime Value (LTV).
</p>

<div class="img-container">
  <img src="data:image/png;base64,{img_churn}" alt="Overall Churn Distribution" />
  <div class="img-caption">Figure 1: Customer Retention vs. Churn Breakdown across 7,043 Accounts</div>
</div>

<h2 class="section-title">2. Key Findings & Hypothesis Testing</h2>

<h3 class="sub-title">Hypothesis 1: Do Fiber Optic users churn more than DSL users?</h3>
<p>
<strong>Finding: YES (Statistically Significant, Chi-Square = 328.26, p &lt; 0.0001, df = 1).</strong><br>
Customers on Fiber Optic internet churn at <strong>41.9%</strong>, compared to only <strong>19.0%</strong> for DSL subscribers and <strong>7.4%</strong> for customers without internet service. While Fiber Optic commands higher monthly prices ($80 to $100+/mo), service friction and price sensitivity make it ConnectTel's primary source of revenue loss.
</p>

<div class="img-container">
  <img src="data:image/png;base64,{img_fiber}" alt="Fiber Optic vs DSL Churn" />
  <div class="img-caption">Figure 2: Churn Rate Comparison across Internet Service Types (Chi-Square: 328.26, p &lt; 0.0001)</div>
</div>

<h3 class="sub-title">Hypothesis 2: Does having a partner reduce customer churn?</h3>
<p>
<strong>Finding: YES (Statistically Significant, Chi-Square = 158.73, p &lt; 0.0001, df = 1).</strong><br>
Customers without a partner churn at <strong>33.0%</strong>, whereas customers with a partner churn at only <strong>19.7%</strong> (a 40% reduction in churn risk). Multi-person households share internet and streaming dependencies, creating substantial household stickiness.
</p>

<div class="img-container">
  <img src="data:image/png;base64,{img_partner}" alt="Partner Status vs Churn" />
  <div class="img-caption">Figure 3: Churn Rate by Partner Status (Chi-Square: 158.73, p &lt; 0.0001)</div>
</div>

<div class="img-container">
  <img src="data:image/png;base64,{img_services}" alt="Contract and Payment Impact" />
  <div class="img-caption">Figure 4: Churn Rate by Contract Commitment, Payment Channel, and Service Bundles</div>
</div>

<h2 class="section-title">3. Model Benchmark & Transparent Threshold Analysis</h2>
<p>
Three distinct model architectures were trained using a <strong>leakage-free Scikit-Learn Pipeline</strong> (integrating StandardScaler inside the cross-validation folds) with <strong>5-Fold Stratified Cross-Validation</strong> and <strong>GridSearchCV</strong> hyperparameter tuning:
</p>

<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>5-Fold CV ROC-AUC</th>
      <th>Holdout Test ROC-AUC</th>
      <th>Test Accuracy</th>
      <th>Test Precision</th>
      <th>Test Recall (&tau;=0.50)</th>
      <th>Test F1-Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Logistic Regression (Tuned)</strong></td>
      <td>0.8457</td>
      <td>0.8408</td>
      <td>73.88%</td>
      <td>50.52%</td>
      <td>78.34%</td>
      <td>0.6143</td>
    </tr>
    <tr>
      <td><strong>Random Forest (Tuned)</strong></td>
      <td>0.8447</td>
      <td>0.8423</td>
      <td>74.88%</td>
      <td>51.72%</td>
      <td>80.48%</td>
      <td>0.6297</td>
    </tr>
    <tr class="highlight-cell">
      <td><strong>XGBoost Classifier (Best Overall)</strong></td>
      <td><strong>0.8485</strong></td>
      <td><strong>0.8457</strong></td>
      <td><strong>79.99%</strong></td>
      <td><strong>66.08%</strong></td>
      <td>50.53%</td>
      <td><strong>0.5727</strong></td>
    </tr>
  </tbody>
</table>

<h3 class="sub-title">Resolving the Decision Threshold in Production</h3>
<p>
In customer churn, default binary classification at threshold &tau; = 0.50 optimizes raw accuracy (79.99%) and precision (66.08%) but misses 49.5% of churners because of the class imbalance (26.5% positive prevalence). 
Because missing an exiting customer costs <strong>$893.30/year</strong> while extending an offer costs only <strong>$10–$15/month</strong>, our production pipeline optimizes the threshold for retention economics:
</p>

<table>
  <thead>
    <tr>
      <th>Operating Threshold (&tau;)</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall (Catch Rate)</th>
      <th>F1-Score</th>
      <th>Business Application</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Default (&tau; = 0.50)</strong></td>
      <td><strong>79.99%</strong></td>
      <td><strong>66.08%</strong></td>
      <td>50.53%</td>
      <td>0.5727</td>
      <td>Conservative; minimizes false alarms.</td>
    </tr>
    <tr>
      <td><strong>F1-Optimal (&tau; = 0.35)</strong></td>
      <td>77.22%</td>
      <td>56.33%</td>
      <td>73.80%</td>
      <td><strong>0.6389</strong></td>
      <td>Peak harmonic balance.</td>
    </tr>
    <tr class="highlight-cell">
      <td><strong>Retention-Optimal (&tau; = 0.28)</strong></td>
      <td>74.38%</td>
      <td>51.12%</td>
      <td><strong>79.14% (~80%)</strong></td>
      <td>0.6212</td>
      <td><strong>Production Standard</strong>: Catches 8 of 10 churners before departure.</td>
    </tr>
  </tbody>
</table>

<div class="img-container">
  <img src="data:image/png;base64,{img_roc}" alt="ROC Curves Comparison" />
  <div class="img-caption">Figure 5: Receiver Operating Characteristic (ROC) Curves across all three models</div>
</div>

<div class="img-container">
  <img src="data:image/png;base64,{img_cm}" alt="Confusion Matrix Comparison" />
  <div class="img-caption">Figure 6: Confusion Matrix Heatmaps on the Test Dataset (1,409 accounts)</div>
</div>

<h2 class="section-title">4. Model Explainability (Verified SHAP Feature Attribution)</h2>
<p>
To eliminate the "Black Box" problem, SHAP (SHapley Additive exPlanations) values were extracted directly from the tuned XGBoost model. Every conclusion is strictly verified against the empirical feature importance and beeswarm distributions below:
</p>

<div class="img-container">
  <img src="data:image/png;base64,{img_shap_imp}" alt="SHAP Global Feature Importance" />
  <div class="img-caption">Figure 7: Global Mean Absolute SHAP Importance Ranking</div>
</div>

<div class="img-container">
  <img src="data:image/png;base64,{img_shap_sum}" alt="SHAP Beeswarm Plot" />
  <div class="img-caption">Figure 8: SHAP Beeswarm Plot: Feature Value vs. Directional Churn Impact</div>
</div>

<table>
  <thead>
    <tr>
      <th>Rank</th>
      <th>Feature Name</th>
      <th>Mean |SHAP| Value</th>
      <th>Direction of Impact</th>
      <th>Verified Empirical Conclusion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1</strong></td>
      <td><span class="code-badge">tenure</span></td>
      <td><strong>0.5134</strong></td>
      <td>Higher &rarr; Strongly Lowers Churn</td>
      <td>Customers in their first 12 months have highest vulnerability. Tenure acts as a strong protective anchor.</td>
    </tr>
    <tr>
      <td><strong>2</strong></td>
      <td><span class="code-badge">Contract_Two year</span></td>
      <td><strong>0.4709</strong></td>
      <td>Present &rarr; Strongly Lowers Churn</td>
      <td>Long-term contracts form the #1 structural barrier to churn. Month-to-Month accounts lack this insulation.</td>
    </tr>
    <tr>
      <td><strong>3</strong></td>
      <td><span class="code-badge">InternetService_Fiber optic</span></td>
      <td><strong>0.3403</strong></td>
      <td>Present &rarr; Strongly Raises Churn</td>
      <td>Fiber optic subscribers experience high dissatisfaction/bill shock, driving 41.9% churn.</td>
    </tr>
    <tr>
      <td><strong>4</strong></td>
      <td><span class="code-badge">Contract_One year</span></td>
      <td><strong>0.2052</strong></td>
      <td>Present &rarr; Strongly Lowers Churn</td>
      <td>Even a 1-year agreement reduces churn by 73% relative to Month-to-Month contracts.</td>
    </tr>
    <tr>
      <td><strong>5</strong></td>
      <td><span class="code-badge">PaymentMethod_Electronic check</span></td>
      <td><strong>0.1876</strong></td>
      <td>Present &rarr; Strongly Raises Churn</td>
      <td>Manual electronic check users churn at 45.3% vs. 16.7% for automated card/bank debit.</td>
    </tr>
    <tr>
      <td><strong>6</strong></td>
      <td><span class="code-badge">TotalChargesPerTenure</span></td>
      <td><strong>0.0987</strong></td>
      <td>Higher &rarr; Strongly Raises Churn</td>
      <td>Our engineered spend-velocity metric accurately captures high monthly burn rates.</td>
    </tr>
    <tr>
      <td><strong>7</strong></td>
      <td><span class="code-badge">OnlineSecurity_Yes</span></td>
      <td><strong>0.0983</strong></td>
      <td>Present &rarr; Strongly Lowers Churn</td>
      <td>Value-added security builds high switching friction, protecting accounts.</td>
    </tr>
    <tr>
      <td><strong>8</strong></td>
      <td><span class="code-badge">TechSupport_Yes</span></td>
      <td><strong>0.0789</strong></td>
      <td>Present &rarr; Strongly Lowers Churn</td>
      <td>Active support access resolves technical pain points before cancellation.</td>
    </tr>
  </tbody>
</table>

<div class="callout">
  <strong>Client Question Answered with Empirical Evidence:</strong><br>
  <em>"Is Month-to-Month contract the biggest driver of churn, or is it the high Monthly Charge?"</em><br><br>
  <strong>Finding:</strong> <strong>Contract Commitment is the single dominant driver.</strong> The top two contract features (<span class="code-badge">Contract_Two year</span>: 0.4709 and <span class="code-badge">Contract_One year</span>: 0.2052) combine for <strong>0.6761 mean SHAP impact</strong>, far exceeding individual price metrics. High monthly charges act as an <em>accelerator</em>: customers with high monthly spend on Month-to-Month terms churn at over 65%, whereas customers with identical high spend on 2-year contracts churn at under 4%.
</div>

<h2 class="section-title">5. Rigorous Financial Impact & ROI Calculations</h2>

<h3 class="sub-title">Mathematical Formula</h3>
<p style="text-align: center; font-family: monospace; font-size: 13.5px; background: #f8fafc; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0;">
Net Revenue Saved = (Total Churners &times; Recall &times; Retention Success Rate &times; ARR_per_churner) - (Retained Customers &times; Annual Incentive Cost)
</p>

<div class="callout">
  <strong>Methodology & Scenario Note:</strong> The calculations below provide: (1) The exact verified empirical results for the 7,043 subscriber dataset cohort, and (2) An <strong>Estimated Enterprise Scenario Projection</strong> modeled across a representative 100,000 subscriber base. Enterprise figures are planning projections and not guaranteed returns; actual realization will vary depending on retention outreach contact rates and campaign responsiveness.
</div>

<h3 class="sub-title">1. Exact Dataset Cohort Analysis (7,043 Active Accounts)</h3>
<ul>
  <li>Total Churned Customers in Dataset: <strong>1,869 accounts</strong> (26.54% churn rate)</li>
  <li>Average Monthly Charge of Churned Subscribers: <strong>$74.44 / month</strong></li>
  <li>Average Annual Recurring Revenue (ARR) per Churned Subscriber: <strong>$893.30</strong> ($74.44 &times; 12)</li>
  <li>Total Annual Subscription Revenue Lost in Dataset: <strong>$1,669,570</strong> ($1.67 Million)</li>
  <li>Flagged at &tau; = 0.28 (79.14% Recall): <strong>1,479 churners accurately flagged</strong></li>
  <li>Conservative Retention Campaign Acceptance: <strong>35.0%</strong></li>
  <li>Number of Subscribers Saved: 1,479 &times; 0.35 = <strong>517 customers saved</strong></li>
  <li>Gross Annual Revenue Protected: 517 &times; $893.30 = <strong>$461,834</strong></li>
  <li>Cost of Retention Program ($10/mo credit for 12 months = $120/yr): 517 &times; $120 = <strong>$62,040</strong></li>
  <li><strong>Net Annual Revenue Saved (Cohort):</strong> $461,834 - $62,040 = <strong>$399,794</strong></li>
</ul>

<h3 class="sub-title">2. Enterprise Scenario Projection (100,000 ConnectTel Subscriber Base)</h3>
<ul>
  <li>Annual Churn Pool (26.54%): <strong>26,535 subscribers</strong></li>
  <li>Annual Revenue at Risk: 26,535 &times; $893.30 = <strong>$23,703,715 ($23.7 Million)</strong></li>
  <li>Subscribers Flagged by Model (79.14% Recall): <strong>21,000 subscribers</strong></li>
  <li>Subscribers Successfully Retained (35% Success): <strong>7,350 subscribers</strong></li>
  <li>Gross Annual Revenue Protected: 7,350 &times; $893.30 = <strong>$6,565,755 ($6.57 Million)</strong></li>
  <li>Retention Campaign Incentive Cost ($120/subscriber): 7,350 &times; $120 = <strong>$882,000</strong></li>
  <li><strong>Estimated Net Annual Revenue Protection:</strong> $6,565,755 - $882,000 = <strong>$5,683,755 (~$5.68 Million Projection)</strong></li>
</ul>

<h3 class="sub-title">Sensitivity Matrix: Net Projected Revenue Saved (100k Customer Base)</h3>
<table>
  <thead>
    <tr>
      <th>Retention Success Rate</th>
      <th>$10/mo Credit ($120/yr)</th>
      <th>$15/mo Credit ($180/yr)</th>
      <th>$20/mo Credit ($240/yr)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>25% (Conservative)</strong></td>
      <td>$4,060,000</td>
      <td>$3,745,000</td>
      <td>$3,430,000</td>
    </tr>
    <tr class="highlight-cell">
      <td><strong>35% (Baseline Expected)</strong></td>
      <td><strong>$5,684,000</strong></td>
      <td><strong>$5,243,000</strong></td>
      <td><strong>$4,802,000</strong></td>
    </tr>
    <tr>
      <td><strong>50% (Optimistic Engagement)</strong></td>
      <td>$8,120,000</td>
      <td>$7,490,000</td>
      <td>$6,860,000</td>
    </tr>
  </tbody>
</table>

<h2 class="section-title">6. Production-Ready Deployment & Verification Evidence</h2>
<p>
To demonstrate complete production readiness beyond an experimental notebook, the project includes a production deployment bundle verified through automated test harnesses:
</p>

<table>
  <thead>
    <tr>
      <th>Production Component</th>
      <th>File / Artifact</th>
      <th>Specification & Operational Function</th>
      <th>Verification Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Serialized Model Pipeline</strong></td>
      <td><span class="code-badge">production/churn_pipeline.joblib</span></td>
      <td>End-to-end preprocessing, scaling, alignment, and XGBoost inference</td>
      <td><strong>Verified (Loads in 2ms)</strong></td>
    </tr>
    <tr>
      <td><strong>CLI Scoring Engine</strong></td>
      <td><span class="code-badge">production/predict.py</span></td>
      <td>Batch scoring of CSV/JSON files with configurable threshold</td>
      <td><strong>Verified (CLI Functional)</strong></td>
    </tr>
    <tr>
      <td><strong>REST API Microservice</strong></td>
      <td><span class="code-badge">production/app.py</span></td>
      <td>Flask/Gunicorn service with <span class="code-badge">/health</span>, <span class="code-badge">/predict</span>, <span class="code-badge">/batch_predict</span></td>
      <td><strong>Verified (HTTP 200)</strong></td>
    </tr>
    <tr>
      <td><strong>Automated QA Suite</strong></td>
      <td><span class="code-badge">production/test_production.py</span></td>
      <td>Integration tests checking deterministic bounds, latency &amp; API</td>
      <td><strong>4 of 4 Tests Passed</strong></td>
    </tr>
    <tr>
      <td><strong>Inference Latency SLA</strong></td>
      <td><span class="code-badge">SLA Benchmark</span></td>
      <td>Average inference latency: <strong>12.98 ms</strong> (95th percentile: 13.59 ms)</td>
      <td><strong>Passes (&lt; 50ms SLA)</strong></td>
    </tr>
    <tr>
      <td><strong>Containerization</strong></td>
      <td><span class="code-badge">production/Dockerfile</span></td>
      <td>Docker specification for multi-worker container deployment</td>
      <td><strong>Production Ready</strong></td>
    </tr>
  </tbody>
</table>

<div class="callout-success">
  <strong>Automated Test Suite Output:</strong><br>
  <span style="font-family: monospace;">
    [PASS] All 7 production artifacts verified successfully.<br>
    [PASS] High-Risk Customer correctly scored: Prob=77.39%, Tier=CRITICAL RISK<br>
    [PASS] Low-Risk Customer correctly scored: Prob=1.71%, Tier=LOW RISK<br>
    [PASS] Average Inference Latency: 12.98 ms (Production SLA &lt; 50ms satisfied)<br>
    [PASS] /predict &amp; /batch_predict API responses valid (HTTP 200)<br>
    <strong>ALL 4 PRODUCTION VERIFICATION TESTS PASSED (100% SUCCESS)</strong>
  </span>
</div>

<h2 class="section-title">7. Actionable Business Retention Recommendations</h2>

<div class="recommendation-card">
  <h4>1. Annual Contract Migration Campaign</h4>
  <p>
    <strong>Target:</strong> High-risk Month-to-Month customers in months 3–9.<br>
    <strong>Action:</strong> Offer a guaranteed <strong>$10/month bill discount</strong> or free streaming add-on conditioned on signing a 12-month agreement. Migrating customers to annual terms reduces churn risk by <strong>73%</strong>.
  </p>
</div>

<div class="recommendation-card">
  <h4>2. Fiber Optic "Value Assurance" & Quality Guarantee</h4>
  <p>
    <strong>Target:</strong> All Fiber Optic internet subscribers (41.9% current churn).<br>
    <strong>Action:</strong> Bundle complimentary <strong>Premium Tech Support</strong> and <strong>Online Security</strong> for the first 6 months. Run automated line health diagnostics to fix speed issues before customers call to cancel.
  </p>
</div>

<div class="recommendation-card">
  <h4>3. Auto-Pay Payment Channel Incentivization</h4>
  <p>
    <strong>Target:</strong> Customers paying via Electronic Check (45.3% current churn).<br>
    <strong>Action:</strong> Provide a <strong>one-time $25 bill credit</strong> for enrolling in Automated Bank Debit or Credit Card Auto-Pay, migrating them into a cohort with only 16.7% churn.
  </p>
</div>

<div class="recommendation-card">
  <h4>4. First 90-Day "New Subscriber Onboarding Shield"</h4>
  <p>
    <strong>Target:</strong> New subscribers in their first 90 days of tenure.<br>
    <strong>Action:</strong> Deploy automated 30, 60, and 90-day satisfaction check-ins and dedicated onboarding callbacks to eliminate early friction.
  </p>
</div>

<div class="footer">
  ConnectTel Telecommunications • Predictive Churn Modeling & Production Retention Strategy Report • Confidential
</div>

</body>
</html>
"""

# 1. Write HTML Report
html_file_path = os.path.join(WORKSPACE_DIR, "ConnectTel_Business_Retention_Report.html")
with open(html_file_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"HTML Report generated at: {html_file_path}")

os.makedirs(DESKTOP_DIR, exist_ok=True)
desktop_html_path = os.path.join(DESKTOP_DIR, "ConnectTel_Business_Retention_Report.html")
with open(desktop_html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"HTML Report copied to Desktop: {desktop_html_path}")

# 2. Convert to PDF using Edge Headless
pdf_file_path = os.path.join(WORKSPACE_DIR, "ConnectTel_Business_Retention_Report.pdf")
desktop_pdf_path = os.path.join(DESKTOP_DIR, "ConnectTel_Business_Retention_Report.pdf")

edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
cmd = [
    edge_exe,
    "--headless",
    "--disable-gpu",
    f"--print-to-pdf={pdf_file_path}",
    html_file_path
]
print("Converting HTML to PDF via Microsoft Edge Headless...")
subprocess.run(cmd, check=True)

shutil.copyfile(pdf_file_path, desktop_pdf_path)
print(f"PDF successfully generated at Desktop: {desktop_pdf_path}")

# 3. Create Clean Text Report (CONNECTTEL_CHURN_REPORT.md)
clean_md = """# ConnectTel Customer Churn Predictive Analysis & Production Retention Report

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
"""

clean_md_path = os.path.join(WORKSPACE_DIR, "CONNECTTEL_CHURN_REPORT.md")
with open(clean_md_path, "w", encoding="utf-8") as f:
    f.write(clean_md)

desktop_md_path = os.path.join(DESKTOP_DIR, "CONNECTTEL_CHURN_REPORT.md")
with open(desktop_md_path, "w", encoding="utf-8") as f:
    f.write(clean_md)
print(f"Cleaned MD Report written to Desktop: {desktop_md_path}")

# Copy production directory to Desktop (ignoring cache)
desktop_prod_dir = os.path.join(DESKTOP_DIR, "production")
shutil.copytree(
    os.path.join(WORKSPACE_DIR, "production"), 
    desktop_prod_dir, 
    dirs_exist_ok=True, 
    ignore=shutil.ignore_patterns('__pycache__', '*.pyc')
)
print(f"Production folder copied to Desktop: {desktop_prod_dir}")

# Copy DEPLOYMENT.md and README.md to Desktop
shutil.copyfile(
    os.path.join(WORKSPACE_DIR, "DEPLOYMENT.md"),
    os.path.join(DESKTOP_DIR, "DEPLOYMENT.md")
)
shutil.copyfile(
    os.path.join(WORKSPACE_DIR, "README.md"),
    os.path.join(DESKTOP_DIR, "README.md")
)
shutil.copyfile(
    os.path.join(WORKSPACE_DIR, "ConnectTel_Customer_Churn_Prediction.ipynb"),
    os.path.join(DESKTOP_DIR, "ConnectTel_Customer_Churn_Prediction.ipynb")
)
print("Copied DEPLOYMENT.md, README.md, and executed notebook to Desktop")

# Re-zip desktop folder
zip_path = r"C:\Users\NAMAN\OneDrive\Desktop\telecom-churn-connecttel.zip"
cmd_zip = [
    "powershell",
    "-Command",
    f"Compress-Archive -Path '{DESKTOP_DIR}' -DestinationPath '{zip_path}' -Force"
]
subprocess.run(cmd_zip, check=True)
print("ZIP file updated with production suite, PDF, and clean reports!")
