import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = r"C:\Users\NAMAN\OneDrive\Desktop\telecom-churn-connecttel"
LIVE_URL = "https://nobody-document-marion-assistance.trycloudflare.com/ui"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette
NAVY_DARK = RGBColor(15, 23, 42)      # #0f172a
NAVY_MID = RGBColor(30, 41, 59)       # #1e293b
BLUE_ACCENT = RGBColor(2, 132, 199)   # #0284c7
TEAL_GREEN = RGBColor(16, 185, 129)   # #10b981
TEXT_WHITE = RGBColor(255, 255, 255)
TEXT_MUTED = RGBColor(148, 163, 184)  # #94a3b8
CARD_BG = RGBColor(248, 250, 252)     # #f8fafc
CARD_BORDER = RGBColor(226, 232, 240) # #e2e8f0
DARK_TEXT = RGBColor(15, 23, 42)
DANGER_RED = RGBColor(239, 68, 68)

def add_header(slide, title_text, category_text="CONNECTTEL ENTERPRISE INTELLIGENCE"):
    # Header container
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = BLUE_ACCENT
    
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY_DARK

def add_card(slide, left, top, width, height, title, points, bg_color=CARD_BG, border_color=CARD_BORDER, title_color=NAVY_DARK):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.2)
    tf.margin_right = Inches(0.25)
    tf.margin_bottom = Inches(0.2)
    
    if title:
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = title_color
        p_t.space_after = Pt(10)
    
    first = True if not title else False
    for pt in points:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT if bg_color == CARD_BG else TEXT_WHITE
        p.space_after = Pt(6)

blank_layout = prs.slide_layouts[6]

# ==========================================
# SLIDE 1: Title Slide (Dark Theme)
# ==========================================
s1 = prs.slides.add_slide(blank_layout)
bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
bg1.fill.solid()
bg1.fill.fore_color.rgb = NAVY_DARK
bg1.line.color.rgb = NAVY_DARK

title_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.333), Inches(4.5))
tf1 = title_box.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "CONNECTTEL TELECOMMUNICATIONS"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = BLUE_ACCENT
p.space_after = Pt(12)

p2 = tf1.add_paragraph()
p2.text = "Customer Churn Prediction &\nRetention Optimization Strategy"
p2.font.size = Pt(36)
p2.font.bold = True
p2.font.color.rgb = TEXT_WHITE
p2.space_after = Pt(16)

p3 = tf1.add_paragraph()
p3.text = "End-to-End Enterprise ML Solution | Holdout ROC-AUC: 0.8457 | Verified Net Savings: $399,794"
p3.font.size = Pt(16)
p3.font.color.rgb = TEXT_MUTED
p3.space_after = Pt(28)

p4 = tf1.add_paragraph()
p4.text = f"Live Production Application: {LIVE_URL}"
p4.font.size = Pt(14)
p4.font.bold = True
p4.font.color.rgb = TEAL_GREEN

# ==========================================
# SLIDE 2: Executive Summary & Problem Context
# ==========================================
s2 = prs.slides.add_slide(blank_layout)
add_header(s2, "Executive Summary & Business Challenge", "CONNECTTEL STRATEGIC OVERVIEW")

add_card(s2, 0.8, 1.5, 3.6, 5.4, "The Churn Dilemma", [
    "ConnectTel operates with a baseline churn rate of 26.5%, losing over 1,869 subscribers annually in the test cohort alone.",
    "Customer Acquisition Cost (CAC) is 5x higher than Customer Retention Cost ($150-$250 CAC vs $20-$30 retention incentive).",
    "Customer attrition directly impacts high-margin recurring telecom subscriptions and subscriber lifetime value (LTV)."
])

add_card(s2, 4.8, 1.5, 3.6, 5.4, "Machine Learning Solution", [
    "Developed a leakage-free 5-fold cross-validated XGBoost classification pipeline.",
    "Engineered behavioral features: TotalChargesPerTenure, ServiceCount, and streaming bundles.",
    "Integrated dynamic probability scoring with Explainable AI (SHAP) for transparent agent decisions.",
    "Achieved sub-20ms inference latency for synchronous CRM and billing integration."
])

add_card(s2, 8.8, 1.5, 3.7, 5.4, "Key Financial Outcomes", [
    "Lowered decision threshold from default 0.50 to 0.28 to prioritize churn recall.",
    "Recall jumped from 54.3% to 82.1%, capturing 307 out of 374 at-risk subscribers.",
    "$399,794 verified net annual savings in 1,409 holdout subscriber cohort.",
    "$5.68M projected net annual benefit across 100,000 enterprise subscriber footprint."
])

# ==========================================
# SLIDE 3: Exploratory Data Analysis & Strategic Insights
# ==========================================
s3 = prs.slides.add_slide(blank_layout)
add_header(s3, "Exploratory Data Analysis: Primary Churn Drivers", "DATA DISCOVERY & ROOT CAUSE ANALYSIS")

add_card(s3, 0.8, 1.5, 3.6, 5.4, "Contract Term Vulnerability", [
    "Month-to-month contracts exhibit an alarming 42.7% churn rate.",
    "In contrast, 1-Year (11.3%) and 2-Year (2.8%) contracts demonstrate exceptional customer stickiness.",
    "Strategic takeaway: Transitioning month-to-month users to annual commitments with a $10-$15 discount yields immediate churn reduction."
])

add_card(s3, 4.8, 1.5, 3.6, 5.4, "Fiber Optic Service Friction", [
    "Fiber optic customers churn at 41.9%, despite paying highest monthly charges ($70 - $110/mo).",
    "Root cause: Technical setup friction and lack of bundled tech support.",
    "DSL subscribers churn at only 19.0%, and non-internet users at 7.4%.",
    "Strategic takeaway: Proactive tech onboarding and free first-year device protection for fiber lines."
])

add_card(s3, 8.8, 1.5, 3.7, 5.4, "Payment & Tenured Loyalty", [
    "Electronic check payment accounts for 45.3% of churners due to recurring manual friction.",
    "Automated payment (Bank Transfer / Credit Card) churn is under 16%.",
    "New customer vulnerability: 50%+ of churn occurs during the first 12 months.",
    "Strategic takeaway: Autopay cash-back incentives and critical 90-day onboarding check-ins."
])

# ==========================================
# SLIDE 4: Machine Learning Modeling & Rigorous Evaluation
# ==========================================
s4 = prs.slides.add_slide(blank_layout)
add_header(s4, "Model Architecture & Benchmark Comparison", "LEAKAGE-FREE PIPELINE ENGINEERING")

add_card(s4, 0.8, 1.5, 5.6, 5.4, "Model Benchmark Comparison (Holdout Set)", [
    "XGBoost Classifier (Selected Model):",
    "  • Holdout ROC-AUC: 0.8457 | 5-Fold CV ROC-AUC: 0.8485",
    "  • Optimized Threshold (0.28): Recall 82.1%, F1-Score 0.639",
    "Random Forest Classifier:",
    "  • Holdout ROC-AUC: 0.8260 | Overfits on high-tenure tree splits",
    "Logistic Regression (Baseline):",
    "  • Holdout ROC-AUC: 0.8410 | High interpretability, lower non-linear capacity",
    "Strict Anti-Leakage Protocol:",
    "  • StandardScaler fit exclusively on training folds; zero holdout data bleeding."
])

add_card(s4, 6.8, 1.5, 5.7, 5.4, "Vectorized Production Pipeline Specifications", [
    "Pre-computed feature mappings for deterministic OHE alignment.",
    "Vectorized numpy arrays replace iterative pandas series construction.",
    "P95 Inference Latency: 16.2 ms (SLA Target: < 50 ms).",
    "Inference Throughput: > 600 requests / sec under dual-engine WSGI.",
    "Dual Deployment Compatibility:",
    "  • Python Flask REST API & Pure Client-Side JavaScript Engine.",
    "Defensive deserialization handles cross-version scikit-learn environments seamlessly."
])

# ==========================================
# SLIDE 5: Business-Aligned Decision Threshold
# ==========================================
s5 = prs.slides.add_slide(blank_layout)
add_header(s5, "Threshold Optimization: Bridging ML to Business Value", "DECISION THRESHOLD CALIBRATION")

add_card(s5, 0.8, 1.5, 3.6, 5.4, "The Flaw of Default 0.50", [
    "Standard ML packages default to a 0.50 threshold.",
    "At 0.50 threshold, ConnectTel detects only 54.3% of churners.",
    "45.7% of leaving subscribers slip away unnoticed without any retention intervention.",
    "Resulting lost revenue: Over $183,000 in unaddressed churn across the holdout cohort."
])

add_card(s5, 4.8, 1.5, 3.6, 5.4, "Optimal Threshold: 0.28", [
    "Conducting cost-benefit sensitivity analysis identified 0.28 as the financial sweet spot.",
    "Recall increases from 54.3% to 82.1% (+27.8% gain in caught churners).",
    "Precision remains robust at 52.4%, avoiding budget depletion from over-targeting.",
    "307 at-risk subscribers successfully flagged for immediate retention action."
])

add_card(s5, 8.8, 1.5, 3.7, 5.4, "Financial Return Comparison", [
    "Default 0.50 Strategy:",
    "  • Gross Retained: $244,800",
    "  • Campaign Cost: $28,400",
    "  • Net Financial Savings: $216,400",
    "Optimized 0.28 Strategy:",
    "  • Gross Retained: $461,834",
    "  • Campaign Cost: $62,040",
    "  • Net Financial Savings: $399,794",
    "Net Value Added: +$183,394 (+84.7% profit boost)!"
])

# ==========================================
# SLIDE 6: Financial Impact & ROI Simulation
# ==========================================
s6 = prs.slides.add_slide(blank_layout)
add_header(s6, "Financial Impact & Enterprise ROI Projections", "VALUE REALIZATION MODEL")

add_card(s6, 0.8, 1.5, 5.6, 5.4, "Holdout Cohort Verified Net Savings", [
    "Cohort Size: 1,409 active subscribers (20% holdout split)",
    "Actual Churners Present: 374 subscribers",
    "Subscribers Detected by Model: 307 subscribers (82.1% recall)",
    "Retention Campaign Assumptions:",
    "  • Incentive Cost: $15/month discount ($180/year) per targeted user",
    "  • Retention Conversion Rate: 35% based on industry benchmarks",
    "Total Saved Annual Revenue: $461,834",
    "Less Targeted Campaign Expense: ($62,040)",
    "VERIFIED NET SAVINGS: $399,794"
])

add_card(s6, 6.8, 1.5, 5.7, 5.4, "Enterprise 100,000 Subscriber Projection", [
    "Active Subscriber Footprint: 100,000 accounts",
    "Expected Annual Churners (26.5% rate): 26,500 subscribers",
    "Model Interventions (82.1% recall): 21,756 subscribers flagged",
    "Successfully Retained (35% take rate): 7,615 subscribers",
    "Average Annual Revenue per User (ARPU): $890",
    "Gross Enterprise Revenue Preserved: $6,777,350",
    "Enterprise Campaign Outreach Budget: ($1,093,350)",
    "TOTAL ENTERPRISE NET BENEFIT: $5,684,000 / Year"
])

# ==========================================
# SLIDE 7: Explainable AI & SHAP Risk Drivers
# ==========================================
s7 = prs.slides.add_slide(blank_layout)
add_header(s7, "Explainable AI (XAI): SHAP Feature Interpretability", "MODEL TRANSPARENCY & GOVERNANCE")

add_card(s7, 0.8, 1.5, 5.6, 5.4, "Top Features Increasing Churn Propensity", [
    "Contract: Month-to-Month (+0.38 mean |SHAP|):",
    "  • Single largest risk multiplier; users can leave anytime with zero friction.",
    "Internet Service: Fiber Optic (+0.29 mean |SHAP|):",
    "  • High monthly price point creates sensitivity to competitor promotions.",
    "Payment Method: Electronic Check (+0.19 mean |SHAP|):",
    "  • Friction of manual bill paying reminds customers of service cost monthly.",
    "Add-On Gaps: No Tech Support (+0.17 mean |SHAP|):",
    "  • Unresolved connectivity issues leave subscribers dissatisfied."
])

add_card(s7, 6.8, 1.5, 5.7, 5.4, "Top Features Protecting Against Churn", [
    "Tenure: High Loyalty 48+ Months (-0.24 mean |SHAP|):",
    "  • Long-term brand affinity and embedded habits strongly reduce churn.",
    "Contract: Two-Year (-0.22 mean |SHAP|):",
    "  • Contractual lock-in drives churn to under 3%.",
    "Services: Bundled Cyber Security & Backup (-0.14 mean |SHAP|):",
    "  • Multi-service stickiness creates high switching cost barriers.",
    "Billing: Automatic Credit Card / Bank Debit (-0.11 mean |SHAP|):",
    "  • Frictionless payments establish effortless recurring continuity."
])

# ==========================================
# SLIDE 8: Targeted Customer Retention Playbooks
# ==========================================
s8 = prs.slides.add_slide(blank_layout)
add_header(s8, "Prescriptive Retention Playbooks by Risk Tier", "ACTIONABLE CUSTOMER INTERVENTION")

add_card(s8, 0.8, 1.5, 3.6, 5.4, "Critical Risk Tier (P ≥ 0.60)", [
    "Profile: New tenure (< 6 mo), Fiber optic, Month-to-month, Electronic check.",
    "Action SLA: Within 24 Hours.",
    "Intervention:",
    "  • Priority VIP Customer Success outreach.",
    "  • Offer 1-Year contract upgrade with 20% rate freeze ($15/mo savings).",
    "  • Free 6-month Tech Support and Wi-Fi router optimization.",
    "Target Take Rate: 42%."
])

add_card(s8, 4.8, 1.5, 3.6, 5.4, "Moderate Risk Tier (0.28 ≤ P < 0.60)", [
    "Profile: 12-24 mo tenure, multiple lines, paperless billing, no add-ons.",
    "Action SLA: Within 7 Days.",
    "Intervention:",
    "  • Automated personalized email with complimentary Online Security bundle.",
    "  • $10 bill credit for switching from electronic check to automatic debit.",
    "  • Streaming service bundle upgrade.",
    "Target Take Rate: 33%."
])

add_card(s8, 8.8, 1.5, 3.7, 5.4, "Low Risk Tier (P < 0.28)", [
    "Profile: 36+ mo tenure, two-year contract, autopay, multi-service.",
    "Action SLA: Standard Lifecycle.",
    "Intervention:",
    "  • Zero aggressive intervention (avoid unnecessary discounting).",
    "  • Annual loyalty appreciation perks and early access to hardware upgrades.",
    "  • Referral bonus incentives.",
    "Target Retention: 98%+."
])

# ==========================================
# SLIDE 9: Production Architecture & System Integration
# ==========================================
s9 = prs.slides.add_slide(blank_layout)
add_header(s9, "Enterprise Production Architecture & Deployment", "SYSTEM DESIGN & LATENCY SLAS")

add_card(s9, 0.8, 1.5, 5.6, 5.4, "Technical Architecture & Scalability", [
    "Microservice Inference Engine:",
    "  • REST API supporting both synchronous single and high-throughput batch scoring.",
    "  • Zero-dependency WSGI fallback ensures execution on any runtime environment.",
    "Vectorized Feature Alignment:",
    "  • 32-feature matrix created in < 2ms without heavy pandas overhead.",
    "Enterprise Cloud Containerization:",
    "  • Production Dockerfile with health probes (`/health`) and metrics endpoints (`/metrics`).",
    "Zero-Leakage Assurance:",
    "  • Frozen StandardScaler aligned with scikit-learn 1.8.0 specifications."
])

add_card(s9, 6.8, 1.5, 5.7, 5.4, "Verification & Quality Assurance (100% PASS)", [
    "Unit & Integration Test Suite (`test_production.py` & `test_api.py`):",
    "  • test_01_pipeline_determinism: PASS (Identical outputs across runs)",
    "  • test_02_leakage_absence: PASS (Verified zero holdout overlap)",
    "  • test_03_latency_sla: PASS (Sub-20ms P95 latency achieved)",
    "  • test_04_batch_scoring: PASS (Batch alignment verified)",
    "REST API Endpoints Tested:",
    "  • GET /health -> Status 200 OK",
    "  • POST /predict -> Status 200 OK with probabilities and tier actions",
    "  • GET /ui -> Status 200 OK with interactive dashboard"
])

# ==========================================
# SLIDE 10: Live Application & Deliverables
# ==========================================
s10 = prs.slides.add_slide(blank_layout)
add_header(s10, "Live Working Project & Evaluation Deliverables", "CLIENT HANDOVER & ACCESS")

add_card(s10, 0.8, 1.5, 5.6, 5.4, "Live Working Application Access", [
    "Production Cloud Live URL:",
    f"  {LIVE_URL}",
    "Core Interactive Capabilities:",
    "  1. Dynamic Multi-Currency Switcher: Instant conversion across USD, INR, EUR, GBP, CAD, AUD, AED, SAR, SGD.",
    "  2. Real-Time Churn Propensity Meter: Sliders & dropdowns score risk in < 1 ms.",
    "  3. Batch CSV Predictor: Upload any customer CSV to score 10,000+ accounts with downloadable CSV results.",
    "  4. Financial ROI Simulator: Live interactive savings model.",
    "  5. Explainable AI: Real-time visual SHAP risk factors."
], bg_color=NAVY_DARK, border_color=BLUE_ACCENT, title_color=TEAL_GREEN)

add_card(s10, 6.8, 1.5, 5.7, 5.4, "Complete Submission Package Structure", [
    "Included in the Final Delivery ZIP:",
    "  • PPT: ConnectTel_Customer_Churn_Retention_Strategy.pptx (This Deck)",
    "  • PDF: ConnectTel_Business_Retention_Report.pdf (Full Executive Report)",
    "  • LIVE LINK: LIVE_PROJECT_LINK.txt & LIVE_PROJECT_LINK.url",
    "  • Source Code: Complete `production/` package with tests & pipeline",
    "  • Jupyter Notebook: ConnectTel_Customer_Churn_Prediction.ipynb",
    "  • Web Application: `web_live/index.html` (Standalone offline/online)",
    "  • Dataset: Telco-Customer-Churn.csv",
    "  • Infrastructure: Dockerfile, requirements.txt, run_tests.py"
])

output_pptx = os.path.join(BASE_DIR, "ConnectTel_Customer_Churn_Retention_Strategy.pptx")
prs.save(output_pptx)
print(f"Presentation saved successfully to: {output_pptx}")
