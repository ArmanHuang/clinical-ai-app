# ================= IMPORT =================
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import qrcode
from xgboost import XGBClassifier
from reportlab.platypus import *
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from io import BytesIO
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(page_title="Clinical AI — Decision Support System", page_icon="🏥", layout="wide")

# ================= UI STYLE =================

bg = "linear-gradient(135deg,#0f172a,#020617)"
card_bg = "#111827"
text_color = "#f9fafb"

st.markdown(f"""
            
<style>

/* ===== GLOBAL ===== */
.stApp {{
    background: {bg};
    color: {text_color};
}}

.block-container {{
    padding-top: 2,5rem;
}}

/* ===== CARD ===== */
.card {{
    background:{card_bg};
    color:{text_color};
    padding:18px;
    border-radius:14px;
    margin-top:12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}}

/* ===== RISK BOX ===== */
.risk-box {{
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}}

.high {{background: linear-gradient(135deg,#ef4444,#dc2626);}}
.medium {{background: linear-gradient(135deg,#f59e0b,#fbbf24);}}
.low {{background: linear-gradient(135deg,#22c55e,#16a34a);}}

.big {{
    font-size:46px;
    font-weight:700;
}}

/* ===== BUTTON ===== */
.stButton>button {{
    width:100%;
    border-radius:10px;
    height:45px;
    font-weight:600;
}}

/* ===== MOBILE ===== */
@media (max-width: 768px) {{
    .big {{font-size:32px;}}
    .risk-box {{padding:20px;}}
}}

</style>
""", unsafe_allow_html=True)

st.title("🏥 Clinical AI — Decision Support System")
st.caption("AI-Powered 30-Day Readmission Risk Prediction")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("readmission_model.json")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

def explain_icd(icd):
    if not icd:
        return "Unknown condition"

    code = icd.upper()[0]

    if code in ["A", "B"]:
        return "Infectious and parasitic diseases"
    elif code in ["C", "D"]:
        return "Neoplasms (cancer and tumors)"
    elif code == "E":
        return "Endocrine, nutritional and metabolic diseases"
    elif code == "F":
        return "Mental and behavioral disorders"
    elif code == "G":
        return "Diseases of the nervous system"
    elif code == "H":
        return "Diseases of the eye and ear"
    elif code == "I":
        return "Cardiovascular diseases"
    elif code == "J":
        return "Respiratory diseases"
    elif code == "K":
        return "Digestive system diseases"
    elif code == "L":
        return "Skin diseases"
    elif code == "M":
        return "Musculoskeletal diseases"
    elif code == "N":
        return "Genitourinary diseases"
    elif code == "O":
        return "Pregnancy and childbirth"
    elif code == "P":
        return "Perinatal conditions"
    elif code == "Q":
        return "Congenital abnormalities"
    elif code == "R":
        return "Symptoms and abnormal findings"
    elif code in ["S", "T"]:
        return "Injury and poisoning"
    elif code in ["V", "W", "X", "Y"]:
        return "External causes of morbidity"
    elif code == "Z":
        return "Factors influencing health status"
    
    return "Unknown medical condition"
# ================= CLINICAL REASONING =================
def clinical_reasoning():
    r = []
    if age >= 65: r.append("Advanced age reduces physiological reserve.")
    if los >= 7: r.append("Prolonged hospitalization indicates severe condition.")
    if prev_adm >= 2: r.append("Frequent admissions indicate instability.")
    if avg_creatinine > 2: r.append("Renal impairment affects recovery.")
    if avg_glucose > 200: r.append("Hyperglycemia increases complication risk.")
    if avg_hemoglobin < 10: r.append("Anemia delays recovery.")
    if num_medications >= 15: r.append("Polypharmacy increases treatment risk.")
    return r

# ================= SHAP INTERPRETATION =================
def shap_interpretation(input_data, shap_values):
    explanations = []
    for i, col in enumerate(input_data.columns):
        val = shap_values[0][i]
        if abs(val) < 0.05:
            continue
        impact = "increases" if val > 0 else "reduces"
        explanations.append(f"{col.replace('_',' ')} {impact} risk (impact={val:.2f})")
    return explanations

# ================= RISK BREAKDOWN =================
def risk_breakdown(input_data, shap_values):
    data = [(col, shap_values[0][i]) for i, col in enumerate(input_data.columns)]
    return sorted(data, key=lambda x: abs(x[1]), reverse=True)

def model_confidence(prob):
    if prob >= 0.8 or prob <= 0.2:
        return "HIGH CONFIDENCE"
    elif prob >= 0.6 or prob <= 0.4:
        return "MODERATE CONFIDENCE"
    else:
        return "LOW CONFIDENCE"

# ================= INPUT =================
col1,col2,col3 = st.columns(3)

with col1:
    age = st.number_input("Age",18,120,65)
    gender = st.selectbox("Gender",["Male","Female"])

with col2:
    los = st.number_input("Length of Stay",1,60,7)
    prev_adm = st.number_input("Previous Admissions",0,20,2)

with col3:
    comorbidity_count = st.number_input("Comorbidity Count",0,15,2)
    num_medications = st.number_input("Medications",0,50,10)

diagnosis_code = st.text_input("ICD-10","I50.9")
st.caption(f"📘 {explain_icd(diagnosis_code)}")
st.markdown(
    '<a href="https://icd.who.int/browse10/2019/en" target="_blank" style="color:#38bdf8;text-decoration:none;">🔗 View Full ICD-10 Reference</a>',
    unsafe_allow_html=True
)

avg_creatinine = st.number_input("Creatinine",0.3,10.0,1.2)
avg_hemoglobin = st.number_input("Hemoglobin",6.0,20.0,12.0)
avg_glucose = st.number_input("Glucose",50.0,500.0,110.0)

# ================= FEATURE =================
input_data = pd.DataFrame({
    "age":[age],
    "length_of_stay":[los],
    "previous_admissions":[prev_adm],
    "comorbidity_count":[comorbidity_count],
    "avg_creatinine":[avg_creatinine],
    "avg_hemoglobin":[avg_hemoglobin],
    "avg_glucose":[avg_glucose],
    "num_medications":[num_medications],
    "gender_M":[1 if gender=="Male" else 0]
})

input_data["los_x_comorb"] = los * comorbidity_count
input_data["glucose_flag"] = int(avg_glucose > 200)
input_data["creatinine_flag"] = int(avg_creatinine > 2)
input_data["hb_flag"] = int(avg_hemoglobin < 10)
input_data["polypharmacy_flag"] = int(num_medications >= 15)

for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_columns]

# ================= ANALYZE =================
if st.button("🔍 Analyze Risk"):

    # =========================
    # PREDICT
    # =========================
    prob = float(model.predict_proba(input_data)[0][1])
    risk_percent = prob * 100

    # =========================
    # LOAD THRESHOLD (MODEL)
    # =========================
    optimal_threshold = joblib.load("threshold.pkl")
    high_cutoff = joblib.load("high_cutoff.pkl")

    # =========================
    # MODEL LOGIC (ilmiah)
    # =========================
    if prob >= high_cutoff:
        level_model = "HIGH"
    elif prob >= optimal_threshold:
        level_model = "MODERATE"
    else:
        level_model = "LOW"

    # =========================
    # UI LOGIC (user friendly)
    # =========================
    if risk_percent >= 50:
        level, css = "HIGH", "high"
    elif risk_percent >= 31:
        level, css = "MODERATE", "medium"
    else:
        level, css = "LOW", "low"

    # ===== UI =====
    colA, colB = st.columns([1.4, 0.9])  # kecilin colB

    with colA:
        st.markdown(f"""
        <div class="risk-box {css}" style="height:230px; display:flex; flex-direction:column; justify-content:center;">
            <div class="big">{risk_percent:.1f}%</div>
            <div>{level} RISK</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            number={'suffix': "%"},
            gauge={
                'axis': {'range':[0,100]},
            }
        ))

        fig.update_layout(
            height=230,  # samain tinggi dengan box
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    # ===== ANALYSIS =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    breakdown = risk_breakdown(input_data, shap_values)
    top3 = breakdown[:3]
    confidence = model_confidence(prob)

    st.markdown("### 🧠 Clinical Insights")
    for r in clinical_reasoning():
        st.write("•", r)

    # ================= RISK SCORING =================
        shap_dict = dict(zip(input_data.columns, shap_values[0]))

        # ambil kontribusi terbesar
        sorted_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        top3 = sorted_factors[:3]

        # total score (normalized)
        total_impact = sum(abs(v) for v in shap_dict.values())

        def scoring_breakdown():
            result = []
            for k, v in top3:
                score = (abs(v) / total_impact) * 100
                result.append((k, v, score))
                return result

    # ================= PDF =================
    def generate_pdf(shap_dict):
        from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
        from reportlab.lib.utils import ImageReader

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        frame = Frame(doc.leftMargin, doc.bottomMargin,
                    doc.width, doc.height, id='normal')

        doc.addPageTemplates([PageTemplate(id='OneCol', frames=[frame])])

        styles = getSampleStyleSheet()
        elements = []

        report_id = datetime.now().strftime("%Y%m%d%H%M")
        now = datetime.now().strftime("%d %b %Y %H:%M")

        # ================= HEADER =================
        elements.append(Paragraph("<b>CLINICAL AI ANALYTICS REPORT</b>", styles["Title"]))
        elements.append(Spacer(1,4))
        elements.append(Paragraph(f"Report ID: {report_id}", styles["Normal"]))
        elements.append(Paragraph(f"Generated: {now}", styles["Normal"]))
        elements.append(Spacer(1,8))

        # ================= EXEC SUMMARY =================
        elements.append(Paragraph("<b>EXECUTIVE SUMMARY</b>", styles["Heading2"]))
        elements.append(Spacer(1,4))

        elements.append(Paragraph(
            f"""
            Patient is classified as <b>{level}</b> risk 
            (<b>{risk_percent:.1f}%</b>) for 30-day readmission.<br/>
            Risk classification is determined using an optimized threshold derived from ROC curve analysis (Youden Index).<br/>
            Model confidence is <b>{confidence}</b>. Interpretation should include clinical judgment.
            """,
            styles["Normal"]
        ))

        elements.append(Spacer(1,6))

        # ================= IMPACT =================
        elements.append(Paragraph("<b>CLINICAL IMPACT</b>", styles["Heading3"]))
        elements.append(Paragraph(
            "Elevated readmission risk may increase hospital burden, cost, and patient complications.",
            styles["Normal"]
        ))
        elements.append(Spacer(1,8))

        # ================= KPI =================
        kpi = Table([
            ["RISK SCORE", "RISK LEVEL", "CONFIDENCE"],
            [f"{risk_percent:.1f}%", level, confidence]
        ], colWidths=[doc.width/3]*3)

        kpi.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1f3b57")),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('BACKGROUND',(0,1),(-1,1),colors.HexColor("#f8fafc")),
            ('GRID',(0,0),(-1,-1),0.25,colors.HexColor("#94a3b8")),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTSIZE',(0,0),(-1,-1),9),   # 👈 kecilin font
            ('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6),
        ]))
        elements.append(kpi)

        elements.append(Spacer(1,10))

        # ================= PROFILE =================
        elements.append(Paragraph("<b>PATIENT PROFILE</b>", styles["Heading2"]))

        profile = Table([
            ["Age", age, "Gender", gender],
            ["LOS", los, "Prev Adm", prev_adm],
            ["Creatinine", avg_creatinine, "Glucose", avg_glucose],
            ["Hemoglobin", avg_hemoglobin, "Comorbidity", comorbidity_count],
            ["Diagnosis", diagnosis_code, "", ""]
        ], colWidths=[doc.width/4]*4)

        profile.setStyle(TableStyle([
            ('BACKGROUND',(0,1),(-1,-1),colors.HexColor("#f8fafc")),
            ('GRID',(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),
            ('FONTSIZE',(0,0),(-1,-1),9),   # 👈 kecilin font
            ('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),5),
        ]))
        elements.append(profile)

        elements.append(Spacer(1,10))

        # ================= TOP FACTORS =================
        elements.append(Paragraph("<b>TOP RISK FACTORS</b>", styles["Heading2"]))

        top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        names = [k.replace("_", " ") for k, v in top_factors]
        values = [v for k, v in top_factors]

        fig, ax = plt.subplots()
        ax.barh(names, values)
        ax.axvline(0)

        for i, v in enumerate(values):
            ax.text(v, i, f"{v:.2f}", va='center')

        buf = BytesIO()
        plt.savefig(buf, bbox_inches='tight')
        plt.close()
        buf.seek(0)

        elements.append(Image(buf, width=doc.width, height=190))

        elements.append(Spacer(1,6))

        elements.append(Paragraph(
            "Key variables contributing to readmission risk based on model explainability.",
            styles["Normal"]
        ))

        elements.append(PageBreak())

        # ================= CLINICAL ANALYSIS =================
        elements.append(Paragraph("<b>CLINICAL INTERPRETATION</b>", styles["Heading2"]))

        for r in clinical_reasoning():
            elements.append(Paragraph(f"• {r}", styles["Normal"]))

        elements.append(Spacer(1,14))

        # ================= SHAP =================
        elements.append(Paragraph("<b>MODEL EXPLAINABILITY (SHAP)</b>", styles["Heading2"]))

        try:
            plt.figure()
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=input_data.iloc[0],
                    feature_names=input_data.columns
                )
            )

            shap_buf = BytesIO()
            plt.savefig(shap_buf, bbox_inches='tight')
            plt.close()
            shap_buf.seek(0)

            elements.append(Image(shap_buf, width=doc.width, height=250))
        except:
            elements.append(Paragraph("SHAP visualization unavailable.", styles["Normal"]))

        elements.append(Spacer(1,14))

        # ================= RECOMMENDATION =================
        elements.append(Paragraph("<b>CLINICAL RECOMMENDATION</b>", styles["Heading2"]))

        elements.append(Paragraph("<b>Critical Actions</b>", styles["Heading3"]))
        elements.append(Paragraph("• Delay discharge until stability confirmed", styles["Normal"]))
        elements.append(Paragraph("• Repeat laboratory evaluation within 24–48 hours", styles["Normal"]))

        elements.append(Paragraph("<b>Moderate Actions</b>", styles["Heading3"]))
        elements.append(Paragraph("• Optimize medication regimen", styles["Normal"]))

        elements.append(Paragraph("<b>Routine Actions</b>", styles["Heading3"]))
        elements.append(Paragraph("• Schedule follow-up within 7 days", styles["Normal"]))

        # ================= QR =================
        qr = qrcode.make(f"AI Clinical Report {report_id}")
        qr_buf = BytesIO()
        qr.save(qr_buf)
        qr_buf.seek(0)

        def footer(canvas, doc):
            img = ImageReader(qr_buf)
            canvas.drawImage(img, A4[0]-55, 20, width=30, height=30)
            canvas.setFont("Helvetica",8)
            canvas.drawString(40,20,"AI Clinical Decision Support")

        doc.build(elements, onFirstPage=footer, onLaterPages=footer)

        buffer.seek(0)
        return buffer
    pdf = generate_pdf(shap_dict)
    st.download_button("📄 Download Full AI Report", pdf, "Clinical_Report.pdf")