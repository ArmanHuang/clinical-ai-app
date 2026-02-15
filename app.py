import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import qrcode
import hashlib
from xgboost import XGBClassifier
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from io import BytesIO
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Clinical AI", layout="wide")

# =====================================================
# MODERN CLEAN MEDICAL UI
# =====================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f4f8fc, #eef3f9);
}
.risk-card {
    background: linear-gradient(135deg, #1c5d99, #4f9ed6);
    color: white;
    padding: 60px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
}
.risk-number {
    font-size: 56px;
    font-weight: 800;
}
.risk-label {
    font-size: 22px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Clinical AI",
    page_icon="🏥",   
    layout="wide"
)

st.title("🏥 Clinical AI — Hospital Decision Support System")
st.caption("30-Day Readmission Risk Prediction | Official Clinical Prototype")

st.markdown("---")

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("readmission_model.json")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# =====================================================
# INPUT SECTION
# =====================================================
st.markdown("### Patient Clinical Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (years)", 18, 120, 65)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    los = st.number_input("Length of Stay (days)", 1, 60, 7)
    prev_adm = st.number_input("Previous Admissions (6 months)", 0, 20, 2)

with col3:
    comorbidity_count = st.number_input("Comorbidity Count", 0, 15, 2)
    num_medications = st.number_input("Number of Medications", 0, 50, 10)

st.markdown("#### Laboratory & Diagnosis")

col4, col5, col6 = st.columns(3)

with col4:
    diagnosis_code = st.text_input("Primary Diagnosis (ICD-10)", "I50.9")

with col5:
    avg_creatinine = st.number_input("Creatinine (mg/dL)", 0.3, 10.0, 1.2)

with col6:
    avg_hemoglobin = st.number_input("Hemoglobin (g/dL)", 6.0, 20.0, 12.0)

avg_glucose = st.number_input("Glucose (mg/dL)", 50.0, 500.0, 110.0)

# =====================================================
# FEATURE PREP
# =====================================================
icd_group = diagnosis_code[0].upper()

input_data = pd.DataFrame({
    "age": [age],
    "length_of_stay": [los],
    "previous_admissions": [prev_adm],
    "comorbidity_count": [comorbidity_count],
    "avg_creatinine": [avg_creatinine],
    "avg_hemoglobin": [avg_hemoglobin],
    "avg_glucose": [avg_glucose],
    "num_medications": [num_medications],
    "gender_M": [1 if gender == "Male" else 0]
})

input_data["los_x_comorb"] = los * comorbidity_count
input_data["glucose_flag"] = int(avg_glucose > 200)
input_data["creatinine_flag"] = int(avg_creatinine > 2)
input_data["hb_flag"] = int(avg_hemoglobin < 10)
input_data["polypharmacy_flag"] = int(num_medications >= 15)

for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

if icd_group in feature_columns:
    input_data[icd_group] = 1

input_data = input_data[feature_columns]

# =====================================================
# ANALYZE
# =====================================================
if st.button("🔍 Analyze Risk"):

    prob = float(model.predict_proba(input_data)[0][1])
    risk_percent = prob * 100

    if prob >= 0.50:
        level = "HIGH"
        color = "#d62828"
    elif prob >= 0.30:
        level = "MODERATE"
        color = "#f77f00"
    else:
        level = "LOW"
        color = "#2a9d8f"

    colA, colB = st.columns([1.3,1])

    with colA:
        st.markdown(f"""
        <div class="risk-card">
            <div class="risk-number">{risk_percent:.1f}%</div>
            <div class="risk-label">{level} RISK</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0,30], 'color': "#2a9d8f"},
                    {'range': [30,50], 'color': "#f77f00"},
                    {'range': [50,100], 'color': "#d62828"}
                ],
            }
        ))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # GENERATE PROFESSIONAL PDF
    # =====================================================
    def generate_pdf():

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        report_id = hashlib.sha256(
            f"{datetime.now()}-{age}-{diagnosis_code}".encode()
        ).hexdigest()[:12].upper()

        formatted_time = datetime.now().strftime("%d %b %Y | %H:%M")

        # HEADER
        elements.append(Paragraph("<b>CLINICAL AI — OFFICIAL RISK REPORT</b>", styles["Title"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Hospital Clinical Decision Support System", styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph(f"Report ID: CAI-{report_id}", styles["Normal"]))
        elements.append(Paragraph(f"Generated: {formatted_time}", styles["Normal"]))
        elements.append(Spacer(1, 0.4 * inch))

        # PATIENT TABLE
        data = [
            ["Age", age],
            ["Gender", gender],
            ["Diagnosis Code", diagnosis_code],
            ["Length of Stay", los],
            ["Previous Admissions", prev_adm],
            ["Comorbidity Count", comorbidity_count],
            ["Number of Medications", num_medications],
            ["Creatinine (mg/dL)", avg_creatinine],
            ["Hemoglobin (g/dL)", avg_hemoglobin],
            ["Glucose (mg/dL)", avg_glucose],
            ["Predicted Risk (%)", f"{risk_percent:.1f}%"],
            ["Risk Category", level]
        ]

        table = Table(data, colWidths=[230, 230])
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.3, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.6 * inch))

        # QR CONTENT
        qr_text = f"""
        Clinical AI Official Report
        Report ID: CAI-{report_id}
        Risk Level: {level}
        Probability: {risk_percent:.1f}%
        Generated: {formatted_time}
        This is a computer-generated clinical document.
        """

        qr = qrcode.make(qr_text)
        qr_buffer = BytesIO()
        qr.save(qr_buffer)
        qr_buffer.seek(0)

        elements.append(Paragraph("Verification QR Code:", styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Image(qr_buffer, width=1.2*inch, height=1.2*inch))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf = generate_pdf()

    st.download_button(
        "📄 Download Official Clinical Report (PDF)",
        pdf,
        file_name="Clinical_AI_Official_Report.pdf",
        mime="application/pdf"
    )

    st.markdown("### Clinical Recommendation")

    if level == "HIGH":
        st.markdown("""
• Delay discharge until stabilization confirmed  
• Repeat laboratory evaluation within 24 hours  
• Comprehensive medication reconciliation  
• Multidisciplinary case review  
• Schedule follow-up within ≤ 7 days  
• Evaluate caregiver support  
""")
    elif level == "MODERATE":
        st.markdown("""
• Confirm discharge readiness  
• Optimize medication adherence  
• Structured discharge counseling  
• Schedule follow-up within ≤ 14 days  
""")
    else:
        st.markdown("""
• Proceed with standard discharge protocol  
• Routine outpatient follow-up  
• Educate patient on warning symptoms  
""")

    st.caption("Clinical AI — Official Hospital Decision Support Prototype")
