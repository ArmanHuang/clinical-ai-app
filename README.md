# 🏥 Clinical AI — Decision Support System

An AI-powered clinical decision support system designed to predict 30-day hospital readmission risk using machine learning and explainable AI (XAI).

## 🚀 Features
- 🔍 Readmission risk prediction using XGBoost
- 📊 Explainable AI with SHAP (feature impact visualization)
- 📈 Risk scoring with optimized threshold (ROC & Youden Index)
- 🧠 Clinical insights & reasoning
- 📄 Automated PDF report generation
- 📚 ICD-10 code interpretation & reference

## 🛠 Tech Stack
- Python
- Streamlit
- XGBoost
- SHAP
- Plotly
- ReportLab

## 🎯 Purpose
This project aims to assist healthcare professionals in identifying high-risk patients and supporting clinical decision-making through transparent and interpretable AI.

---

⚠️ This project is for educational and research purposes only and not intended for real clinical use.

## 📊 Dataset

This project uses clinical data derived from the **MIMIC-IV (Medical Information Mart for Intensive Care IV)** database provided by PhysioNet.

- Source: https://physionet.org/content/mimiciv/
- Developed by: MIT Lab for Computational Physiology
- Content: De-identified electronic health records (EHR) of ICU patients
- Includes: demographics, diagnoses (ICD codes), lab results, medications, and hospital admissions

MIMIC-IV is a publicly available dataset widely used for healthcare AI research.

⚠️ Access to the dataset requires credentialing and completion of data usage training via PhysioNet.