#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import tempfile
import streamlit as st

# ───────────────────────── App Config ─────────────────────────
st.set_page_config(
    page_title="Medical AI Prediction System",
    layout="wide",
    page_icon="🏥",
)

# ───────────────────────── Global Styles ──────────────────────
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-size: 5rem; /* 全局基础字体，原本是16px，这里+4px */
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-size: 24px; /* 按钮字体加大 */
        padding: 0.5em 1em;
    }
    .stTextInput>div>div>input {
        font-size: 22px; /* 输入框内容字体加大 */
        border-radius: 8px;
    }
    label[data-baseweb="form-control"] > div:first-child {
        font-size: 24px; /* 输入框标签加大 */
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── Model & Features ───────────────────
@st.cache_resource
def load_model():
    return joblib.load("result/LGBM-dart_model.pkl")

@st.cache_resource
def load_features():
    return joblib.load("result/LGBM-dart_features.pkl")

model = load_model()
feature_list = load_features()
explainer = shap.TreeExplainer(model)

# ───────────────────────── Name Mapping (fixed) ───────────────
COLUMN_MAPPING = {
    "SB": "SBP", "DB": "DBP", "T": "Temp",
    "score1": "APS III", "score2": "WBC", "score6": "PLT",
    "score7": "AG", "score8": "HCO₃⁻", "SC1": "RDW",
    "Na": "Na⁺", "BUN": "BUN", "Cre": "Creatinine", "Lac": "Lac"
}
std_feature_list = [COLUMN_MAPPING.get(f, f) for f in feature_list]

# ───────────────────────── Defaults & Labels ──────────────────
DEFAULTS = {
    "SBP": 122.5, "DBP": 84.8, "APS III": 29, "WBC": 7.9,
    "PLT": 165.4, "AG": 9, "HCO₃⁻": 21, "RDW": 15.3,
    "Na⁺": 137.3, "BUN": 14.7, "Temp": 37, "Lac": 0.9, "Creatinine": 0.9
}
LABELS = {
    "SBP": "Systolic Blood Pressure (SBP) – mmHg",
    "DBP": "Diastolic Blood Pressure (DBP) – mmHg",
    "APS III": "Acute Physiology Score III (APSIII)",
    "WBC": "White Blood Cell Count (WBC) – 10³/µL",
    "AG": "Anion Gap (AG) – mmol/L",
    "HCO₃⁻": "Bicarbonate (HCO₃⁻) – mmol/L",
    "Na⁺": "Sodium (Na⁺) – mmol/L",
    "BUN": "Blood Urea Nitrogen (BUN) – mg/dL",
    "Temp": "Body Temperature (Temp) – °C",
    "RDW": "Red Cell Distribution Width (RDW) – fl",
    "PLT": "Platelet Count (PLT) – 10³/µL",
    "Lac": "Lactate (Lac) – mmol/L",
    "Creatinine": "Creatinine (Cre) – mg/dL"
}

# ───────────────────────── UI Header ──────────────────────────
st.title("🏥 Medical AI Decision Support System")
st.markdown(
    "Enter the 12 bedside test indicators below. The system will predict **in-hospital mortality risk** "
    "and provide a **SHAP force plot (static)** explanation.\n"
)

# ───────────────────────── Helpers ────────────────────────────
_num_pattern = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

def _to_float(text: str, default: float, name: str) -> float:
    t = text.strip()
    if _num_pattern.match(t):
        try:
            return float(t)
        except Exception:
            st.warning(f"⚠️ {name}: cannot parse '{t}', fallback to default {default}.")
            return float(default)
    else:
        st.warning(f"⚠️ {name}: invalid number '{t}', fallback to default {default}.")
        return float(default)

def user_input_features() -> pd.DataFrame:
    st.markdown("### 👨‍⚕️ Enter the 12 clinical indicators")
    left, right = st.columns(2)
    data = {}
    for i, feat in enumerate(std_feature_list):
        col = left if i < 6 else right
        val_str = col.text_input(
            label=LABELS.get(feat, feat),
            value=str(DEFAULTS.get(feat, 0)),
            placeholder="Enter any real number (no limits)",
            help="No min/max or decimal-place limits. Scientific notation supported (e.g., 1e-3)."
        )
        data[feat] = _to_float(val_str, DEFAULTS.get(feet, 0) if (feet:=feat) else 0, feat)  # safe get
    df = pd.DataFrame([data], columns=std_feature_list).astype(float)
    return df

# ───────────────────────── Main Form ──────────────────────────
input_df = user_input_features()

if st.button("Start Prediction"):
    # ——— Predict
    X = input_df[std_feature_list]
    proba = model.predict_proba(X)[0, 1] * 100.0
    proba_int = round(proba, 2)

    st.markdown(
        f"""
        <div style='text-align:center; font-size:28px; color:#c62828; margin:20px 0; font-weight:800;'>
            🤖 Predicted in-hospital mortality probability: <span style='font-size:40px;'>{proba_int}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("🔍 SHAP Force Plot (Static, Matplotlib)")

    # ——— SHAP values & base value（版本安全）
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # 二分类：取正类
        sv_vec = np.array(shap_values[-1][0], dtype=float)
        base_val = float(np.ravel(explainer.expected_value)[-1])
    else:
        sv_vec = np.array(shap_values[0], dtype=float)
        base_val = float(np.ravel(explainer.expected_value)[0])

    feature_values = X.iloc[0].values.astype(float)
    short_names    = std_feature_list

    # ——— 静态 force_plot（不依赖任何 JS/CDN）
    plt.figure(figsize=(10, 3.8))
    shap.force_plot(
        base_val,
        sv_vec,
        feature_values,
        feature_names=short_names,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.close()






