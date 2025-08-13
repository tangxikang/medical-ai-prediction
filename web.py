#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import shap
import joblib
import tempfile
import streamlit as st
import streamlit.components.v1 as components

# ───────────────────────── App Config ─────────────────────────
st.set_page_config(
    page_title="Medical AI Prediction System",
    layout="wide",
    page_icon="🏥",
)

# ───────────────────────── Global Styles ──────────────────────
st.markdown("""
<style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-size: 20px;
        padding: 0.5em 1em;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        border-radius: 8px;
    }
    label[data-baseweb="form-control"] > div:first-child {
        font-size: 20px;
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
    "and provide a **SHAP force plot** explanation.\n"
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
        data[feat] = _to_float(val_str, DEFAULTS.get(feat, 0), feat)
    df = pd.DataFrame([data], columns=std_feature_list).astype(float)
    return df

# ───────────────────────── Main Form ──────────────────────────
input_df = user_input_features()

if st.button("Start Prediction"):
    # ——— Predict
    input_df = input_df[std_feature_list]
    proba = model.predict_proba(input_df)[0, 1] * 100.0
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
    st.subheader("🔍 SHAP Force Plot Explanation")

    # ——— SHAP values & base value (version-safe)
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):  # binary classifier
        sv_vec = np.array(shap_values[-1][0], dtype=float)
        base_val = float(np.ravel(explainer.expected_value)[-1])
    else:
        sv_vec = np.array(shap_values[0], dtype=float)
        base_val = float(np.ravel(explainer.expected_value)[0])

    short_names    = std_feature_list
    feature_values = input_df.iloc[0].values.astype(float)

    # ——— Try new API, fallback to old API
    try:
        force_obj  = shap.plots.force(base_val, sv_vec, feature_values,
                                      feature_names=short_names, matplotlib=False)
        force_html = force_obj.html()
    except TypeError:
        force_obj  = shap.force_plot(base_val, sv_vec, feature_values,
                                     feature_names=short_names, matplotlib=False)
        force_html = force_obj.html()

    # ——— Stable-width centered container (fix tiny/squeezed chart)
    html_all = f"""
    <head>
      {shap.getjs()}
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        html, body {{ margin:0; padding:0; }}
        .outer {{
          width: 100%;
          display: flex;
          justify-content: center;            /* center horizontally */
        }}
        .inner {{
          width: min(1200px, max(720px, 60vw)); /* ★ stable width for SHAP JS */
          overflow-x: auto;                     /* allow scroll on very small screens */
        }}
        .inner > div:first-child {{ display: inline-block !important; }}
      </style>
    </head>
    <body>
      <div class="outer">
        <div class="inner" id="shap-holder">
          {force_html}
        </div>
      </div>
      <script>
        (function() {{
          // ensure SHAP root doesn't force 100% width
          var holder = document.getElementById('shap-holder');
          if (!holder) return;
          var child = holder.firstElementChild;
          if (!child) return;
          child.style.display = 'inline-block';
          child.style.width   = 'auto';
          child.style.maxWidth = '100%';
        }})();
      </script>
    </body>
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp.write(html_all.encode("utf-8"))
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=520, scrolling=True)

    os.remove(tmp_path)
