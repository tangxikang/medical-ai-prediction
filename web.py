import streamlit as st
import pandas as pd
import shap
import joblib
import tempfile
import os
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Medical AI Prediction System",
    layout="wide",
    page_icon="🏥",
)

# ─────────────────── 1. Custom CSS ───────────────────
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            font-size: 20px;
            padding: 0.5em 1em;
        }
        .stNumberInput>div>input {
            font-size: 18px;
            border-radius: 8px;
        }
        label[data-baseweb="form-control"] > div:first-child {
            font-size: 20px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────── 2. Load the model ───────────────
@st.cache_resource
def load_model():
    """Load the pre‑trained LGBM‑dart model from disk."""
    return joblib.load("result/LGBM-dart_model.pkl")

model = load_model()
explainer = shap.TreeExplainer(model)

# ─────────────────── 3. Feature meta data ────────────
FEATURE_NAMES = [
    "SBP",
    "DBP",
    "APSIII",
    "WBC",
    "AG",
    "HCO3",
    "Na",
    "BUN",
    "Temp",
    "RDW",
    "PLT",
    "Lactate",
]

DEFAULTS = {
    "SBP": 120.0,
    "DBP": 80.0,
    "APSIII": 40.0,
    "WBC": 7.0,
    "AG": 12.0,
    "HCO3": 24.0,
    "Na": 140.0,
    "BUN": 15.0,
    "Temp": 37.0,
    "RDW": 13.0,
    "PLT": 200.0,
    "Lactate": 1.0,
}

LABELS = {
    "SBP": "Systolic Blood Pressure (SBP) – mmHg",
    "DBP": "Diastolic Blood Pressure (DBP) – mmHg",
    "APSIII": "Acute Physiology Score III (APSIII)",
    "WBC": "White Blood Cell Count (WBC) – 10^3/µL",
    "AG": "Anion Gap (AG) – mmol/L",
    "HCO3": "Bicarbonate (HCO₃⁻) – mmol/L",
    "Na": "Sodium (Na⁺) – mmol/L",
    "BUN": "Blood Urea Nitrogen (BUN) – mg/dL",
    "Temp": "Body Temperature (Temp) – °C",        # ✅ 添加括号
    "RDW": "Red Cell Distribution Width (RDW) – fl",
    "PLT": "Platelet Count (PLT) – 10^3/µL",
    "Lactate": "Lactate (Lac) – mmol/L",            # ✅ 添加括号
}


# ─────────────────── 4. Page header ──────────────────
st.title("🏥 Medical AI Decision Support System")
st.markdown(
    "Enter the 12 bedside test indicators below. The system will predict **in‑hospital mortality risk** and provide a **SHAP force plot** explanation.\n"
)

# ─────────────────── 5. Collect user input ───────────

def user_input_features() -> pd.DataFrame:
    st.markdown("### 👨‍⚕️ Enter the 12 clinical indicators")
    left, right = st.columns(2)
    data: dict[str, int] = {}

    RANGE_LIMITS = {
        "SBP": (0, 500),
        "DBP": (0, 500),
        "APSIII": (0, 500),
        "WBC": (0, 5000),
        "AG": (0, 100),
        "HCO3": (0, 100),
        "Na": (0, 300),
        "BUN": (0, 300),
        "Temp": (0, 45),
        "RDW": (0, 100),
        "PLT": (0, 5000),
        "Lactate": (0, 100),
    }

    for i, feat in enumerate(FEATURE_NAMES):
        col = left if i < 6 else right
        min_val, max_val = RANGE_LIMITS[feat]
        data[feat] = col.number_input(
            label=LABELS[feat],
            min_value=min_val,
            max_value=max_val,
            value=int(DEFAULTS[feat]),
            step=1,
            format="%d"
        )

    return pd.DataFrame([data])


input_df = user_input_features()

# ─────────────────── 6. Prediction & SHAP plot ───────
if st.button("Start Prediction"):
    # 6‑1   Probability prediction
    proba = model.predict_proba(input_df)[0, 1] * 100  # to percent
    st.markdown(
        f"""
        <div style='text-align:center; font-size:22px; color:red; margin:20px 0;'>
            <strong>🤖 Predicted in‑hospital mortality probability: {proba:.2f}%</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("🔍 SHAP Force Plot Explanation")

    # 6‑2   Compute SHAP values
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # positive class for binary models
        base_val = explainer.expected_value[1]
    else:
        sv = shap_values[0]
        base_val = explainer.expected_value

    # 6‑3   Feature names shown in the force plot – must exactly match the input labels
    import re
    display_names = [re.search(r"\((.*?)\)", LABELS[f]).group(1) if "(" in LABELS[f] else LABELS[f] for f in FEATURE_NAMES]

    feature_values = input_df.iloc[0].values  # ndarray aligned with sv

    force_html = shap.plots.force(
        base_value=base_val,
        shap_values=sv,
        features=feature_values,
        feature_names=display_names,
        matplotlib=False,
    ).html()

    # 6‑4   Embed the plot inside Streamlit
    html_all = f"<head>{shap.getjs()}</head><body>{force_html}</body>"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp.write(html_all.encode("utf-8"))
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=380)
    os.remove(tmp_path)
