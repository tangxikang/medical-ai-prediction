#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

# ───────────────────────── App Config ─────────────────────────
st.set_page_config(
    page_title="Medical AI Prediction System",
    layout="centered",  # 居中布局
    page_icon="🏥",
)

# ───────────────────────── Global Styles ──────────────────────
# 说明：将可视区域宽度限制在接近 A4 比例；输入区放在 expander；结果只占一张 A4 画布
st.markdown("""
<style>
    /* 控制主容器宽度，接近 A4 比例（约 794×1123 px @96DPI） */
    .block-container {
        max-width: 820px;   /* ≈ A4 宽度 */
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    /* 精简全局字体，避免排版挤出一页 */
    html, body, [class*="css"] {
        font-size: 0.95rem;
    }
    /* 大按钮样式 */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        font-size: 1.1rem;
        padding: 0.5em 1em;
        font-weight: 600;
    }
    /* 输入框与标签适度放大但不过度 */
    .stTextInput>div>div>input {
        font-size: 0.95rem;
        border-radius: 8px;
    }
    label[data-baseweb="form-control"] > div:first-child {
        font-size: 0.95rem;
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
    "Na⁺": 137.3, "BUN": 14.7, "Temp": 37.0, "Lac": 0.9, "Creatinine": 0.9
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
    "RDW": "Red Cell Distribution Width (RDW) – fL",
    "PLT": "Platelet Count (PLT) – 10³/µL",
    "Lac": "Lactate (Lac) – mmol/L",
    "Creatinine": "Creatinine (Cre) – mg/dL"
}

# ───────────────────────── UI Header ──────────────────────────
st.title("🏥 Medical AI Decision Support System")
st.caption("All results are constrained to a single A4-sized canvas for clean print/export.")

# ───────────────────────── Helpers ────────────────────────────
_num_pattern = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")

def _to_float(text: str, default: float, name: str) -> float:
    t = str(text).strip()
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
    with st.expander("👉 Enter the 12 clinical indicators", expanded=True):
        left, right = st.columns(2)
        data = {}
        n_left = int(np.ceil(len(std_feature_list) / 2.0))
        for i, feat in enumerate(std_feature_list):
            col = left if i < n_left else right
            val_str = col.text_input(
                label=LABELS.get(feat, feat),
                value=str(DEFAULTS.get(feat, 0)),
                placeholder="Enter any real number (scientific notation supported)",
            )
            data[feat] = _to_float(val_str, DEFAULTS.get(feat, 0), feat)
    df = pd.DataFrame([data], columns=std_feature_list).astype(float)
    return df

# ───────────────────────── Result Figure (A4) ─────────────────
def render_a4_figure_and_pdf(X: pd.DataFrame, proba_pct: float):
    """
    返回：(fig, pdf_bytes)
      - fig: A4 画布的 Matplotlib Figure（页面渲染）
      - pdf_bytes: 同内容 A4 PDF 的二进制字节（用于下载）
    """
    # 全局 Matplotlib 字体与画布 = A4
    mpl.rcParams.update({
        "figure.figsize": (8.27, 11.69),  # A4 英寸
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8
    })

    # 计算 SHAP
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):                 # 二分类：取正类
        sv_vec = np.array(shap_values[-1][0], dtype=float)
        base_val = float(np.ravel(explainer.expected_value)[-1])
    else:                                             # 某些版本返回 ndarray
        sv_vec = np.array(shap_values[0], dtype=float)
        base_val = float(np.ravel(explainer.expected_value)[0])

    feature_values = X.iloc[0].values.astype(float)
    short_names    = list(X.columns)

    # ── 构建 A4 单页：上部文字、右侧关键信息、下部 SHAP（force_plot）
    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=True)
    gs = fig.add_gridspec(nrows=12, ncols=6)  # 细网格方便排版

    # 顶部标题（跨整行）
    ax_title = fig.add_subplot(gs[0:2, :])
    ax_title.axis('off')
    ax_title.text(
        0.5, 0.55, "Medical AI Decision Support – One-Page Report",
        ha="center", va="center", fontsize=16, weight="bold"
    )
    ax_title.text(
        0.5, 0.15, "In-hospital Mortality Prediction with SHAP Explanation (A4)",
        ha="center", va="center", fontsize=10, color="#555555"
    )

    # 左上：预测结果（大字突出）
    ax_score = fig.add_subplot(gs[2:4, 0:4])
    ax_score.axis('off')
    ax_score.text(
        0.5, 0.55,
        f"Predicted In-hospital Mortality: {proba_pct:.2f}%",
        ha="center", va="center", fontsize=22, color="#c62828", weight="bold"
    )
    ax_score.text(
        0.5, 0.15,
        "Higher value indicates higher risk",
        ha="center", va="center", fontsize=9, color="#777777"
    )

    # 右上：关键输入值摘要（表格风格）
    ax_table = fig.add_subplot(gs[2:6, 4:6])
    ax_table.axis('off')
    # 显示 6-8 个关键指标，保持单页美观；若需要可改成全部
    key_feats = ["SBP", "DBP", "APS III", "WBC", "AG", "HCO₃⁻", "Na⁺", "BUN"]
    rows = []
    for k in key_feats:
        if k in X.columns:
            rows.append((k, float(X.iloc[0][k])))
    # 动态绘制为文本表格
    y0 = 0.95
    ax_table.text(0.0, y0, "Key Inputs", fontsize=11, weight="bold")
    y = y0 - 0.10
    for name, val in rows:
        ax_table.text(0.0, y, f"{name}", fontsize=9, weight="semibold")
        ax_table.text(0.60, y, f"{val:.3g}", fontsize=9)
        y -= 0.08

    # 中部：说明
    ax_note = fig.add_subplot(gs[4:6, 0:4])
    ax_note.axis('off')
    ax_note.text(
        0, 0.9,
        "How to read SHAP:",
        fontsize=11, weight="bold"
    )
    ax_note.text(
        0, 0.65,
        "- Positive SHAP values push the prediction higher (toward death).",
        fontsize=9
    )
    ax_note.text(
        0, 0.45,
        "- Negative SHAP values push the prediction lower (toward survival).",
        fontsize=9
    )
    ax_note.text(
        0, 0.25,
        "This one-page view is optimized for A4 printing/export.",
        fontsize=9, color="#666666"
    )

    # 下部：SHAP force plot（静态，无 JS）
    ax_shap = fig.add_subplot(gs[6:12, :])
    try:
        shap.force_plot(
            base_val,
            sv_vec,
            feature_values,
            feature_names=short_names,
            matplotlib=True,
            show=False,
            ax=ax_shap
        )
    except TypeError:
        # 某些 shap 版本不支持 ax 参数；退化为独立绘制后粘贴
        plt.sca(ax_shap)
        shap.force_plot(
            base_val,
            sv_vec,
            feature_values,
            feature_names=short_names,
            matplotlib=True,
            show=False
        )

    # ——— 生成 PDF（与 fig 同内容）
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    pdf_bytes = pdf_buf.getvalue()

    return fig, pdf_bytes

# ───────────────────────── Main ───────────────────────────────
input_df = user_input_features()

# 将输入限制为模型需要的列顺序
missing = [c for c in std_feature_list if c not in input_df.columns]
if missing:
    st.error(f"Missing required features: {missing}")
else:
    if st.button("Start Prediction"):
        X = input_df[std_feature_list]
        # ——— 预测
        proba = model.predict_proba(X)[0, 1] * 100.0
        fig, pdf_bytes = render_a4_figure_and_pdf(X, proba)

        # 只展示这张 A4 画布
        st.pyplot(fig, use_container_width=False)

        # 下载按钮（A4 PDF）
        st.download_button(
            label="📄 Download A4 PDF",
            data=pdf_bytes,
            file_name="Medical_AI_OnePage_A4.pdf",
            mime="application/pdf",
            type="primary"
        )

        # 释放图像资源
        plt.close(fig)
