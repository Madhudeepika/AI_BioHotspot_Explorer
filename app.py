import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path
import py3Dmol

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="AI-BioHotspot Explorer | Madhu Deepika",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# CUSTOM CSS — Dark DNA Background (Streamlit-safe)
# ------------------------------------------
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0a1a2a 0%, #041018 100%), 
                    url("https://i.ibb.co/X2cszK1/ai-dna-genomics-bg.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        color: #e6f2ff;
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 25, 40, 0.9);
        color: #e6f2ff;
    }

    h1, h2, h3, h4, label, p, span {
        color: #e6f2ff !important;
    }

    [data-testid="stExpander"] {
        background: rgba(10, 25, 40, 0.75);
        border-radius: 10px;
        color: #e6f2ff;
    }

    button[kind="primary"] {
        background-color: #1e88e5 !important;
        color: white !important;
        border-radius: 8px !important;
    }

    .stDataFrame {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
    }

    footer {
        visibility: hidden;
    }

    body::before {
        content: "01001001 01001100 01001101 01010100 01010000 01000101 01011000";
        font-family: monospace;
        font-size: 10px;
        color: rgba(255,255,255,0.05);
        position: fixed;
        top: 25%;
        left: 10%;
        z-index: 0;
        transform: rotate(-25deg);
        white-space: pre;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------
with st.sidebar:
    st.title("AI-BioHotspot Explorer")
    gene = st.selectbox("Select Gene", ["TP53", "BRCA1"], index=0)
    model_choice = st.radio("Select Model", ["Logistic Regression", "Deep Learning"], index=1)
    top_n_choice = st.selectbox("Show Top Hotspots", ["Top 5", "Top 10", "Top 15", "Top 20"], index=1)
    top_n = int(top_n_choice.split(" ")[1])
    show_exp = st.checkbox("Show Explainability", True)
    show_eff = st.checkbox("Show Efficiency", True)
    show_3d = st.checkbox("Show 3D Protein Structure", True)

EXPORTS = Path("exports")

def safe_load_csv(name):
    p = EXPORTS / name
    if p.exists():
        return pd.read_csv(p)
    else:
        st.warning(f"Missing file: exports/{name}")
        return None

df_lr = safe_load_csv("enhanced_hotspot_predictions_lr.csv")
df_dl = safe_load_csv("enhanced_hotspot_predictions_dl.csv")
feat_imp = safe_load_csv("feature_importance_lr.csv")
runtime_log = safe_load_csv("runtime_log.csv")
hde = safe_load_csv("HDE_summary.csv")

st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.title("🔬 AI-BioHotspot Explorer Dashboard")

# ------------------------------------------
# PREDICTIONS PANEL
# ------------------------------------------
with st.expander("▾ Predictions", expanded=True):
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    if model_choice == "Logistic Regression":
        df = df_lr
        prob_col = "AI_Hotspot_Prob_LR"
    else:
        df = df_dl
        prob_col = "DL_Hotspot_Prob"

    if df is not None:
        top = df.sort_values(prob_col, ascending=False).head(top_n)
        st.dataframe(top[["codon", prob_col, "literature_mentions", "conservation_score", "mutation_count"]])
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(top["codon"].astype(str), top[prob_col], color="#00bfff")
        ax.set_xlabel("Codon")
        ax.set_ylabel("Predicted Probability")
        ax.set_title(f"{top_n_choice} Predicted Hotspots")
        st.pyplot(fig)
    else:
        st.info("No data found. Upload exports folder.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# EXPLAINABILITY PANEL
# ------------------------------------------
with st.expander("▾ Explainability"):
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    if show_exp and feat_imp is not None:
        fig2, ax2 = plt.subplots(figsize=(6,2))
        ax2.barh(feat_imp["feature"], feat_imp["abs_coef"], color="skyblue")
        ax2.set_xlabel("Importance")
        st.pyplot(fig2)
    else:
        st.info("Feature importance not available.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# EFFICIENCY PANEL
# ------------------------------------------
with st.expander("▾ Efficiency & Sustainability"):
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    if show_eff and hde is not None:
        st.subheader("HDE Summary")
        st.dataframe(hde)
    if runtime_log is not None:
        st.subheader("Runtime")
        st.dataframe(runtime_log)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# 3D PROTEIN STRUCTURE PANEL (Streamlit-safe)
# ------------------------------------------
with st.expander("▾ 3D Protein Structure Viewer", expanded=False):
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    if show_3d:
        st.write("🧬 Interactive 3D Model of TP53 (PDB ID: 1TUP)")
        xyz = py3Dmol.view(query='pdb:1TUP')
        xyz.setStyle({'cartoon': {'color': 'spectrum'}})
        xyz.zoomTo()
        html = xyz._make_html()
        st.markdown("<div style='border:1px solid #1e88e5; border-radius:8px; padding:5px;'>", unsafe_allow_html=True)
        st.components.v1.html(html, height=500)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Enable 3D viewer in sidebar.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# COMING SOON PANEL
# ------------------------------------------
with st.expander("🔮 Coming Soon"):
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.write("""
    • Full multi-gene dashboard (input any gene)  
    • LLM-powered interpretive reasoning (how & why)  
    • Structural mutation impact predictions  
    • Export to PDF / publication-ready report  
    • Energy & sustainability visualization (power usage, green AI)  
    • Deploy mobile-friendly interface / API  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# FOOTER
# ------------------------------------------
st.markdown(
    """
    <hr style='border:1px solid #1e88e5;'>
    <div style='text-align:center; font-size:15px; color:#89cff0;'>
        Developed by <b>Oruganti Amsu Madhu Deepika</b> — All Rights Reserved © 2025
    </div>
    """,
    unsafe_allow_html=True
)
