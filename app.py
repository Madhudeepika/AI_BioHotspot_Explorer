import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path

st.set_page_config(page_title="AI-BioHotspot Explorer (Lab Mode)", layout="wide", initial_sidebar_state="expanded")

# DNA + binary background styling
st.markdown("""
<style>
.stApp {
  background-image: url('https://images.unsplash.com/photo-1581092795366-0e07c1f4b6f9?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80');
  background-size: cover;
  background-attachment: fixed;
  color: #e6f2ff;
}
.panel {
  background: rgba(10, 25, 40, 0.75);
  padding: 12px;
  border-radius: 8px;
  color: #e6f2ff;
}
.binary {
  font-family: monospace;
  color: rgba(255,255,255,0.06);
  position: absolute;
  top: 10%;
  left: 5%;
  font-size: 10px;
  transform: rotate(-20deg);
  pointer-events: none;
  z-index: 0;
}
</style>
<div class="binary">01001001 01001010 01001100 01001101 01010100 01010000 01000101</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.title("AI-BioHotspot Explorer")
    gene = st.selectbox("Select Gene", ["TP53", "BRCA1"], index=0)
    model_choice = st.radio("Select Model", ["Logistic Regression", "Deep Learning"], index=1)
    top_n = st.slider("Top N hotspots", 3, 20, 8)
    show_exp = st.checkbox("Show Explainability", True)
    show_eff = st.checkbox("Show Efficiency", True)

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

# Predictions Panel
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
        ax.bar(top["codon"].astype(str), top[prob_col])
        ax.set_xlabel("Codon")
        ax.set_ylabel("Predicted Probability")
        ax.set_title(f"Top {top_n} predicted hotspots")
        st.pyplot(fig)
    else:
        st.info("No data found. Upload exports folder.")
    st.markdown("</div>", unsafe_allow_html=True)

# Explainability Panel
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

# Efficiency Panel
with st.expander("▾ Efficiency & Sustainability"):
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    if show_eff and hde is not None:
        st.subheader("HDE Summary")
        st.dataframe(hde)
    if runtime_log is not None:
        st.subheader("Runtime")
        st.dataframe(runtime_log)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr><center>© AI-BioHotspot Explorer – Lab Mode</center>", unsafe_allow_html=True)
