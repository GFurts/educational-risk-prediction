from pathlib import Path

import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).parent.parent

modelo = joblib.load(BASE_DIR / "models/risk_model.joblib")
features = joblib.load(BASE_DIR / "models/features.joblib")
threshold = joblib.load(BASE_DIR / "models/threshold.joblib")
st.set_page_config(page_title="Passos Mágicos — Risco Educacional", page_icon="🎓")

st.title("🎓 Predição de Risco Educacional")
st.markdown("**Associação Passos Mágicos** — Ferramenta de apoio pedagógico")
st.divider()

st.subheader("Indicadores do aluno")

col1, col2 = st.columns(2)

with col1:
    iaa = st.slider("IAA — Autoavaliação", 0.0, 10.0, 5.0, 0.1)
    ieg = st.slider("IEG — Engajamento", 0.0, 10.0, 5.0, 0.1)
    ips = st.slider("IPS — Psicossocial", 0.0, 10.0, 5.0, 0.1)
    ida = st.slider("IDA — Aprendizagem", 0.0, 10.0, 5.0, 0.1)

with col2:
    ipp = st.slider("IPP — Psicopedagógico", 0.0, 10.0, 5.0, 0.1)
    ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 5.0, 0.1)
    ano = st.selectbox("Ano", [2022, 2023, 2024])

st.divider()

if st.button("Analisar risco", type="primary"):
    entrada = np.array([[iaa, ieg, ips, ida, ipp, ipv, ano]])
    probabilidade = modelo.predict_proba(entrada)[0][1]
    em_risco = probabilidade >= threshold

    st.subheader("Resultado")

    if em_risco:
        st.error("⚠️ Aluno em risco de defasagem")
    else:
        st.success("✅ Aluno sem risco de defasagem")

    st.metric("Probabilidade de risco", f"{probabilidade:.1%}")

    st.progress(float(probabilidade))
    st.caption(f"Threshold de decisão: {threshold:.0%}")
