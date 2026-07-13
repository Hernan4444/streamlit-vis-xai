import streamlit as st
from utils.data import get_dataset_summary


def render_home(df):
    summary = get_dataset_summary(df)

    st.title("Proyecto de Analitica y XAI para Dataset Tabular Airbnb")
    st.write("Explora datos, construye intuicion visual y entiende predicciones con SHAP y LIME.")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Registros", f"{summary['rows']:,}")
    with c2:
        st.metric("Variables", str(summary["cols"]))

    c4, c5 = st.columns(2)
    with c4:
        st.metric("Variables numericas", str(summary["cantidad_numericas"]))
        st.caption(", ".join(summary["columnas_numericas"][:6]))
    with c5:
        st.metric("Variables categoricas", str(summary["cantidad_categoricas"]))
        st.caption(", ".join(summary["columnas_categoricas"][:6]))

    st.markdown("### ¿Cómo usar esta aplicacion?")
    st.markdown("1. Revisa el estado general del _dataset_ en esta pagina.")
    st.markdown("2. Analiza distribuciones, relaciones y composicion con graficos interactivos en Visualizaciones.")
    st.markdown("3. En Prediccion y XAI ingresa un caso, predice con el pipeline y analiza explicaciones locales y globales.")
