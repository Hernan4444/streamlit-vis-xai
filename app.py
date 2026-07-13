import streamlit as st
from models.predictor import load_pipeline
from paginas.exploration import render_exploration
from paginas.home import render_home
from paginas.prediction_xai import render_prediction_xai
from utils.data import analyze_dataset, load_dataset


def main():
    st.set_page_config(
        page_title="Analitica + ML + XAI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Navegacion")
        page = st.radio(
            "Navegacion",
            options=[
                "Inicio",
                "Visualizaciones",
                "XAI",
            ],
            label_visibility="hidden",
        )

    df = load_dataset("Airbnb_Locations.csv")

    if page == "Inicio":
        render_home(df)
    elif page == "Visualizaciones":
        dataset_info = analyze_dataset(df)
        render_exploration(
            df=df,
            columnas_numericas=dataset_info["numericas"],
            columnas_categoricas=dataset_info["categoricas"],
        )
    else:
        pipeline = load_pipeline("pipeline_model.pkl")
        render_prediction_xai(df, pipeline)


if __name__ == "__main__":
    main()
