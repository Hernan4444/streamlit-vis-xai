from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import streamlit as st
from models.predictor import build_fixed_form, get_pipeline_features, predict_with_pipeline
from utils.data import match_pipeline_features
from utils.xai import (
    build_lime_figure,
    build_shap_waterfall_figure,
    prepare_xai_artifacts,
)


def render_prediction_xai(df, pipeline):
    st.header("Prediccion y Explicabilidad del Modelo (XAI)")

    feature_names = get_pipeline_features(pipeline, df)
    x_data = match_pipeline_features(df, feature_names)

    st.subheader("Formulario de atributos")
    user_input = build_fixed_form(x_data)

    # Primero mostramos la prediccion para feedback rapido.
    prediction_result = predict_with_pipeline(pipeline, user_input)

    st.success(f"Prediccion: {prediction_result['prediction']}")

    if prediction_result["probabilities"] is not None and prediction_result["classes"] is not None:
        prob_df = pd.DataFrame(
            {
                "Clase": prediction_result["classes"],
                "Probabilidad": prediction_result["probabilities"],
            }
        ).sort_values("Probabilidad", ascending=False)
        st.subheader("Probabilidades por clase")
        st.dataframe(prob_df, width="stretch")
        class_idx = int(prob_df.index[0])
    else:
        class_idx = 0

    xai_artifacts = prepare_xai_artifacts(pipeline, x_data)

    preprocessor = xai_artifacts["preprocessor"]
    x_instance = preprocessor.transform(prediction_result["input_df"])
    if hasattr(x_instance, "toarray"):
        x_instance = x_instance.toarray()
    x_instance = np.asarray(x_instance)

    def _run_lime():
        return build_lime_figure(
            lime_explainer=xai_artifacts["lime_explainer"],
            estimator=xai_artifacts["estimator"],
            x_instance=x_instance,
            class_idx=class_idx,
        )

    def _run_shap():
        shap_single = xai_artifacts["shap_explainer"](x_instance)
        waterfall_fig, _ = build_shap_waterfall_figure(
            shap_values=shap_single,
            x_instance=x_instance,
            feature_names=xai_artifacts["feature_names"],
            class_idx=class_idx,
        )
        return waterfall_fig

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_lime = executor.submit(_run_lime)
        future_shap = executor.submit(_run_shap)

        with st.spinner("Calculando LIME..."):
            lime_fig, _ = future_lime.result()

    st.subheader("LIME Explanation")
    st.pyplot(lime_fig)

    with st.spinner("Calculando SHAP..."):
        waterfall_fig = future_shap.result()

    st.subheader("SHAP Waterfall Plot")
    st.pyplot(waterfall_fig)