"""Funciones para carga de pipeline y predicciones."""

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_COLUMNS = [
    "tiempo_respuesta",
    "es_superhost",
    "tipo_propiedad",
    "capacidad",
    "puntaje_promedio_comunicación",
    "puntaje_promedio_localización",
    "servicio_tv_cable",
    "servicio_aire_acondicionado",
]


@st.cache_resource()
def load_pipeline(model_path="pipeline_model.pkl"):
    """Carga el pipeline entrenado desde disco."""
    return joblib.load(model_path)


def get_pipeline_features(pipeline, df):
    """Obtiene la lista de features esperadas por el pipeline."""
    if hasattr(pipeline, "feature_names_in_"):
        return [str(c) for c in pipeline.feature_names_in_]

    # Fallback general cuando no hay metadatos en el modelo.
    return [str(c) for c in MODEL_COLUMNS if c in df.columns]


def _binary_to_int(value):
    """Convierte representaciones binarias comunes a 0/1."""
    text = str(value).strip().lower()
    if text in {"1", "si", "sí", "true", "t", "yes", "y"}:
        return 1
    if text in {"0", "no", "false", "f", "n"}:
        return 0
    return int(value)


def _binary_options(series):
    """Genera opciones binarias normalizadas para selectbox."""
    unique_values = series.dropna().unique().tolist()
    normalized = sorted({_binary_to_int(v) for v in unique_values})
    if normalized == [0, 1]:
        return ["No", "Si"]
    return unique_values


def build_fixed_form(features_df, form_key="prediction_form"):
    """Crea un formulario fijo con columnas conocidas del modelo."""
    user_input = {}

    for col in MODEL_COLUMNS:
        if col not in features_df.columns:
            st.error(f"No existe la columna requerida por el modelo: {col}")
            user_input["_submitted"] = False
            return user_input

    with st.form(form_key, clear_on_submit=False):
        left, right = st.columns(2)
        col_tiempo = MODEL_COLUMNS[0]
        tiempo_opts = sorted(features_df[col_tiempo].dropna().astype(str).unique().tolist())
        user_input[col_tiempo] = left.selectbox(col_tiempo, options=tiempo_opts)

        col_superhost = MODEL_COLUMNS[1]
        superhost_opts = _binary_options(features_df[col_superhost])
        superhost_selected = left.selectbox(col_superhost, options=superhost_opts)
        user_input[col_superhost] = _binary_to_int(superhost_selected)

        col_tipo = MODEL_COLUMNS[2]
        tipo_opts = sorted(features_df[col_tipo].dropna().astype(str).unique().tolist())
        user_input[col_tipo] = left.selectbox(col_tipo, options=tipo_opts)

        col_capacidad = MODEL_COLUMNS[3]
        user_input[col_capacidad] = left.slider(
            col_capacidad,
            min_value=int(features_df[col_capacidad].min()),
            max_value=int(features_df[col_capacidad].max()),
            value=int(features_df[col_capacidad].median()),
            step=1,
        )

        col_com = MODEL_COLUMNS[4]
        user_input[col_com] = right.slider(
            col_com,
            min_value=int(features_df[col_com].min()),
            max_value=int(features_df[col_com].max()),
            value=int(features_df[col_com].median()),
            step=1,
        )

        col_loc = MODEL_COLUMNS[5]
        user_input[col_loc] = right.slider(
            col_loc,
            min_value=int(features_df[col_loc].min()),
            max_value=int(features_df[col_loc].max()),
            value=int(features_df[col_loc].median()),
            step=1,
        )

        col_tv = MODEL_COLUMNS[6]
        tv_opts = _binary_options(features_df[col_tv])
        tv_selected = right.selectbox(col_tv, options=tv_opts)
        user_input[col_tv] = _binary_to_int(tv_selected)

        col_aire = MODEL_COLUMNS[7]
        aire_opts = _binary_options(features_df[col_aire])
        aire_selected = right.selectbox(col_aire, options=aire_opts)
        user_input[col_aire] = _binary_to_int(aire_selected)

        submitted = st.form_submit_button("Predecir", use_container_width=True)

    user_input["_submitted"] = submitted
    return user_input


def predict_with_pipeline(pipeline, input_row):
    """Ejecuta prediccion y probabilidades (si aplica)."""
    input_df = pd.DataFrame([input_row])
    prediction = pipeline.predict(input_df)

    result = {
        "prediction": prediction[0],
        "input_df": input_df,
        "classes": None,
        "probabilities": None,
    }

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        classes = getattr(pipeline, "classes_", np.arange(len(probs)))
        result["classes"] = classes
        result["probabilities"] = probs

    return result
