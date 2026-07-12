"""Funciones de explicabilidad para SHAP y LIME."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer


@st.cache_resource(show_spinner=False)
def prepare_xai_artifacts(_pipeline, x_train):
    """Prepara objetos reutilizables para SHAP/LIME usando la parte final del pipeline."""
    if not hasattr(_pipeline, "steps") or len(_pipeline.steps) < 2:
        raise ValueError("El pipeline debe tener al menos preprocesador y estimador.")

    preprocessor = _pipeline[:-1]
    estimator = _pipeline[-1]

    x_transformed = preprocessor.transform(x_train)
    if hasattr(x_transformed, "toarray"):
        x_transformed = x_transformed.toarray()

    x_transformed = np.asarray(x_transformed)

    if hasattr(preprocessor, "get_feature_names_out"):
        transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        transformed_feature_names = [f"feature_{i}" for i in range(x_transformed.shape[1])]

    background_size = min(250, x_transformed.shape[0])
    background = x_transformed[:background_size]

    explainer = shap.Explainer(estimator, background, feature_names=transformed_feature_names)

    lime_explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=transformed_feature_names,
        class_names=[str(c) for c in getattr(estimator, "classes_", [])] or ["class_0", "class_1"],
        mode="classification",
        discretize_continuous=True,
    )

    return {
        "preprocessor": preprocessor,
        "estimator": estimator,
        "x_transformed": x_transformed,
        "feature_names": transformed_feature_names,
        "shap_explainer": explainer,
        "lime_explainer": lime_explainer,
    }


def _select_class_axis(shap_values, class_idx):
    """Extrae matrices de valores SHAP y base_values para una clase."""
    values = np.asarray(shap_values.values)
    base_values = np.asarray(shap_values.base_values)

    if values.ndim == 3:
        values = values[:, :, class_idx]
    if base_values.ndim >= 2:
        base_values = base_values[:, class_idx]

    return values, base_values


def build_shap_summary_figure(
    shap_values,
    x_sample,
    feature_names,
    class_idx,
):
    """Construye figura SHAP summary plot para una clase objetivo."""
    values, _ = _select_class_axis(shap_values, class_idx)
    plt.close("all")
    fig = plt.figure(figsize=(9, 5))
    shap.summary_plot(values, x_sample, feature_names=feature_names, show=False)
    return fig


def build_shap_waterfall_figure(
    shap_values,
    x_instance,
    feature_names,
    class_idx,
):
    """Construye waterfall plot y tabla de contribuciones SHAP."""
    values, base_values = _select_class_axis(shap_values, class_idx)
    one_values = values[0]
    one_base = base_values[0] if np.ndim(base_values) else float(base_values)

    explanation = shap.Explanation(
        values=one_values,
        base_values=one_base,
        data=x_instance[0],
        feature_names=feature_names,
    )

    plt.close("all")
    shap.plots.waterfall(explanation, max_display=12, show=False)
    fig = plt.gcf()

    contrib = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": one_values,
        }
    ).sort_values("contribution", key=np.abs, ascending=False)

    return fig, contrib


def build_lime_figure(
    lime_explainer,
    estimator,
    x_instance,
    class_idx,
):
    """Genera explicacion LIME para una instancia."""
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("El estimador final no soporta predict_proba para LIME.")

    exp = lime_explainer.explain_instance(
        data_row=x_instance[0],
        predict_fn=estimator.predict_proba,
        num_features=10,
        top_labels=1,
    )

    fig = exp.as_pyplot_figure(label=class_idx)
    explanation_list = exp.as_list(label=class_idx)
    return fig, explanation_list


