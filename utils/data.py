"""Funciones para carga y analisis de datos tabulares."""
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st


def text_to_number(x):
    return 1 if x == "Si" else 0

def number_to_text(x):
    return "Si" if x == 1 else "No"


@st.cache_data
def load_dataset(data_path):
    df = pd.read_csv(data_path)
    df.es_superhost = df.es_superhost.map(number_to_text)
    df.servicio_aire_acondicionado = df.servicio_aire_acondicionado.map(number_to_text)
    return df

def analyze_dataset(df):
    """Retorna columnas numericas y categoricas detectadas automaticamente."""
    columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    columnas_categoricas = [c for c in df.columns if c not in columnas_numericas]
    return {"numericas": columnas_numericas, "categoricas": columnas_categoricas}


def get_dataset_summary(df):
    """Calcula metricas de resumen para la pagina de inicio."""
    info = analyze_dataset(df)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "cantidad_numericas": len(info["numericas"]),
        "cantidad_categoricas": len(info["categoricas"]),
        "columnas_numericas": info["numericas"],
        "columnas_categoricas": info["categoricas"],
    }


def _normalize_colname(name):
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in text.lower().strip() if ch.isalnum())


def _to_binary_int(value):
    """Convierte valores binarios comunes a 0/1."""
    text = str(value).strip().lower()
    if text in {"1", "si", "sí", "true", "t", "yes", "y"}:
        return 1
    if text in {"0", "no", "false", "f", "n"}:
        return 0
    return value


def match_pipeline_features(df, feature_names):
    """Alinea columnas del DataFrame con las features esperadas por el pipeline."""
    direct = {c: c for c in df.columns}
    normalized = {_normalize_colname(c): c for c in df.columns}

    selected_columns = []
    for expected in feature_names:
        if expected in direct:
            selected_columns.append(direct[expected])
            continue

        norm_expected = _normalize_colname(expected)
        if norm_expected in normalized:
            selected_columns.append(normalized[norm_expected])
            continue

        raise KeyError(f"No se pudo encontrar la columna esperada por el modelo: {expected}")

    aligned = df[selected_columns].copy()
    aligned.columns = feature_names

    binary_cols = {"es_superhost", "servicio_aire_acondicionado", "servicio_tv_cable"}
    for col in binary_cols.intersection(set(aligned.columns)):
        aligned[col] = aligned[col].map(_to_binary_int)
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

    return aligned
