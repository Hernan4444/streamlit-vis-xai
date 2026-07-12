# Aplicacion Streamlit - Exploracion, Prediccion y XAI

Aplicacion de analitica para el dataset de Airbnb con 3 secciones principales: Inicio, Exploracion y XAI.

## Estructura actual

1. `app.py`: entry point y navegacion principal.
2. `paginas/home.py`: resumen del dataset.
3. `paginas/exploration.py`: mapa por ciudad y visualizaciones en tabs (Altair, Seaborn, Plotly).
4. `paginas/prediction_xai.py`: formulario de prediccion y explicaciones LIME/SHAP.
5. `models/predictor.py`: carga de pipeline, formulario y prediccion.
6. `utils/data.py`: carga de datos, resumen y alineacion de features para el pipeline.
7. `utils/xai.py`: utilidades para SHAP y LIME.
8. `Airbnb_Locations.csv`: datos de entrada.
9. `pipeline_model.pkl`: modelo entrenado.
10. `requirements.txt`: dependencias.

## Flujo de la app

1. **Inicio**: metricas generales del dataset.
2. **Exploracion**:
	- filtros simples por pais y capacidad,
	- mapa por ciudad,
	- tab Altair (barras + pie interactivos por aire acondicionado),
	- tab Seaborn (boxplot + heatmap de correlacion),
	- tab Plotly con graficos equivalentes.
3. **XAI**:
	- prediccion primero,
	- luego explicaciones calculadas en paralelo,
	- se muestra LIME y despues SHAP waterfall.

## Ejecucion

1. Instalar dependencias:
	- `pip install -r requirements.txt`
2. Ejecutar la app:
	- `streamlit run app.py`

