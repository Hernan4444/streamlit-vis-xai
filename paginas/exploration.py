"""Pagina de exploracion analitica con multiples librerias."""

import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import streamlit as st


def _apply_filters(df):
    """Aplica solo dos filtros sencillos: pais y capacidad."""
    filtered = df.copy()

    with st.expander("Filtros de visualizacion", expanded=True):
        f1, f2 = st.columns(2)

        paises = sorted(filtered["pais"].dropna().astype(str).unique().tolist())
        opcion_pais = f1.selectbox("Pais", options=["Todos"] + paises)
        if opcion_pais != "Todos":
            filtered = filtered[filtered["pais"].astype(str) == opcion_pais]

        if not filtered["capacidad"].dropna().empty:
            minimo = int(filtered["capacidad"].min())
            maximo = int(filtered["capacidad"].max())
            rango = f2.slider("Capacidad", min_value=minimo, max_value=maximo, value=(minimo, maximo))
            filtered = filtered[(filtered["capacidad"] >= rango[0]) & (filtered["capacidad"] <= rango[1])]

    return filtered


def interactive_view_altair(df):
    """Visualizaciones interactivas con Altair."""
    st.subheader("Propiedad y servicio de aire acondicionado")
    st.write("Puedes presionar la leyenda del pie chart para filtrar los datos.")

    selection = alt.selection_point(
        name="filtro_aire_acondicionado",
        fields=["servicio_aire_acondicionado"],
        bind="legend",
    )

    color_altair = alt.Color(
        "servicio_aire_acondicionado:N",
        legend=alt.Legend(title="Aire Acondicionado"),
        scale=alt.Scale(scheme="set2"),
    )

    pie = (
        alt.Chart(df)
        .mark_arc()
        .encode(
            theta="count()",
            color=color_altair,
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        )
        .properties(width=200)
    )

    bar = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count()", axis=alt.Axis(title="Cantidad de Airbnb")),
            y=alt.Y("tipo_propiedad:N", axis=alt.Axis(labelLimit=200), title="Tipo de Propiedad"),
            color=color_altair,
        )
        .transform_filter(selection)
        .properties(height=300, width=200)
    )

    juntos = (bar | pie).add_params(selection)
    st.altair_chart(juntos, width="stretch")


def render_exploration(df, columnas_numericas, columnas_categoricas):
    st.header("Exploracion del Dataset")
    st.write("Visualizaciones combinando Matplotlib/Seaborn, Altair y Plotly.")

    filtered = _apply_filters(df)
    st.subheader("DataFrame interactivo")
    st.dataframe(filtered, height=320)

    if filtered.empty:
        st.warning("No hay datos luego de aplicar filtros.")
        return

    st.subheader("Mapa de Airbnb por ciudad")
    opciones_ciudad = ["Todas"] + sorted(filtered["ciudad"].dropna().astype(str).unique().tolist())
    ciudad_seleccionada = st.selectbox("Ciudad", opciones_ciudad, key="map_ciudad")

    if ciudad_seleccionada == "Todas":
        datos_mapa = filtered
    else:
        datos_mapa = filtered[filtered["ciudad"].astype(str) == ciudad_seleccionada]

    if datos_mapa.empty:
        st.warning("No hay puntos para mostrar en el mapa con la ciudad seleccionada.")
    else:
        st.map(datos_mapa, latitude="latitud", longitude="longitud")

    columnas_numericas = ["capacidad", "camas"]
    columnas_categoricas = [
        "pais",
        "es_superhost",
        "servicio_aire_acondicionado",
        "tiempo_respuesta",
    ]

    tab1, tab2, tab3 = st.tabs(["Altair", "Matplotlib / Seaborn", "Plotly"])

    with tab1:
        interactive_view_altair(filtered)

    with tab2:
        # Gráfico 1: Boxplot con Seaborn
        box_y = st.selectbox("Boxplot - variable numerica", columnas_numericas, key="box_num")
        box_x = st.selectbox("Boxplot - agrupacion", columnas_categoricas, key="box_cat")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax_box = sns.boxplot(data=filtered, x=box_x, y=box_y)
        ax_box.set_title(f"Boxplot de {box_y} por {box_x}")
        ax_box.tick_params(axis="x", rotation=25)
        st.pyplot(fig)

        # Gráfico 2: Heatmap de correlacion
        corr = filtered[columnas_numericas].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax_corr = sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f")
        ax_corr.set_title("Heatmap de correlacion")
        st.pyplot(fig)

    with tab3:
        plotly_template = "plotly_white"

        hist_col = st.selectbox("Histograma Plotly - columna", columnas_numericas, key="px_hist_col")
        fig_hist = px.histogram(
            filtered,
            x=hist_col,
            nbins=30,
            title=f"Histograma de {hist_col}",
            template=plotly_template,
        )
        st.plotly_chart(fig_hist, width="stretch")

        box_num = st.selectbox("Boxplot Plotly - variable", columnas_numericas, key="px_box_num")
        box_cat = st.selectbox("Boxplot Plotly - agrupacion", columnas_categoricas, key="px_box_cat")
        fig_box = px.box(
            filtered,
            x=box_cat,
            y=box_num,
            color=box_cat,
            title=f"Boxplot de {box_num} por {box_cat}",
            template=plotly_template,
        )
        st.plotly_chart(fig_box, width="stretch")

        corr = filtered[columnas_numericas].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Blues",
            title="Heatmap de correlacion",
            template=plotly_template,
        )
        st.plotly_chart(fig_corr, width="stretch")

        color_map = {"Si": "#2E8B57", "No": "#D9534F"}

        fig_bar = px.histogram(
            filtered,
            y="tipo_propiedad",
            color="servicio_aire_acondicionado",
            barmode="stack",
            title="Cantidad de Airbnb por tipo de propiedad",
            template=plotly_template,
            color_discrete_map=color_map,
        )
        col_bar, col_pie = st.columns((3, 2))
        col_bar.plotly_chart(fig_bar, width="stretch")

        fig_pie = px.pie(
            filtered,
            names="servicio_aire_acondicionado",
            title="Proporcion de Airbnb con y sin aire acondicionado",
            template=plotly_template,
            color="servicio_aire_acondicionado",
            color_discrete_map=color_map,
        )
        col_pie.plotly_chart(fig_pie, width="stretch")
