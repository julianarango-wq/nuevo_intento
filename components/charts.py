"""Gráficos interactivos con Altair (sin servicios externos)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

COLOR_PRIMARY = "#2563eb"
COLOR_SECOND = "#0f766e"
COLOR_ALERT = "#b45309"


def mostrar(chart: alt.Chart, texto_alt: str) -> None:
    """Renderiza un gráfico con nombre accesible."""
    st.altair_chart(chart, width="stretch")


def histograma(
    df: pd.DataFrame,
    columna: str,
    titulo: str,
    color: str = COLOR_PRIMARY,
    bins: int = 40,
) -> None:
    """Histograma con densidad implícita por conteo."""
    if df.empty or columna not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return
    grafico = (
        alt.Chart(df)
        .mark_bar(color=color, opacity=0.85)
        .encode(
            x=alt.X(f"{columna}:Q", bin=alt.Bin(maxbins=bins), title=columna),
            y=alt.Y("count():Q", title="Frecuencia"),
            tooltip=[alt.Tooltip("count():Q", title="Frecuencia")],
        )
        .properties(height=360, title=titulo)
    )
    mostrar(grafico, titulo)


def dispersion(
    df: pd.DataFrame,
    eje_x: str,
    eje_y: str,
    titulo: str,
    color: str | None = None,
    columnas_tooltip: list[str] | None = None,
) -> None:
    """Gráfico de dispersión con tooltips y color opcional por categoría."""
    if df.empty or eje_x not in df.columns or eje_y not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return
    columnas = [c for c in (columnas_tooltip or [eje_x, eje_y]) if c in df.columns]
    codificacion = {
        "x": alt.X(f"{eje_x}:Q", title=eje_x),
        "y": alt.Y(f"{eje_y}:Q", title=eje_y),
        "tooltip": columnas,
    }
    if color and color in df.columns:
        codificacion["color"] = alt.Color(
            f"{color}:N", title=color, scale=alt.Scale(scheme="tableau10")
        )
    grafico = (
        alt.Chart(df)
        .mark_circle(size=48, opacity=0.7)
        .encode(**codificacion)
        .properties(height=400, title=titulo)
    )
    mostrar(grafico, titulo)


def matriz_correlacion(correlacion: pd.DataFrame, titulo: str = "Matriz de correlación") -> None:
    """Mapa de calor de la matriz de correlación con valores anotados."""
    if correlacion.empty:
        st.info("No hay variables numéricas para calcular correlaciones.")
        return
    datos = (
        correlacion.rename_axis("Variable 1")
        .reset_index()
        .melt(id_vars="Variable 1", var_name="Variable 2", value_name="Correlación")
    )
    base = alt.Chart(datos).encode(
        x=alt.X("Variable 2:N", title=None, sort=list(correlacion.columns), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Variable 1:N", title=None, sort=list(correlacion.index)),
        color=alt.Color(
            "Correlación:Q",
            scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
            legend=alt.Legend(title="Correlación"),
        ),
        tooltip=[
            alt.Tooltip("Variable 1:N", title="Variable 1"),
            alt.Tooltip("Variable 2:N", title="Variable 2"),
            alt.Tooltip("Correlación:Q", format=".3f"),
        ],
    )
    rectangulos = base.mark_rect().properties(width=430, height=380)
    etiquetas = base.mark_text(fontSize=11).encode(
        text=alt.Text("Correlación:Q", format=".2f"),
        color=alt.value("#111827"),
    )
    mostrar((rectangulos + etiquetas).properties(title=titulo), titulo)


def lineas_temporales(
    df: pd.DataFrame,
    columna_x: str,
    series: dict[str, str],
    titulo: str,
) -> None:
    """Compara varias series en el tiempo."""
    if df.empty or columna_x not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return
    columnas = [col for col in series.values() if col in df.columns]
    largo = df.melt(id_vars=columna_x, value_vars=columnas, var_name="Serie", value_name="Valor")
    largo["Serie"] = largo["Serie"].replace({v: k for k, v in series.items()})
    tipo = ":T" if pd.api.types.is_datetime64_any_dtype(df[columna_x]) else ":Q"
    grafico = (
        alt.Chart(largo)
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X(f"{columna_x}{tipo}", title=columna_x),
            y=alt.Y("Valor:Q", title="Consumo (kWh)"),
            color=alt.Color("Serie:N", title="Serie"),
            tooltip=[
                alt.Tooltip(f"{columna_x}{tipo}", title=columna_x),
                alt.Tooltip("Serie:N", title="Serie"),
                alt.Tooltip("Valor:Q", format=".3f"),
            ],
        )
        .properties(height=400, title=titulo)
    )
    mostrar(grafico, titulo)


def matriz_confusion(matriz: np.ndarray, titulo: str = "Matriz de confusión") -> None:
    """Mapa de calor de la matriz de confusión."""
    datos = pd.DataFrame(
        [
            {"Real": real, "Predicción": prediccion, "Casos": int(valor)}
            for real, fila in zip(["Bajo", "Alto"], np.asarray(matriz))
            for prediccion, valor in zip(["Bajo", "Alto"], fila)
        ]
    )
    base = alt.Chart(datos).encode(
        x=alt.X("Predicción:N", title=None, sort=["Bajo", "Alto"]),
        y=alt.Y("Real:N", title=None, sort=["Bajo", "Alto"]),
        color=alt.Color("Casos:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Casos")),
        tooltip=["Real:N", "Predicción:N", "Casos:Q"],
    )
    rectangulos = base.mark_rect().properties(width=380, height=320)
    etiquetas = base.mark_text(fontSize=16).encode(
        text=alt.Text("Casos:Q"), color=alt.value("#111827")
    )
    mostrar((rectangulos + etiquetas).properties(title=titulo), titulo)


def curva_roc(fpr: np.ndarray, tpr: np.ndarray, valor_auc: float) -> None:
    """Curva ROC con la diagonal de referencia."""
    datos = pd.DataFrame(
        {"FPR": np.asarray(fpr), "TPR": np.asarray(tpr)}
    )
    curva = (
        alt.Chart(datos)
        .mark_line(color=COLOR_ALERT, strokeWidth=2)
        .encode(
            x=alt.X("FPR:Q", title="Tasa de falsos positivos", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("TPR:Q", title="Tasa de verdaderos positivos", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color=COLOR_SECOND, strokeDash=[5, 5])
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])))
    )
    mostrar(
        (curva + diagonal).properties(height=380, title=f"Curva ROC (AUC = {valor_auc:.2f})"),
        "Curva ROC",
    )


def barras(
    df: pd.DataFrame,
    eje_x: str,
    eje_y: str,
    titulo: str,
    color: str = COLOR_PRIMARY,
    altura: int = 340,
) -> None:
    """Gráfico de barras verticales."""
    if df.empty or eje_x not in df.columns or eje_y not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return
    grafico = (
        alt.Chart(df)
        .mark_bar(color=color, opacity=0.9)
        .encode(
            x=alt.X(f"{eje_x}:N", title=eje_x, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{eje_y}:Q", title=eje_y),
            tooltip=[alt.Tooltip(f"{eje_x}:N"), alt.Tooltip(f"{eje_y}:Q", format=".3f")],
        )
        .properties(height=altura, title=titulo)
    )
    mostrar(grafico, titulo)


def areas_apiladas(df: pd.DataFrame, eje_x: str, columnas: list[str], titulo: str) -> None:
    """Áreas apiladas para composición de demanda por franja."""
    if df.empty or eje_x not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return
    columnas = [col for col in columnas if col in df.columns]
    largo = df.melt(id_vars=eje_x, value_vars=columnas, var_name="Concepto", value_name="kWh")
    grafico = (
        alt.Chart(largo)
        .mark_area(opacity=0.8)
        .encode(
            x=alt.X(f"{eje_x}:N", title=eje_x),
            y=alt.Y("kWh:Q", title="kWh", stack="zero"),
            color=alt.Color("Concepto:N", title="Concepto"),
            tooltip=[alt.Tooltip(f"{eje_x}:N"), alt.Tooltip("Concepto:N"), alt.Tooltip("kWh:Q", format=".3f")],
        )
        .properties(height=360, title=titulo)
    )
    mostrar(grafico, titulo)


def lineas_flujo(df: pd.DataFrame, eje_x: str, eje_y: str, color_por: str, titulo: str) -> None:
    """Evolución anual de flujos de efectivo por escenario."""
    if df.empty:
        st.info("No hay datos suficientes para este gráfico.")
        return
    grafico = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X(f"{eje_x}:Q", title=eje_x),
            y=alt.Y(f"{eje_y}:Q", title="COP"),
            color=alt.Color(f"{color_por}:N", title=color_por),
            tooltip=[alt.Tooltip(f"{eje_x}:Q"), alt.Tooltip(f"{color_por}:N"), alt.Tooltip(f"{eje_y}:Q", format=",.0f")],
        )
        .properties(height=380, title=titulo)
    )
    mostrar(grafico, titulo)
