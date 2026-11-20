"""Tarjetas de indicadores con ``st.metric``."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config.settings import COL_ELECTRO, NOTEBOOK_REFERENCE


def _fmt(valor: float, decimales: int = 3) -> str:
    return f"{valor:,.{decimales}f}"


def render_dataset_metrics(df: pd.DataFrame, serie: pd.Series, stats: dict) -> None:
    """Indicadores generales del dataset y de la serie diaria consolidada."""
    columnas = st.columns(4, border=True)
    with columnas[0]:
        st.metric("Registros", f"{len(df):,}", help="Filas después de aplicar filtros.")
    with columnas[1]:
        st.metric("Electrodomésticos", df[COL_ELECTRO].nunique(), help="Categorías distintas.")
    with columnas[2]:
        st.metric("Días con datos", len(serie), help="Fechas únicas en la serie consolidada.")
    with columnas[3]:
        st.metric(
            "Consumo diario promedio",
            f"{_fmt(stats['promedio'])} kWh",
            help="Celda 16 del notebook: promedio de Consumo_Diario_Total por fecha.",
        )
    columnas = st.columns(3, border=True)
    with columnas[0]:
        st.metric("Consumo diario mínimo", f"{_fmt(stats['minimo'])} kWh")
    with columnas[1]:
        st.metric("Consumo diario máximo", f"{_fmt(stats['maximo'])} kWh")
    with columnas[2]:
        st.metric("Desviación estándar", f"{_fmt(stats['desviacion'])} kWh")


def render_regression_metrics(metricas: dict) -> None:
    """Métricas de la red densa de regresión."""
    columnas = st.columns(5, border=True)
    etiquetas = [
        ("MSE", "MSE", 6),
        ("RMSE", "RMSE (kWh)", 6),
        ("MAE", "MAE (kWh)", 6),
        ("MAPE", "MAPE", 4),
        ("R2", "R²", 6),
    ]
    for columna, (clave, etiqueta, decimales) in zip(columnas, etiquetas):
        valor = metricas[clave] * 100 if clave == "MAPE" else metricas[clave]
        sufijo = " %" if clave == "MAPE" else ""
        with columna:
            st.metric(etiqueta, f"{valor:,.{decimales}f}{sufijo}")
    st.caption(
        "Valores de referencia impresos en el notebook: "
        + ", ".join(
            f"{clave} = {valor}" for clave, valor in NOTEBOOK_REFERENCE["Regresión"].items()
        )
    )


def render_classification_metrics(res: dict) -> None:
    """Métricas del clasificador binario."""
    columnas = st.columns(4, border=True)
    datos = [
        ("Exactitud", res["exactitud"]),
        ("Precisión", res["precision"]),
        ("Recall", res["recall"]),
        ("F1-score", res["f1"]),
    ]
    for columna, (etiqueta, valor) in zip(columnas, datos):
        with columna:
            st.metric(etiqueta, f"{valor:.4f}")
    st.caption(
        "Exactitud reportada en el notebook: "
        + NOTEBOOK_REFERENCE["Clasificación"]["Exactitud"]
    )


def render_rnn_metrics(metricas: dict, referencia: str | None = None) -> None:
    """Métricas de la red recurrente."""
    columnas = st.columns(4, border=True)
    for columna, (clave, etiqueta) in zip(
        columnas,
        [("MSE", "MSE"), ("RMSE", "RMSE (kWh)"), ("MAE", "MAE (kWh)"), ("R2", "R²")],
    ):
        with columna:
            st.metric(etiqueta, f"{metricas[clave]:,.6f}")
    if referencia:
        st.caption(
            "Valores de referencia impresos en el notebook ("
            + referencia
            + "): "
            + ", ".join(
                f"{clave} = {valor}"
                for clave, valor in NOTEBOOK_REFERENCE[referencia].items()
            )
        )


def render_energy_metrics(sizing: dict, battery: dict) -> None:
    """Indicadores de dimensionamiento fotovoltaico y de batería."""
    columnas = st.columns(4, border=True)
    valores = [
        ("Potencia FV requerida", f"{sizing['potencia_fv']:.2f} kWp"),
        ("Paneles", str(sizing["cantidad_paneles"])),
        ("Potencia instalada", f"{sizing['potencia_instalada']:.2f} kWp"),
        ("Generación diaria", f"{sizing['generacion_diaria']:.2f} kWh"),
    ]
    for columna, (etiqueta, valor) in zip(columnas, valores):
        with columna:
            st.metric(etiqueta, valor)
    columnas = st.columns(3, border=True)
    for columna, (etiqueta, valor) in zip(
        columnas,
        [
            ("Energía para respaldo", f"{battery['energia_respaldo']:.4f} kWh"),
            ("Batería nominal", f"{battery['capacidad_bateria']:.2f} kWh"),
            ("Capacidad útil", f"{battery['capacidad_util']:.2f} kWh"),
        ],
    ):
        with columna:
            st.metric(etiqueta, valor)


def render_financial_metrics(resultados: pd.DataFrame) -> None:
    """VPN y recuperación descontada por escenario."""
    for _, fila in resultados.iterrows():
        st.markdown(f"**{fila['Escenario']}**")
        columnas = st.columns(3, border=True)
        with columnas[0]:
            st.metric("Inversión", f"COP {fila['Inversion_COP']:,.0f}")
        with columnas[1]:
            vpn = fila["VPN_COP"]
            st.metric("VPN", f"COP {vpn:,.0f}", delta=f"positivo" if vpn > 0 else "negativo")
        with columnas[2]:
            recuperacion = fila["Recuperacion_descuento_anios"]
            st.metric(
                "Recuperación descontada",
                "No se alcanza" if pd.isna(recuperacion) else f"{int(recuperacion)} años",
            )
