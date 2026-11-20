"""Tablas explorables con búsqueda, orden y descarga."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializa un DataFrame a CSV en memoria."""
    return df.to_csv(index=False).encode("utf-8-sig")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serializa un DataFrame a XLSX en memoria."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resultados")
    return buffer.getvalue()


def render_downloads(df: pd.DataFrame, nombre: str, clave: str) -> None:
    """Botones de descarga en CSV y Excel."""
    if df.empty:
        return
    columnas = st.columns(2)
    with columnas[0]:
        st.download_button(
            "⬇️ Descargar CSV",
            data=df_to_csv_bytes(df),
            file_name=f"{nombre}.csv",
            mime="text/csv",
            key=f"descarga_csv_{clave}",
            width="stretch",
        )
    with columnas[1]:
        st.download_button(
            "⬇️ Descargar Excel",
            data=df_to_excel_bytes(df),
            file_name=f"{nombre}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"descarga_xlsx_{clave}",
            width="stretch",
        )


def render_table(
    df: pd.DataFrame,
    titulo: str,
    clave: str,
    busqueda: bool = True,
    altura: int = 420,
    descargar: bool = True,
    columnas_visibles: list[str] | None = None,
) -> pd.DataFrame:
    """Muestra una tabla con búsqueda, orden nativo y descargas."""
    st.markdown(f"**{titulo}**")
    if df is None or df.empty:
        st.info("No hay filas para mostrar con los filtros actuales.")
        return pd.DataFrame()

    vista = df
    if columnas_visibles:
        vista = vista[[c for c in columnas_visibles if c in vista.columns]]
    if busqueda:
        termino = st.text_input(
            "Buscar en la tabla",
            value="",
            key=f"busqueda_tabla_{clave}",
            help="Filtra filas por coincidencia de texto en cualquier columna.",
        )
        if termino.strip():
            mask = vista.astype(str).apply(
                lambda col: col.str.contains(termino.strip(), case=False, na=False)
            ).any(axis=1)
            vista = vista[mask]

    st.caption(f"{len(vista):,} filas de {len(df):,}")
    st.dataframe(
        vista,
        width="stretch",
        hide_index=True,
        height=min(altura, max(120, 38 * min(len(vista) + 1, 12))),
    )
    if descargar:
        render_downloads(vista, nombre=clave, clave=clave)
    return vista
