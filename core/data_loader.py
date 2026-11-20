"""Carga, validación y normalización del dataset de inventario de carga."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import (
    DATETIME_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    SUPPORTED_EXTENSIONS,
    default_data_candidates,
)


def extension_of(name: str) -> str:
    """Devuelve la extensión normalizada de un nombre de archivo."""
    suffix = Path(name).suffix.lower()
    if suffix == ".xls":
        return ".xls"
    return suffix if suffix in SUPPORTED_EXTENSIONS else ""


@st.cache_data(show_spinner=False, max_entries=20)
def read_uploaded_bytes(payload: bytes, suffix: str) -> pd.DataFrame:
    """Lee un archivo cargado por el usuario según su extensión."""
    if not payload:
        raise ValueError("El archivo cargado está vacío.")
    buffer = io.BytesIO(payload)
    if suffix == ".csv":
        try:
            return pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="latin-1")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(buffer)
    if suffix == ".json":
        return pd.read_json(buffer)
    if suffix == ".parquet":
        return pd.read_parquet(buffer)
    raise ValueError(f"Formato no soportado: {suffix or 'desconocido'}.")


@st.cache_data(show_spinner=False, max_entries=20)
def read_local_path(path: str, modification_time: float) -> pd.DataFrame:
    """Lee un archivo local. ``modification_time`` invalida la caché si cambia."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    suffix = extension_of(file_path.name)
    if not suffix:
        raise ValueError(f"Extensión no soportada: {file_path.suffix}")
    if suffix == ".csv":
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding="latin-1")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    return pd.read_parquet(file_path)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de columnas obligatorias ausentes."""
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas y convierte fechas y numéricas a sus tipos correctos.

    No elimina filas: conserva todas las observaciones del notebook original.
    """
    missing = validate_schema(df)
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(missing)
            + ". Revisa la estructura descrita en data/README.md."
        )
    result = df.copy()
    for column in DATETIME_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def find_default_data() -> Path | None:
    """Devuelve el primer archivo de datos disponible en las rutas configuradas."""
    for candidate in default_data_candidates():
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def load_from_path(path: Path) -> pd.DataFrame:
    """Carga y normaliza un archivo local."""
    return normalize(read_local_path(str(path), path.stat().st_mtime))


def load_from_upload(upload) -> pd.DataFrame:
    """Carga y normaliza un archivo subido con ``st.file_uploader``."""
    suffix = extension_of(upload.name)
    if not suffix:
        raise ValueError(
            "Formato no soportado. Usa CSV, Excel (.xlsx/.xls), JSON o Parquet."
        )
    payload = upload.getvalue()
    upload.seek(0)
    return normalize(read_uploaded_bytes(payload, suffix))


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de tipo, valores nulos y cardinalidad por columna."""
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "Columna": column,
                "Tipo": str(series.dtype),
                "No nulos": int(series.notna().sum()),
                "Nulos": int(series.isna().sum()),
                "Valores únicos": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)
