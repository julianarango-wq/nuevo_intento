"""Transformaciones y cálculos derivados del notebook (sin lógica de UI)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import (
    COL_ALTO,
    COL_CLUSTER,
    COL_CONSUMO_KWH,
    COL_ELECTRO,
    COL_FECHA,
    COL_H_ACTIVA,
    COL_HORAS,
    COL_POTENCIA,
    COL_TEORICO,
    COL_TOTAL,
    FRANJAS,
    PERFIL_HORARIO,
)


def add_consumo_teorico(df: pd.DataFrame) -> pd.DataFrame:
    """Celda 2: Consumo_Teorico = (Potencia_W * Horas_Uso_Dia) / 1000."""
    result = df.copy()
    result[COL_TEORICO] = (result[COL_POTENCIA] * result[COL_HORAS]) / 1000.0
    return result


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Celda 5: Alto_Consumo = 1 si el consumo supera su mediana."""
    result = df.copy()
    mediana = result[COL_TOTAL].median()
    result[COL_ALTO] = (result[COL_TOTAL] > mediana).astype(int)
    return result


def build_daily_series(df: pd.DataFrame) -> pd.Series:
    """Celdas 7 y 16: un único Consumo_Diario_Total por fecha."""
    return (
        df.groupby(COL_FECHA)[COL_TOTAL]
        .first()
        .sort_index()
        .dropna()
    )


def daily_continuity(series: pd.Series) -> dict:
    """Celda 8: comprueba la continuidad diaria de la serie consolidada."""
    fechas_esperadas = pd.date_range(
        start=series.index.min(), end=series.index.max(), freq="D"
    )
    fechas_faltantes = fechas_esperadas.difference(series.index)
    return {
        "dias_esperados": len(fechas_esperadas),
        "dias_disponibles": len(series),
        "dias_faltantes": len(fechas_faltantes),
        "fecha_inicial": series.index.min(),
        "fecha_final": series.index.max(),
        "duplicados": bool(series.index.duplicated().any()),
        "nulos": int(series.isna().sum()),
        "fechas_faltantes": list(fechas_faltantes[:20]),
        "continua": len(fechas_faltantes) == 0,
    }


def daily_stats(series: pd.Series) -> dict:
    """Celda 16: estadísticas del consumo diario consolidado."""
    return {
        "promedio": round(float(series.mean()), 3),
        "minimo": round(float(series.min()), 3),
        "maximo": round(float(series.max()), 3),
        "desviacion": round(float(series.std()), 3),
    }


def apply_filters(
    df: pd.DataFrame,
    fecha_range: tuple | list | None,
    appliances: list[str] | None,
    numeric_ranges: dict[str, tuple[float, float]] | None,
    query: str,
) -> pd.DataFrame:
    """Aplica fecha, categorías, rangos numéricos y búsqueda de texto."""
    result = df
    if fecha_range and len(fecha_range) == 2:
        inicio, fin = pd.to_datetime(fecha_range[0]), pd.to_datetime(fecha_range[1])
        result = result[
            (result[COL_FECHA] >= inicio) & (result[COL_FECHA] <= fin)
        ]
    if appliances:
        result = result[result[COL_ELECTRO].isin(appliances)]
    for column, (low, high) in (numeric_ranges or {}).items():
        if column in result.columns:
            result = result[
                (result[column] >= low) & (result[column] <= high)
            ]
    if query and query.strip():
        mask = result.astype(str).apply(
            lambda col: col.str.contains(query.strip(), case=False, na=False)
        ).any(axis=1)
        result = result[mask]
    return result.reset_index(drop=True)


def appliance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Celda 17: inventario de respaldo por electrodoméstico."""
    return (
        df.groupby(COL_ELECTRO)
        .agg(
            Potencia_W=(COL_POTENCIA, "mean"),
            Horas_Uso_Dia=(COL_HORAS, "mean"),
            Consumo_kWh=(COL_CONSUMO_KWH, "mean"),
        )
        .round(3)
        .sort_values(COL_CONSUMO_KWH, ascending=False)
    )


def cluster_profile(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Celdas 13 y 14: promedios y cantidad de registros por clúster."""
    profile = df.groupby(COL_CLUSTER)[numeric_columns].mean().round(3)
    counts = df[COL_CLUSTER].value_counts().sort_index()
    return profile, counts


def hours_activity_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Celda 15: estadísticas, correlación y promedios por electrodoméstico."""
    stats = df[[COL_HORAS, COL_H_ACTIVA]].describe().round(3)
    correlation = df[[COL_HORAS, COL_H_ACTIVA]].corr().round(3)
    by_appliance = (
        df.groupby(COL_ELECTRO)[[COL_HORAS, COL_H_ACTIVA]].mean().round(3)
    )
    return stats, correlation, by_appliance


def build_hourly_profile() -> tuple[pd.DataFrame, pd.Series]:
    """Celda 18: perfil horario y consumo estimado por franja."""
    profile = pd.DataFrame(
        PERFIL_HORARIO,
        columns=[
            COL_ELECTRO,
            "Consumo_kWh_dia",
            "Madrugada",
            "Manana",
            "Tarde",
            "Noche",
        ],
    )
    for franja in FRANJAS:
        profile[franja + "_kWh"] = profile["Consumo_kWh_dia"] * profile[franja]
    resumen = profile[[franja + "_kWh" for franja in FRANJAS]].sum()
    return profile, resumen


def size_battery(
    consumo_noche: float,
    consumo_madrugada: float,
    horas_madrugada: float = 6.0,
    horas_respaldo: float = 2.0,
    profundidad_descarga: float = 0.80,
    eficiencia_inversor: float = 0.90,
) -> dict:
    """Celdas 19 y 24: capacidad nominal preliminar de batería."""
    energia_respaldo = consumo_noche + consumo_madrugada * (
        horas_respaldo / horas_madrugada
    )
    capacidad = energia_respaldo / (profundidad_descarga * eficiencia_inversor)
    return {
        "energia_respaldo": energia_respaldo,
        "capacidad_bateria": capacidad,
        "capacidad_util": capacidad * profundidad_descarga,
    }


def size_pv(
    consumo_diario: float,
    radiacion_solar: float,
    rendimiento_sistema: float,
    potencia_panel_w: float,
) -> dict:
    """Celdas 20 y 24: dimensionamiento fotovoltaico preliminar."""
    potencia_fv = consumo_diario / (radiacion_solar * rendimiento_sistema)
    cantidad_paneles = int(potencia_fv * 1000 / potencia_panel_w)
    if cantidad_paneles * potencia_panel_w < potencia_fv * 1000:
        cantidad_paneles += 1
    potencia_instalada = cantidad_paneles * potencia_panel_w / 1000.0
    generacion_diaria = potencia_instalada * radiacion_solar * rendimiento_sistema
    return {
        "potencia_fv": potencia_fv,
        "cantidad_paneles": cantidad_paneles,
        "potencia_instalada": potencia_instalada,
        "generacion_diaria": generacion_diaria,
    }


def build_generation(generacion_diaria: float, factores: dict[str, float]) -> dict[str, float]:
    """Distribuye la generación diaria entre franjas según el notebook."""
    return {franja: generacion_diaria * factores[franja] for franja in FRANJAS}


def simulate_hybrid_balance(
    consumo: dict[str, float],
    generacion: dict[str, float],
    capacidad_bateria: float,
    profundidad_descarga: float = 0.80,
    eficiencia_inversor: float = 0.90,
    eficiencia_carga: float = 0.95,
    estado_inicial: float | None = None,
) -> tuple[pd.DataFrame, float, dict]:
    """Celdas 21 y 22: balance híbrido por franjas con prioridades de despacho.

    Prioridad: consumo solar directo → carga de batería → descarga de batería
    → red eléctrica.
    """
    capacidad_util = capacidad_bateria * profundidad_descarga
    estado_bateria = capacidad_util if estado_inicial is None else estado_inicial
    filas = []
    for franja in FRANJAS:
        demanda = consumo[franja]
        solar = generacion[franja]
        solar_directa = min(demanda, solar)
        demanda_restante = demanda - solar_directa
        excedente_solar = solar - solar_directa

        espacio_bateria = capacidad_util - estado_bateria
        energia_carga = min(excedente_solar * eficiencia_carga, espacio_bateria)
        estado_bateria += energia_carga

        energia_disponible_ac = estado_bateria * eficiencia_inversor
        descarga_ac = min(demanda_restante, energia_disponible_ac)
        energia_extraida = descarga_ac / eficiencia_inversor
        estado_bateria -= energia_extraida

        energia_red = demanda_restante - descarga_ac
        excedente_no_aprovechado = max(
            0.0, excedente_solar - energia_carga / eficiencia_carga
        )
        filas.append(
            {
                "Franja": franja,
                "Demanda_kWh": demanda,
                "Generacion_kWh": solar,
                "Solar_directa_kWh": solar_directa,
                "Carga_bateria_kWh": energia_carga,
                "Descarga_bateria_AC_kWh": descarga_ac,
                "Red_kWh": energia_red,
                "Excedente_no_aprovechado_kWh": excedente_no_aprovechado,
                "Estado_bateria_kWh": estado_bateria,
            }
        )

    balance = pd.DataFrame(filas)
    totales = {
        "consumo_total": balance["Demanda_kWh"].sum(),
        "generacion_solar": balance["Generacion_kWh"].sum(),
        "energia_red": balance["Red_kWh"].sum(),
    }
    return balance, estado_bateria, totales


def simulate_multiday(
    consumo: dict[str, float],
    generacion: dict[str, float],
    capacidad_bateria: float,
    dias: int = 7,
    profundidad_descarga: float = 0.80,
    eficiencia_inversor: float = 0.90,
    eficiencia_carga: float = 0.95,
) -> pd.DataFrame:
    """Celdas 23 y 24: simulación de varios días con batería inicialmente vacía."""
    estado_bateria = 0.0
    filas = []
    for dia in range(1, dias + 1):
        balance, estado_bateria, _ = simulate_hybrid_balance(
            consumo,
            generacion,
            capacidad_bateria,
            profundidad_descarga,
            eficiencia_inversor,
            eficiencia_carga,
            estado_inicial=estado_bateria,
        )
        filas.append(
            {
                "Dia": dia,
                "Red_kWh": balance["Red_kWh"].sum(),
                "Solar_directa_kWh": balance["Solar_directa_kWh"].sum(),
                "Carga_bateria_kWh": balance["Carga_bateria_kWh"].sum(),
                "Descarga_bateria_AC_kWh": balance["Descarga_bateria_AC_kWh"].sum(),
                "Excedente_kWh": balance["Excedente_no_aprovechado_kWh"].sum(),
                "SOC_final_kWh": estado_bateria,
            }
        )
    return pd.DataFrame(filas)


def safe_range(values: pd.Series) -> tuple[float, float]:
    """Devuelve un rango válido para sliders incluso con series constantes."""
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if math.isclose(low, high):
        padding = abs(low) * 0.01 if low else 1.0
        return low - padding, high + padding
    return low, high
