"""Análisis estadístico, clustering, métricas y evaluación financiera."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from config.settings import RANDOM_STATE


def descriptive_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Celda 1: descripción completa y separación de columnas numéricas/categóricas."""
    descripcion = df.describe(include="all")
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return descripcion, numericas, categoricas


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Celda 1: matriz de correlación de variables numéricas."""
    return df[columns].corr()


def fit_kmeans(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Celdas 3 y 13: K-Means sobre variables numéricas escaladas."""
    scaler = StandardScaler()
    escalado = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    resultado = df.copy()
    resultado["Cluster"] = kmeans.fit_predict(escalado)
    return resultado, kmeans, scaler


def classification_results(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Celdas 5 y 12: matriz de confusión, reporte, ROC y métricas."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    reporte = classification_report(
        y_true,
        y_pred,
        target_names=["Bajo Consumo", "Alto Consumo"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "y_pred": y_pred,
        "matriz": confusion_matrix(y_true, y_pred),
        "reporte": reporte,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc(fpr, tpr),
        "exactitud": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def persistence_baseline(
    serie: pd.Series,
    split_idx: int,
    rnn_metrics: dict,
) -> tuple[dict, pd.DataFrame]:
    """Celda 10: baseline de persistencia frente a la RNN corregida."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    real = serie.iloc[split_idx:].values
    predicho = serie.iloc[split_idx - 1: -1].values
    mae = mean_absolute_error(real, predicho)
    rmse = np.sqrt(mean_squared_error(real, predicho))
    r2 = r2_score(real, predicho)
    metricas = {"MAE_kWh": mae, "RMSE_kWh": rmse, "R2": r2}
    comparacion = pd.DataFrame(
        {
            "Modelo": ["RNN corregida", "Persistencia"],
            "MAE_kWh": [rnn_metrics["MAE"], mae],
            "RMSE_kWh": [rnn_metrics["RMSE"], rmse],
            "R2": [rnn_metrics["R2"], r2],
        }
    )
    return metricas, comparacion


def compute_simple_financials(
    escenarios: dict,
    tasa: float,
    degradacion: float,
    anios: int,
) -> pd.DataFrame:
    """Celdas 25 y 26: VPN y recuperación descontada sin reposiciones."""
    filas = []
    for nombre, parametros in escenarios.items():
        flujo = [-parametros["inversion"]]
        for anio in range(1, anios + 1):
            beneficio = (
                parametros["ahorro_kwh_dia"]
                * 365
                * parametros["tarifa"]
                * (1 - degradacion) ** (anio - 1)
            )
            flujo.append(beneficio - parametros["mantenimiento"])
        vpn = sum(valor / (1 + tasa) ** i for i, valor in enumerate(flujo))
        acumulado = -parametros["inversion"]
        recuperacion = None
        for anio, valor in enumerate(flujo[1:], 1):
            acumulado += valor / (1 + tasa) ** anio
            if acumulado >= 0 and recuperacion is None:
                recuperacion = anio
        filas.append(
            {
                "Escenario": nombre,
                "Inversion_COP": parametros["inversion"],
                "VPN_COP": round(vpn, 2),
                "Recuperacion_descuento_anios": recuperacion,
            }
        )
    return pd.DataFrame(filas)


def compute_financials_with_replacements(
    escenarios: dict,
    tasa: float,
    degradacion: float,
    anios: int,
    anio_bateria: int,
    porcentaje_bateria: float,
    anio_inversor: int,
    porcentaje_inversor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Celdas 27 y 30: VPN, recuperación y flujos detallados con reposiciones."""
    resultados = []
    flujos_detallados = []
    for nombre, parametros in escenarios.items():
        inversion = parametros["inversion"]
        flujo = [-inversion]
        vpn = -inversion
        saldo = -inversion
        recuperacion = None
        costo_bateria = inversion * porcentaje_bateria
        costo_inversor = inversion * porcentaje_inversor

        for anio in range(1, anios + 1):
            beneficio = (
                parametros["ahorro_kwh_dia"]
                * 365
                * parametros["tarifa"]
                * (1 - degradacion) ** (anio - 1)
            )
            mantenimiento = parametros["mantenimiento"]
            reposicion_bateria = costo_bateria if anio == anio_bateria else 0
            reposicion_inversor = costo_inversor if anio == anio_inversor else 0
            neto = (
                beneficio
                - mantenimiento
                - reposicion_bateria
                - reposicion_inversor
            )
            descontado = neto / (1 + tasa) ** anio
            flujo.append(neto)
            vpn += descontado
            saldo += descontado
            if saldo >= 0 and recuperacion is None:
                recuperacion = anio
            flujos_detallados.append(
                {
                    "Escenario": nombre,
                    "Año": anio,
                    "Ahorro_bruto_COP": beneficio,
                    "Mantenimiento_COP": mantenimiento,
                    "Reposicion_bateria_COP": reposicion_bateria,
                    "Reposicion_inversor_COP": reposicion_inversor,
                    "Flujo_neto_COP": neto,
                }
            )
        resultados.append(
            {
                "Escenario": nombre,
                "Inversion_COP": inversion,
                "Reposicion_bateria_COP": costo_bateria,
                "Reposicion_inversor_COP": costo_inversor,
                "VPN_COP": round(vpn, 2),
                "Recuperacion_descuento_anios": recuperacion,
                "Saldo_final_COP": round(saldo, 2),
            }
        )
    return pd.DataFrame(resultados), pd.DataFrame(flujos_detallados)


def compute_sensitivity(
    escenarios: dict,
    tasa: float,
    degradacion: float,
    anios: int,
    tarifas: tuple[int, ...],
    factores_inversion: tuple[float, ...],
    anios_bateria: tuple[int, ...],
    porcentaje_bateria: float = 0.30,
    anio_inversor: int = 15,
    porcentaje_inversor: float = 0.18,
) -> pd.DataFrame:
    """Celda 28: sensibilidad de VPN y recuperación a tarifa, inversión y reposición."""
    filas = []
    for nombre, parametros in escenarios.items():
        for tarifa in tarifas:
            for factor in factores_inversion:
                for anio_bateria in anios_bateria:
                    inversion = parametros["inversion"] * factor
                    bateria = inversion * porcentaje_bateria
                    inversor = inversion * porcentaje_inversor
                    vpn = -inversion
                    saldo = -inversion
                    recuperacion = None
                    for anio in range(1, anios + 1):
                        beneficio = (
                            parametros["ahorro_kwh_dia"]
                            * 365
                            * tarifa
                            * (1 - degradacion) ** (anio - 1)
                        )
                        repos_bateria = bateria if anio == anio_bateria else 0
                        repos_inversor = inversor if anio == anio_inversor else 0
                        flujo = (
                            beneficio
                            - parametros["mantenimiento"]
                            - repos_bateria
                            - repos_inversor
                        )
                        descontado = flujo / (1 + tasa) ** anio
                        vpn += descontado
                        saldo += descontado
                        if saldo >= 0 and recuperacion is None:
                            recuperacion = anio
                    filas.append(
                        {
                            "Escenario": nombre,
                            "Tarifa_COP_kWh": tarifa,
                            "Factor_inversion": factor,
                            "Anio_reposicion_bateria": anio_bateria,
                            "VPN_COP": round(vpn, 0),
                            "Recuperacion_anios": recuperacion,
                        }
                    )
    return pd.DataFrame(filas)


def summarize_sensitivity(sensibilidad: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Celda 29: resumen y porcentaje de escenarios con VPN positivo."""
    resumen = (
        sensibilidad.groupby("Escenario")
        .agg(
            VPN_minimo=("VPN_COP", "min"),
            VPN_promedio=("VPN_COP", "mean"),
            VPN_maximo=("VPN_COP", "max"),
            Escenarios_favorables=("VPN_COP", lambda x: (x > 0).sum()),
            Total_escenarios=("VPN_COP", "count"),
        )
        .round(0)
    )
    porcentaje = (
        resumen["Escenarios_favorables"] / resumen["Total_escenarios"] * 100
    ).round(2)
    return resumen, porcentaje
