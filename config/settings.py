"""Configuración central de la aplicación de análisis de carga eléctrica.

Todas las constantes analíticas reproducen literalmente los valores usados en el
notebook ``Copia_de_proyecto11.ipynb``. Las rutas son relativas al proyecto o se
leen desde variables de entorno; no hay rutas absolutas en el código.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv es opcional en ejecuciones mínimas.
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
STYLES_PATH = ASSETS_DIR / "styles.css"

# La variable INVENTARIO_DATA_PATH permite apuntar al archivo original sin
# incluir rutas privadas en el código fuente.
ENV_DATA_PATH = os.getenv("INVENTARIO_DATA_PATH", "").strip()
LOCAL_DATA_FILENAME = "inventario_carga_.xlsx"

APP_TITLE = "Análisis de Carga Eléctrica y Modelos Predictivos"
APP_SUBTITLE = "Exploración, segmentación, series temporales y balance energético"
APP_DESCRIPTION = (
    "Aplicación interactiva que traduce el notebook original a un flujo guiado: "
    "carga tus datos, explora patrones de consumo, entrena modelos y simula "
    "escenarios energéticos y financieros."
)

COL_FECHA = "Fecha"
COL_MES = "Mes"
COL_ELECTRO = "Electrodomestico"
COL_POTENCIA = "Potencia_W"
COL_HORAS = "Horas_Uso_Dia"
COL_CONSUMO_KWH = "Consumo_kWh"
COL_H_ACTIVA = "H_activa"
COL_TOTAL = "Consumo_Diario_Total"
COL_TEORICO = "Consumo_Teorico"
COL_CLUSTER = "Cluster"
COL_ALTO = "Alto_Consumo"

REQUIRED_COLUMNS = (
    COL_FECHA,
    COL_MES,
    COL_ELECTRO,
    COL_POTENCIA,
    COL_HORAS,
    COL_CONSUMO_KWH,
    COL_H_ACTIVA,
    COL_TOTAL,
)

NUMERIC_COLUMNS = (
    COL_POTENCIA,
    COL_HORAS,
    COL_CONSUMO_KWH,
    COL_H_ACTIVA,
    COL_TOTAL,
)

DATETIME_COLUMNS = (COL_FECHA, COL_MES)

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls", ".json", ".parquet")

# Parámetros de modelos (valores del notebook).
RANDOM_STATE = 42
REG_TEST_SIZE = 0.20
REG_EPOCHS = 50
REG_BATCH_SIZE = 32
REG_LEARNING_RATE = 0.01
REG_VALIDATION_SPLIT = 0.1
N_CLUSTERS = 3
CLF_TEST_SIZE = 0.30
CLF_EPOCHS = 50
CLF_BATCH_SIZE = 4
CLF_THRESHOLD = 0.5
RNN_SEQUENCE_LENGTH = 5
RNN_EPOCHS = 50
RNN_BATCH_SIZE = 4
RNN_TRAIN_FRACTION = 0.80
RNN_LEGACY_START = "2026-01-01"

# Parámetros energéticos (celdas 18-24 del notebook).
FRANJAS = ("Madrugada", "Manana", "Tarde", "Noche")

PERFIL_HORARIO = (
    ("Aire Acondicionado", 6.488, 0.10, 0.20, 0.35, 0.35),
    ("Nevera", 4.542, 0.25, 0.25, 0.25, 0.25),
    ("Lavadora", 1.197, 0.00, 0.60, 0.40, 0.00),
    ("Computadora", 0.899, 0.00, 0.35, 0.35, 0.30),
    ("Iluminación LED", 0.637, 0.05, 0.10, 0.15, 0.70),
    ("Televisor", 0.599, 0.00, 0.10, 0.20, 0.70),
    ("Microondas", 0.500, 0.00, 0.40, 0.30, 0.30),
    ("Calefactor", 0.000, 0.00, 0.00, 0.00, 0.00),
)

# Escenario 1: cargas esenciales (celdas 19-23).
ESCENARIO_1_CONSUMO = {
    "Madrugada": 1.16735,
    "Manana": 1.57375,
    "Tarde": 1.66550,
    "Noche": 2.27040,
}
ESCENARIO_1_CONSUMO_DIARIO = 6.677
ESCENARIO_1_RADIACION = 4.14
ESCENARIO_1_RENDIMIENTO = 0.75
ESCENARIO_1_PANEL_W = 550
ESCENARIO_1_CONSUMO_NOCHE = 2.2704
ESCENARIO_1_MADRUGADA_2H = 1.16735
ESCENARIO_1_GENERACION_FACTORES = {"Madrugada": 0.0, "Manana": 0.35, "Tarde": 0.65, "Noche": 0.0}

# Escenario 2: vivienda completa (celda 24).
ESCENARIO_2_CONSUMO = {
    "Madrugada": 1.81615,
    "Manana": 3.78955,
    "Tarde": 4.56510,
    "Noche": 4.69120,
}
ESCENARIO_2_CONSUMO_DIARIO = 14.862
ESCENARIO_2_RADIACION = 4.14
ESCENARIO_2_RENDIMIENTO = 0.75
ESCENARIO_2_PANEL_W = 550
ESCENARIO_2_GENERACION_FACTORES = {"Madrugada": 0.0, "Manana": 0.35, "Tarde": 0.65, "Noche": 0.0}

DEFAULT_DOD = 0.80
DEFAULT_ETA_INVERSOR = 0.90
DEFAULT_ETA_CARGA = 0.95
SIMULATION_DAYS = 7

# Parámetros financieros (celdas 25-30).
FIN_TASA = 0.10
FIN_DEGRADACION = 0.005
FIN_ANIOS = 25
FIN_TARIFA = 945
FIN_TARIFAS_SENSIBILIDAD = (900, 945, 990)
FIN_FACTORES_INVERSION = (0.90, 1.00, 1.10)
FIN_ANIOS_BATERIA_SENSIBILIDAD = (10, 12, 15)
DEFAULT_ANIO_REPOSICION_BATERIA = 12
DEFAULT_PORCENTAJE_BATERIA = 0.30
DEFAULT_ANIO_REPOSICION_INVERSOR = 15
DEFAULT_PORCENTAJE_INVERSOR = 0.18

FINANCIAL_SCENARIOS = {
    "Cargas esenciales": {
        "inversion": 15_500_000,
        "ahorro_kwh_dia": 6.677 - 0.781,
        "tarifa": FIN_TARIFA,
        "mantenimiento": 470_000,
    },
    "Vivienda completa": {
        "inversion": 32_550_000,
        "ahorro_kwh_dia": 14.862 - 1.211,
        "tarifa": FIN_TARIFA,
        "mantenimiento": 580_000,
    },
}

# Valores impresos por el notebook, usados sólo como referencia histórica.
NOTEBOOK_REFERENCE = {
    "Regresión": {
        "MSE": "2.431655",
        "RMSE": "1.559377",
        "MAE": "1.244944",
        "MAPE": "6.33 %",
        "R2": "0.115809",
    },
    "RNN corregida": {
        "MSE": "2.115687",
        "RMSE": "1.454540 kWh",
        "MAE": "1.173413 kWh",
        "R2": "0.201866",
    },
    "Persistencia": {
        "MAE": "1.469017 kWh",
        "RMSE": "1.825983 kWh",
        "R2": "-0.257817",
    },
    "Clasificación": {"Exactitud": "0.5288194444444444"},
}

NOTEBOOK_CONCLUSIONS = """
1. **Análisis del consumo:** Se realizó un análisis exploratorio del inventario de carga eléctrica, utilizando registros sintéticos de consumo diario y variables asociadas al uso de electrodomésticos.

2. **Modelo de regresión:** Se implementó una red neuronal densa para estimar el consumo diario total. En la evaluación inicial se obtuvo un R² de 0.0884, lo que evidencia una capacidad explicativa limitada y la necesidad de mejorar la configuración y validación del modelo.

3. **Segmentación mediante K-Means:** Se implementó un agrupamiento en tres clústeres. La interpretación de los grupos requiere revisar las variables utilizadas y sus características antes de establecer conclusiones sobre los patrones de consumo de los electrodomésticos.

4. **Análisis de series temporales:** Se desarrolló una segunda versión de una red neuronal SimpleRNN utilizando 1.200 observaciones diarias consecutivas. En el conjunto de prueba obtuvo un MAE de 1.1734 kWh, RMSE de 1.4545 kWh y R² de 0.2019. Su desempeño fue superior al modelo de persistencia evaluado, aunque su capacidad predictiva continúa siendo limitada.

5. **Clasificación binaria:** Se entrenó una red neuronal para diferenciar registros de consumo alto y no alto. La exactitud obtenida fue del 52.88 %, con un desempeño desigual entre las dos clases, por lo que se considera un resultado exploratorio.

6. **Alcance:** Los modelos constituyen ejercicios experimentales desarrollados con datos sintéticos. Sus resultados no representan mediciones reales ni demuestran ahorros energéticos o viabilidad de una instalación fotovoltaica.
"""


def default_data_candidates() -> list[Path]:
    """Devuelve las ubicaciones candidatas del dataset, en orden de prioridad."""
    candidates: list[Path] = []
    if ENV_DATA_PATH:
        candidates.append(Path(ENV_DATA_PATH).expanduser())
    candidates.append(DATA_DIR / LOCAL_DATA_FILENAME)
    return candidates
