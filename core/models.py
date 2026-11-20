"""Modelos del notebook con carga diferida de TensorFlow.

Las redes neuronales se mantienen con la arquitectura y los hiperparámetros
originales. TensorFlow es opcional: se importa sólo cuando se entrena un modelo,
de modo que el resto de la aplicación funciona con dependencias ligeras.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import (
    CLF_BATCH_SIZE,
    CLF_EPOCHS,
    CLF_TEST_SIZE,
    COL_CONSUMO_KWH,
    COL_ELECTRO,
    COL_HORAS,
    COL_POTENCIA,
    COL_TEORICO,
    COL_TOTAL,
    RANDOM_STATE,
    REG_BATCH_SIZE,
    REG_EPOCHS,
    REG_LEARNING_RATE,
    REG_TEST_SIZE,
    REG_VALIDATION_SPLIT,
    RNN_BATCH_SIZE,
    RNN_EPOCHS,
    RNN_LEGACY_START,
    RNN_SEQUENCE_LENGTH,
    RNN_TRAIN_FRACTION,
)
from core.processing import add_binary_target, add_consumo_teorico


@lru_cache(maxsize=1)
def tf_available() -> bool:
    """Comprueba una sola vez si TensorFlow está disponible."""
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def _require_tensorflow():
    """Importa TensorFlow o explica cómo instalarlo."""
    if not tf_available():
        raise RuntimeError(
            "TensorFlow no está instalado en este entorno. Para habilitar el "
            "entrenamiento de redes neuronales ejecuta: "
            "pip install -r requirements-ml.txt"
        )
    import tensorflow as tf

    return tf


def tensorflow_message() -> str:
    """Mensaje de estado para la interfaz."""
    if tf_available():
        return "TensorFlow disponible: puedes entrenar las redes neuronales."
    return (
        "TensorFlow no está instalado. Las secciones de datos, clustering, "
        "energía y finanzas funcionan normalmente. Para entrenar las redes "
        "ejecuta `pip install -r requirements-ml.txt` y reinicia la aplicación."
    )


@dataclass
class RegressionResult:
    """Resultado de la red densa de regresión (celda 2)."""

    preprocessor: ColumnTransformer
    model: object
    history: object
    metrics: dict
    y_true: np.ndarray
    y_pred: np.ndarray
    errores: np.ndarray
    feature_names: list[str]


@dataclass
class ClassifierResult:
    """Resultado de la red binaria de clasificación (celda 5)."""

    model: object
    X_test: pd.DataFrame
    y_test: np.ndarray
    y_prob: np.ndarray
    mediana: float


@dataclass
class RNNResult:
    """Resultado de las RNN simple (celdas 4 y 9)."""

    model: object
    history: object
    y_real: np.ndarray
    y_pred: np.ndarray
    fechas: np.ndarray | None
    metrics: dict
    split_idx: int
    version: str


@st.cache_resource(show_spinner=False)
def train_regression_model(
    df: pd.DataFrame,
    epochs: int = REG_EPOCHS,
    batch_size: int = REG_BATCH_SIZE,
    learning_rate: float = REG_LEARNING_RATE,
    test_size: float = REG_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> RegressionResult:
    """Celdas 2 y 3: preprocesamiento y red densa para Consumo_Diario_Total."""
    tf = _require_tensorflow()
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    trabajado = add_consumo_teorico(df)
    features = [
        COL_HORAS,
        COL_POTENCIA,
        COL_CONSUMO_KWH,
        COL_TEORICO,
        COL_ELECTRO,
    ]
    X = trabajado[features].copy()
    y = trabajado[COL_TOTAL].values

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [COL_HORAS, COL_POTENCIA, COL_CONSUMO_KWH, COL_TEORICO],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [COL_ELECTRO]),
        ]
    )
    X_proc = preprocessor.fit_transform(X)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=test_size, random_state=random_state
    )

    modelo = Sequential(
        [
            tf.keras.Input(shape=(X_train.shape[1],)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    history = modelo.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=REG_VALIDATION_SPLIT,
    )
    y_pred = modelo.predict(X_test, verbose=0).flatten()
    mse = mean_squared_error(y_test, y_pred)
    metricas = {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }
    return RegressionResult(
        preprocessor=preprocessor,
        model=modelo,
        history=history,
        metrics=metricas,
        y_true=np.asarray(y_test),
        y_pred=y_pred,
        errores=np.asarray(y_test) - y_pred,
        feature_names=features,
    )


def predict_consumption(
    resultado: RegressionResult,
    horas: float,
    potencia: float,
    consumo_kwh: float,
    electrodomestico: str,
) -> float:
    """Celda 6: estimación individual de consumo diario total."""
    teorico = (potencia * horas) / 1000.0
    entrada = pd.DataFrame(
        [[horas, potencia, consumo_kwh, teorico, electrodomestico]],
        columns=[
            COL_HORAS,
            COL_POTENCIA,
            COL_CONSUMO_KWH,
            COL_TEORICO,
            COL_ELECTRO,
        ],
    )
    procesada = resultado.preprocessor.transform(entrada)
    if hasattr(procesada, "toarray"):
        procesada = procesada.toarray()
    return float(resultado.model.predict(procesada, verbose=0)[0][0])


@st.cache_resource(show_spinner=False)
def train_classifier_model(
    df: pd.DataFrame,
    epochs: int = CLF_EPOCHS,
    batch_size: int = CLF_BATCH_SIZE,
    test_size: float = CLF_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> ClassifierResult:
    """Celdas 5 y 12: red binaria con Horas_Uso_Dia como única variable."""
    tf = _require_tensorflow()
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential

    trabajado = add_binary_target(df)
    X = trabajado[[COL_HORAS]].copy()
    y = trabajado["Alto_Consumo"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    modelo = Sequential(
        [
            tf.keras.Input(shape=(X_train.shape[1],)),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    modelo.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    probabilidades = modelo.predict(X_test, verbose=0).ravel()
    return ClassifierResult(
        model=modelo,
        X_test=X_test,
        y_test=np.asarray(y_test),
        y_prob=probabilidades,
        mediana=float(trabajado[COL_TOTAL].median()),
    )


def predict_class_probability(resultado: ClassifierResult, horas: float) -> float:
    """Probabilidad de alto consumo para un nuevo valor de horas de uso."""
    entrada = pd.DataFrame([[horas]], columns=[COL_HORAS])
    return float(resultado.model.predict(entrada, verbose=0).ravel()[0])


def _create_sequences(data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Celda 4: construcción de secuencias de la serie temporal."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


@st.cache_resource(show_spinner=False)
def train_legacy_rnn(
    df: pd.DataFrame,
    sequence_length: int = RNN_SEQUENCE_LENGTH,
    epochs: int = RNN_EPOCHS,
    batch_size: int = RNN_BATCH_SIZE,
) -> RNNResult:
    """Celda 4: RNN original con fechas simuladas y escalado de toda la serie."""
    tf = _require_tensorflow()
    from tensorflow.keras.layers import Dense, SimpleRNN
    from tensorflow.keras.models import Sequential

    temporal = df.copy()
    temporal["Fecha"] = pd.date_range(
        start=RNN_LEGACY_START, periods=len(temporal), freq="D"
    )
    temporal = temporal.set_index("Fecha")
    escalador = StandardScaler()
    escalados = escalador.fit_transform(
        temporal[COL_TOTAL].values.reshape(-1, 1)
    )
    X, y = _create_sequences(escalados, sequence_length)
    indice = int(len(X) * 0.8)
    X_train, X_test = X[:indice], X[indice:]
    y_train, y_test = y[:indice], y[indice:]

    modelo = Sequential(
        [
            tf.keras.Input(shape=(sequence_length, 1)),
            SimpleRNN(32, activation="tanh"),
            Dense(1),
        ]
    )
    modelo.compile(optimizer="adam", loss="mse")
    history = modelo.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0
    )
    prediccion = modelo.predict(X_test, verbose=0)
    real = escalador.inverse_transform(y_test).flatten()
    predicho = escalador.inverse_transform(prediccion).flatten()
    return RNNResult(
        model=modelo,
        history=history,
        y_real=real,
        y_pred=predicho,
        fechas=temporal.index[indice + sequence_length :].to_numpy(),
        metrics={
            "MSE": float(mean_squared_error(real, predicho)),
            "RMSE": float(np.sqrt(mean_squared_error(real, predicho))),
            "MAE": float(mean_absolute_error(real, predicho)),
            "R2": float(r2_score(real, predicho)),
        },
        split_idx=indice,
        version="v1 (fechas simuladas)",
    )


@st.cache_resource(show_spinner=False)
def train_corrected_rnn(
    values: np.ndarray,
    dates: np.ndarray,
    sequence_length: int = RNN_SEQUENCE_LENGTH,
    epochs: int = RNN_EPOCHS,
    batch_size: int = RNN_BATCH_SIZE,
    train_fraction: float = RNN_TRAIN_FRACTION,
    random_state: int = RANDOM_STATE,
) -> RNNResult:
    """Celda 9: RNN corregida con split cronológico y escalado sólo con train."""
    tf = _require_tensorflow()
    from tensorflow.keras.layers import Dense, SimpleRNN
    from tensorflow.keras.models import Sequential

    datos = np.asarray(values, dtype=float).reshape(-1, 1)
    indice = int(len(datos) * train_fraction)
    train = datos[:indice]
    test = datos[indice:]
    escalador = StandardScaler()
    train_scaled = escalador.fit_transform(train)
    test_scaled = escalador.transform(test)

    X_train, y_train = _create_sequences(train_scaled, sequence_length)
    contexto = np.vstack([train_scaled[-sequence_length:], test_scaled])
    X_test, y_test = _create_sequences(contexto, sequence_length)

    tf.keras.utils.set_random_seed(random_state)
    modelo = Sequential(
        [
            tf.keras.Input(shape=(sequence_length, 1)),
            SimpleRNN(32, activation="tanh"),
            Dense(1),
        ]
    )
    modelo.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = modelo.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=0,
    )
    prediccion = modelo.predict(X_test, verbose=0)
    real = escalador.inverse_transform(y_test).flatten()
    predicho = escalador.inverse_transform(prediccion).flatten()
    mse = mean_squared_error(real, predicho)
    metricas = {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(real, predicho)),
        "R2": float(r2_score(real, predicho)),
    }
    fechas_test = pd.to_datetime(dates)[indice:]
    return RNNResult(
        model=modelo,
        history=history,
        y_real=real,
        y_pred=predicho,
        fechas=fechas_test.to_numpy(),
        metrics=metricas,
        split_idx=indice,
        version="v2 (serie diaria corregida)",
    )


def build_rnn_results_table(resultado: RNNResult) -> pd.DataFrame:
    """Celda 9: tabla real vs predicho con error absoluto."""
    tabla = pd.DataFrame(
        {
            "Fecha": pd.to_datetime(resultado.fechas),
            "Consumo_Real_kWh": resultado.y_real,
            "Consumo_Predicho_kWh": resultado.y_pred,
        }
    )
    tabla["Error_kWh"] = tabla["Consumo_Real_kWh"] - tabla["Consumo_Predicho_kWh"]
    return tabla
