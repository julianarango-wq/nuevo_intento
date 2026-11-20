"""Barra lateral: origen de datos, filtros y parámetros ajustables."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import (
    COL_ELECTRO,
    COL_FECHA,
    DEFAULT_ANIO_REPOSICION_BATERIA,
    DEFAULT_ANIO_REPOSICION_INVERSOR,
    DEFAULT_DOD,
    DEFAULT_ETA_CARGA,
    DEFAULT_ETA_INVERSOR,
    DEFAULT_PORCENTAJE_BATERIA,
    DEFAULT_PORCENTAJE_INVERSOR,
    FIN_ANIOS,
    FIN_DEGRADACION,
    FIN_TARIFA,
    FIN_TASA,
    N_CLUSTERS,
    NUMERIC_COLUMNS,
    REG_BATCH_SIZE,
    REG_EPOCHS,
    REG_LEARNING_RATE,
    RNN_BATCH_SIZE,
    RNN_EPOCHS,
    RNN_SEQUENCE_LENGTH,
    SUPPORTED_EXTENSIONS,
    default_data_candidates,
)
from core.processing import safe_range


def render_reset_control() -> None:
    """Reinicia filtros y parámetros sin borrar la caché de datos."""
    if st.sidebar.button("🔄 Reiniciar filtros y parámetros", width="stretch"):
        prefijos = ("fuente_", "filtro_", "param_")
        for clave in list(st.session_state.keys()):
            if clave.startswith(prefijos):
                del st.session_state[clave]
        st.rerun()


def render_data_source() -> dict:
    """Selecciona el origen de datos (archivo subido o ruta local configurada)."""
    st.sidebar.header("1. Datos")
    modo = st.sidebar.segmented_control(
        "Origen",
        options=["Subir archivo", "Ruta local"],
        default="Subir archivo",
        key="fuente_modo",
        help="Sube un CSV/Excel/JSON/Parquet o usa la ruta configurada en .env.",
    )
    resultado = {"modo": modo, "upload": None, "path": None}
    if modo == "Subir archivo":
        resultado["upload"] = st.sidebar.file_uploader(
            "Archivo de datos",
            type=list(SUPPORTED_EXTENSIONS),
            key="fuente_upload",
            help="CSV, Excel (.xlsx/.xls), JSON o Parquet con las columnas del notebook.",
        )
    else:
        disponibles = [str(ruta) for ruta in default_data_candidates() if ruta.is_file()]
        sugerencia = disponibles[0] if disponibles else ""
        st.sidebar.text_input(
            "Ruta del archivo",
            value=sugerencia,
            key="fuente_path",
            help="También puedes definir INVENTARIO_DATA_PATH en un archivo .env local.",
        )
        ruta = st.session_state.get("fuente_path", "").strip()
        resultado["path"] = Path(ruta) if ruta else None
        if not sugerencia:
            st.sidebar.caption(
                "Coloca `inventario_carga_.xlsx` en la carpeta `data/` o configura "
                "`INVENTARIO_DATA_PATH`."
            )
    return resultado


def render_filters(df: pd.DataFrame) -> dict:
    """Controles de fecha, categoría, rango numérico y búsqueda."""
    st.sidebar.header("2. Filtros")
    fecha_min = df[COL_FECHA].min()
    fecha_max = df[COL_FECHA].max()
    fechas = st.sidebar.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
        key="filtro_fechas",
    )
    fecha_range = tuple(fechas) if isinstance(fechas, (tuple, list)) and len(fechas) == 2 else None

    electrodomesticos = st.sidebar.multiselect(
        "Electrodomésticos",
        options=sorted(df[COL_ELECTRO].dropna().unique().tolist()),
        default=[],
        key="filtro_electro",
        help="Si no seleccionas ninguno, se incluyen todos.",
    )
    rangos: dict[str, tuple[float, float]] = {}
    with st.sidebar.expander("Rangos numéricos", expanded=False):
        for columna in NUMERIC_COLUMNS:
            if columna not in df.columns:
                continue
            bajo, alto = safe_range(df[columna].dropna())
            rangos[columna] = st.slider(
                columna,
                min_value=bajo,
                max_value=alto,
                value=(bajo, alto),
                key=f"filtro_{columna}",
            )
    busqueda = st.sidebar.text_input(
        "Buscar texto",
        value="",
        key="filtro_busqueda",
        help="Busca en todas las columnas, sin distinción de mayúsculas.",
    )
    return {
        "fecha_range": fecha_range,
        "appliances": electrodomesticos,
        "numeric_ranges": rangos,
        "query": busqueda,
    }


def render_parameters() -> dict:
    """Hiperparámetros de modelos, energía y análisis financiero."""
    parametros: dict = {}

    with st.sidebar.expander("3. Modelos", expanded=False):
        parametros["n_clusters"] = st.slider(
            "Cantidad de clústeres (K-Means)",
            2, 8, N_CLUSTERS, key="param_clusters",
        )
        parametros["reg_epochs"] = st.slider(
            "Épocas (regresión)", 5, 200, REG_EPOCHS, key="param_reg_epochs"
        )
        parametros["reg_batch"] = st.slider(
            "Tamaño de lote (regresión)", 8, 128, REG_BATCH_SIZE, key="param_reg_batch"
        )
        parametros["reg_lr"] = st.number_input(
            "Tasa de aprendizaje (regresión)",
            min_value=0.0001, max_value=0.5, value=REG_LEARNING_RATE,
            step=0.0001, format="%.4f", key="param_reg_lr",
        )
        parametros["clf_epochs"] = st.slider(
            "Épocas (clasificación)", 5, 200, 50, key="param_clf_epochs"
        )
        parametros["clf_batch"] = st.slider(
            "Tamaño de lote (clasificación)", 1, 64, 4, key="param_clf_batch"
        )
        parametros["rnn_version"] = st.segmented_control(
            "Versión de RNN",
            options=["v2", "v1"],
            default="v2",
            key="param_rnn_version",
            help="v2 usa la serie diaria corregida; v1 reproduce las fechas simuladas del notebook.",
        )
        parametros["rnn_seq"] = st.slider(
            "Longitud de secuencia", 2, 15, RNN_SEQUENCE_LENGTH, key="param_rnn_seq"
        )
        parametros["rnn_epochs"] = st.slider(
            "Épocas (RNN)", 5, 200, RNN_EPOCHS, key="param_rnn_epochs"
        )
        parametros["rnn_batch"] = st.slider(
            "Tamaño de lote (RNN)", 1, 64, RNN_BATCH_SIZE, key="param_rnn_batch"
        )

    with st.sidebar.expander("4. Energía", expanded=False):
        parametros["radiacion"] = st.slider(
            "Radiación solar (kWh/m²·día)", 1.0, 8.0, 4.14, 0.01, key="param_radiacion"
        )
        parametros["rendimiento"] = st.slider(
            "Rendimiento del sistema", 0.30, 1.00, 0.75, 0.01, key="param_rendimiento"
        )
        parametros["panel_w"] = st.selectbox(
            "Potencia por panel (W)", [250, 450, 550, 600], index=2, key="param_panel"
        )
        parametros["dod"] = st.slider(
            "Profundidad de descarga", 0.50, 1.00, DEFAULT_DOD, 0.01, key="param_dod"
        )
        parametros["eta_inv"] = st.slider(
            "Eficiencia del inversor", 0.70, 1.00, DEFAULT_ETA_INVERSOR, 0.01, key="param_eta_inv"
        )
        parametros["eta_carga"] = st.slider(
            "Eficiencia de carga", 0.70, 1.00, DEFAULT_ETA_CARGA, 0.01, key="param_eta_carga"
        )

    with st.sidebar.expander("5. Finanzas", expanded=False):
        parametros["tasa"] = st.slider(
            "Tasa de descuento", 0.0, 0.30, FIN_TASA, 0.01, key="param_tasa"
        )
        parametros["degradacion"] = st.slider(
            "Degradación anual", 0.0, 0.02, FIN_DEGRADACION, 0.0005, key="param_degradacion"
        )
        parametros["anios"] = st.slider(
            "Horizonte (años)", 5, 40, FIN_ANIOS, key="param_anios"
        )
        parametros["tarifa"] = st.number_input(
            "Tarifa (COP/kWh)", 100, 5000, FIN_TARIFA, 5, key="param_tarifa"
        )
        parametros["anio_bateria"] = st.slider(
            "Año de reposición de batería", 5, 25, DEFAULT_ANIO_REPOSICION_BATERIA,
            key="param_anio_bateria",
        )
        parametros["pct_bateria"] = st.slider(
            "Costo de batería (% inversión)", 0.0, 0.60, DEFAULT_PORCENTAJE_BATERIA,
            0.01, key="param_pct_bateria",
        )
        parametros["anio_inversor"] = st.slider(
            "Año de reposición de inversor", 5, 25, DEFAULT_ANIO_REPOSICION_INVERSOR,
            key="param_anio_inversor",
        )
        parametros["pct_inversor"] = st.slider(
            "Costo de inversor (% inversión)", 0.0, 0.50, DEFAULT_PORCENTAJE_INVERSOR,
            0.01, key="param_pct_inversor",
        )
    return parametros
