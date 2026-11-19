"""Punto de entrada de la aplicación Streamlit.

Traduce el notebook ``Copia_de_proyecto11.ipynb`` a una interfaz guiada. Cada
pestaña conserva los cálculos originales; las métricas se recalculan con los
datos y parámetros seleccionados.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from components import charts, metrics, tables
from components.sidebar import render_data_source, render_filters, render_parameters, render_reset_control
from config.settings import (
    APP_DESCRIPTION,
    APP_SUBTITLE,
    APP_TITLE,
    COL_CLUSTER,
    COL_ELECTRO,
    COL_HORAS,
    COL_TOTAL,
    ESCENARIO_1_CONSUMO,
    ESCENARIO_1_CONSUMO_DIARIO,
    ESCENARIO_1_CONSUMO_NOCHE,
    ESCENARIO_1_GENERACION_FACTORES,
    ESCENARIO_1_MADRUGADA_2H,
    ESCENARIO_2_CONSUMO,
    ESCENARIO_2_CONSUMO_DIARIO,
    ESCENARIO_2_GENERACION_FACTORES,
    FINANCIAL_SCENARIOS,
    FIN_FACTORES_INVERSION,
    FIN_TARIFAS_SENSIBILIDAD,
    FIN_ANIOS_BATERIA_SENSIBILIDAD,
    NUMERIC_COLUMNS,
    NOTEBOOK_CONCLUSIONS,
    NOTEBOOK_REFERENCE,
    REQUIRED_COLUMNS,
    SIMULATION_DAYS,
    STYLES_PATH,
)
from core import analysis, data_loader, models, processing

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def aplicar_estilos() -> None:
    """Carga el CSS local, sin depender de servicios externos."""
    try:
        estilos = STYLES_PATH.read_text(encoding="utf-8")
        st.html(f"<style>{estilos}</style>")
    except OSError:
        pass


aplicar_estilos()

st.title(f"⚡ {APP_TITLE}")
st.markdown(f"#### {APP_SUBTITLE}")
st.caption(APP_DESCRIPTION)
st.divider()

render_reset_control()
origen = render_data_source()

# ---------------------------------------------------------------------------
# Carga y validación
# ---------------------------------------------------------------------------
dataset = None
mensaje_error = None
with st.status("Cargando dataset…", expanded=False) as estado:
    try:
        if origen["upload"] is not None:
            dataset = data_loader.load_from_upload(origen["upload"])
            nombre_origen = origen["upload"].name
        elif origen["path"] is not None:
            dataset = data_loader.load_from_path(origen["path"])
            nombre_origen = origen["path"].name
        else:
            encontrado = data_loader.find_default_data()
            if encontrado is not None:
                dataset = data_loader.load_from_path(encontrado)
                nombre_origen = encontrado.name
        if dataset is not None:
            st.session_state["nombre_origen"] = nombre_origen
            estado.update(label=f"Dataset cargado: {nombre_origen}", state="complete")
        else:
            estado.update(label="Esperando un archivo de datos", state="complete")
    except Exception as exc:  # mensaje útil sin detener el resto de la app
        mensaje_error = str(exc)
        estado.update(label="No se pudo cargar el dataset", state="error")

if mensaje_error:
    st.error(f"**No se pudo cargar el archivo:** {mensaje_error}")
    st.stop()

if dataset is None:
    st.info(
        "### Aún no hay datos cargados\n"
        "Sube un archivo desde la barra lateral o coloca `inventario_carga_.xlsx` "
        "en la carpeta `data/`. También puedes definir `INVENTARIO_DATA_PATH` en "
        "un archivo `.env` local."
    )
    with st.expander("Estructura esperada del archivo", expanded=True):
        st.markdown(
            "Columnas obligatorias: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Columna": list(REQUIRED_COLUMNS),
                    "Tipo": ["fecha", "fecha", "texto", "numérico", "numérico", "numérico", "numérico", "numérico"],
                    "Descripción": [
                        "Fecha del registro",
                        "Mes del registro",
                        "Nombre del electrodoméstico",
                        "Potencia en vatios",
                        "Horas de uso al día",
                        "Consumo unitario en kWh",
                        "Horas activas",
                        "Consumo diario total (se repite por fecha)",
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )
    st.stop()

filtros = render_filters(dataset)
parametros = render_parameters()
df = processing.apply_filters(dataset, **filtros)

if df.empty:
    st.warning(
        "Los filtros actuales excluyeron todos los registros. Ajusta el rango de "
        "fechas, los electrodomésticos o los rangos numéricos en la barra lateral, "
        "o pulsa **Reiniciar filtros y parámetros**."
    )
    st.stop()

df_trabajo = processing.add_consumo_teorico(df)
serie_diaria = processing.build_daily_series(df)
stats_diarios = processing.daily_stats(serie_diaria)
continuidad = processing.daily_continuity(serie_diaria)
columnas_numericas = [c for c in NUMERIC_COLUMNS if c in df.columns]
matriz_correlacion = analysis.correlation_matrix(df, columnas_numericas)

st.success(
    f"Datos listos: **{len(df):,} registros** de `{st.session_state.get('nombre_origen', 'dataset')}`."
)

pestañas = st.tabs(
    [
        "📊 Resumen",
        "🔎 Datos y filtros",
        "📈 EDA",
        "🧩 Clustering",
        "🔮 Regresión",
        "⏱️ Series temporales",
        "🏷️ Clasificación",
        "⚡ Energía",
        "💰 Finanzas",
        "📝 Conclusiones",
    ]
)

# ---------------------------------------------------------------------------
# 1. Resumen
# ---------------------------------------------------------------------------
with pestañas[0]:
    st.subheader("Vista general")
    metrics.render_dataset_metrics(df, serie_diaria, stats_diarios)
    st.divider()
    col_izquierda, col_derecha = st.columns(2)
    with col_izquierda:
        st.markdown("**Primeros registros**")
        st.dataframe(df.head(10), width="stretch", hide_index=True)
    with col_derecha:
        resumen_numerico, numeric_cols, categoric_cols = analysis.descriptive_summary(df)
        st.markdown("**Estadísticas descriptivas**")
        st.dataframe(resumen_numerico.astype(str), width="stretch", height=340)
        st.caption(
            f"Numéricas: {', '.join(numeric_cols)} · Categóricas: {', '.join(categoric_cols)}"
        )
    st.divider()
    st.markdown("**Continuidad de la serie diaria** (celdas 7 y 8)")
    c1, c2, c3 = st.columns(3, border=True)
    c1.metric("Días esperados", continuidad["dias_esperados"])
    c2.metric("Días disponibles", continuidad["dias_disponibles"])
    c3.metric("Días faltantes", continuidad["dias_faltantes"])
    if continuidad["continua"]:
        st.success("La serie tiene continuidad diaria completa.")
    else:
        st.warning("Existen fechas faltantes en el rango analizado.")
    st.caption(
        "Los registros de consumo diario son sintéticos, según se indica en las "
        "conclusiones del notebook."
    )

# ---------------------------------------------------------------------------
# 2. Datos y filtros
# ---------------------------------------------------------------------------
with pestañas[1]:
    st.subheader("Explorador de datos")
    with st.container(border=True):
        st.markdown("**Filtros activos**")
        st.write(
            {
                "Fechas": [str(filtros["fecha_range"][0]), str(filtros["fecha_range"][1])]
                if filtros["fecha_range"] else " Todas",
                "Electrodomésticos": filtros["appliances"] or "Todos",
                "Búsqueda": filtros["query"] or "—",
            }
        )
    tabla_datos = tables.render_table(
        df, "Registros filtrados", "registros", columnas_visibles=list(df.columns), altura=460
    )
    with st.expander("Perfil de columnas", expanded=False):
        tables.render_table(
            data_loader.column_overview(df), "Tipos y valores nulos", "perfil_columnas",
            busqueda=False, descargar=True,
        )
    with st.expander("Inventario de respaldo por electrodoméstico (celda 17)", expanded=False):
        inventario = processing.appliance_summary(df)
        tables.render_table(inventario, "Promedios por electrodoméstico", "inventario", busqueda=False)

# ---------------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------------
with pestañas[2]:
    st.subheader("Análisis exploratorio")
    charts.histograma(df, COL_TOTAL, "Distribución de Consumo Diario Total")
    charts.dispersion(
        df,
        COL_HORAS,
        COL_TOTAL,
        "Horas de uso vs Consumo Diario Total",
        color=COL_ELECTRO,
        columnas_tooltip=[COL_ELECTRO, COL_HORAS, COL_TOTAL],
    )
    charts.matriz_correlacion(matriz_correlacion)
    with st.expander("Horas de uso frente a horas activas (celda 15)", expanded=False):
        estadisticas, correlacion, por_electro = processing.hours_activity_stats(df)
        c1, c2 = st.columns(2)
        with c1:
            tables.render_table(estadisticas, "Estadísticas", "horas_stats", busqueda=False, descargar=False)
        with c2:
            tables.render_table(correlacion, "Correlación", "horas_corr", busqueda=False, descargar=False)
        tables.render_table(por_electro, "Promedios por electrodoméstico", "horas_electro", descargar=True)

# ---------------------------------------------------------------------------
# 4. Clustering
# ---------------------------------------------------------------------------
with pestañas[3]:
    st.subheader("Segmentación con K-Means")
    st.caption(
        "Celda 3: K-Means sobre las variables numéricas escaladas "
        f"({', '.join(columnas_numericas)}), con {parametros['n_clusters']} clústeres."
    )
    df_cluster, _, _ = analysis.fit_kmeans(
        df, columnas_numericas, n_clusters=parametros["n_clusters"]
    )
    perfiles, cantidades = processing.cluster_profile(df_cluster, columnas_numericas)
    charts.dispersion(
        df_cluster,
        COL_HORAS,
        COL_TOTAL,
        "Clústeres de equipos",
        color="Cluster",
        columnas_tooltip=[COL_CLUSTER, COL_ELECTRO, COL_HORAS, COL_TOTAL],
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        tables.render_table(perfiles, "Promedios por clúster (celdas 13 y 14)", "perfiles_cluster", busqueda=False)
    with c2:
        tables.render_table(
            cantidades.rename("Registros").reset_index(), "Registros por clúster",
            "cantidad_cluster", busqueda=False, descargar=False,
        )
    tables.render_table(df_cluster, "Datos etiquetados con clúster", "datos_cluster", altura=360)

# ---------------------------------------------------------------------------
# 5. Regresión
# ---------------------------------------------------------------------------
with pestañas[4]:
    st.subheader("Red neuronal de regresión")
    if not models.tf_available():
        st.warning(models.tensorflow_message())
    else:
        st.caption(
            "Celda 2: variable derivada `Consumo_Teorico = (Potencia_W × Horas_Uso_Dia) / 1000`, "
            "ColumnTransformer (StandardScaler + OneHotEncoder) y red densa 64 → 32 → 1."
        )
        with st.spinner("Entrenando la red de regresión…"):
            resultado_reg = models.train_regression_model(
                df,
                epochs=parametros["reg_epochs"],
                batch_size=parametros["reg_batch"],
                learning_rate=parametros["reg_lr"],
            )
        metrics.render_regression_metrics(resultado_reg.metrics)
        c1, c2 = st.columns(2)
        with c1:
            charts.histograma(
                pd.DataFrame({"Error": resultado_reg.errores}),
                "Error",
                "Histograma de errores (real − predicho)",
                color="#7c3aed",
            )
        with c2:
            datos_reg = pd.DataFrame(
                {
                    "Real": resultado_reg.y_true,
                    "Predicho": resultado_reg.y_pred,
                }
            )
            charts.dispersion(datos_reg, "Real", "Predicho", "Real vs predicho", columnas_tooltip=["Real", "Predicho"])

        st.divider()
        st.markdown("### Simulador de predicción individual (celda 6)")
        with st.form("simulador_regresion"):
            c1, c2 = st.columns(2)
            horas = c1.number_input("Horas de uso al día", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
            potencia = c2.number_input("Potencia (W)", min_value=0.0, value=150.0, step=10.0)
            kwh = c1.number_input("Consumo unitario (kWh)", min_value=0.0, value=0.75, step=0.05)
            opciones = sorted(df[COL_ELECTRO].dropna().unique().tolist())
            electro = c2.selectbox("Electrodoméstico", options=opciones, index=0)
            calcular = st.form_submit_button("Calcular consumo predicho")
        if calcular:
            try:
                prediccion = models.predict_consumption(
                    resultado_reg, horas, potencia, kwh, electro
                )
                st.success(f"**Consumo diario total estimado: {prediccion:.4f} kWh**")
            except Exception as exc:
                st.error(f"No se pudo realizar la predicción: {exc}")
        st.caption(
            "La predicción utiliza el mismo preprocesador ajustado con los datos "
            "de entrenamiento; las categorías no vistas se ignoran "
            "(`OneHotEncoder(handle_unknown='ignore')`)."
        )

# ---------------------------------------------------------------------------
# 6. Series temporales
# ---------------------------------------------------------------------------
with pestañas[5]:
    st.subheader("Series temporales y redes recurrentes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fecha inicial", str(continuidad["fecha_inicial"].date()))
    c2.metric("Fecha final", str(continuidad["fecha_final"].date()))
    c3.metric("Fechas duplicadas", "Sí" if continuidad["duplicados"] else "No")

    if len(serie_diaria) < parametros["rnn_seq"] + 3:
        st.warning("La serie es demasiado corta para construir secuencias con la longitud elegida.")
    elif not models.tf_available():
        st.warning(models.tensorflow_message())
    elif parametros["rnn_version"] == "v2":
        st.caption(
            "Celda 9: división cronológica 80/20, escalado sólo con entrenamiento, "
            "secuencias de 5 días y `SimpleRNN(32)` con semilla 42."
        )
        with st.spinner("Entrenando la RNN corregida…"):
            resultado_rnn = models.train_corrected_rnn(
                serie_diaria.to_numpy(),
                serie_diaria.index.to_numpy(),
                sequence_length=parametros["rnn_seq"],
                epochs=parametros["rnn_epochs"],
                batch_size=parametros["rnn_batch"],
            )
        metrics.render_rnn_metrics(resultado_rnn.metrics, "RNN corregida")
        tabla_rnn = models.build_rnn_results_table(resultado_rnn)
        lineas = tabla_rnn.rename(
            columns={"Fecha": "fecha", "Consumo_Real_kWh": "real", "Consumo_Predicho_kWh": "predicho"}
        )
        charts.lineas_temporales(
            lineas, "fecha", {"Real": "real", "Predicción RNN": "predicho"},
            "Consumo real y predicho — RNN corregida",
        )
        st.divider()
        st.markdown("### Comparación con el modelo de persistencia (celda 10)")
        metricas_base, comparacion = analysis.persistence_baseline(
            serie_diaria, resultado_rnn.split_idx, resultado_rnn.metrics
        )
        c1, c2, c3 = st.columns(3, border=True)
        c1.metric("Persistencia MAE", f"{metricas_base['MAE_kWh']:.6f} kWh")
        c2.metric("Persistencia RMSE", f"{metricas_base['RMSE_kWh']:.6f} kWh")
        c3.metric("Persistencia R²", f"{metricas_base['R2']:.6f}")
        tables.render_table(comparacion, "RNN corregida vs persistencia", "comparacion_rnn", busqueda=False)
        with st.expander("Predicciones del conjunto de prueba", expanded=False):
            tables.render_table(tabla_rnn, "Real vs predicho", "resultados_rnn", altura=360)
    else:
        st.caption(
            "Celda 4: versión original del notebook, con fechas simuladas desde "
            "2026-01-01 y escalado de toda la serie (no sólo entrenamiento)."
        )
        with st.spinner("Entrenando la RNN original…"):
            resultado_legacy = models.train_legacy_rnn(
                df,
                sequence_length=parametros["rnn_seq"],
                epochs=parametros["rnn_epochs"],
                batch_size=parametros["rnn_batch"],
            )
        metrics.render_rnn_metrics(resultado_legacy.metrics)
        datos_grafico = pd.DataFrame(
            {
                "fecha": pd.to_datetime(resultado_legacy.fechas),
                "real": resultado_legacy.y_real,
                "predicho": resultado_legacy.y_pred,
            }
        )
        charts.lineas_temporales(
            datos_grafico, "fecha", {"Real": "real", "Predicción RNN": "predicho"},
            "Predicción temporal — RNN v1",
        )
        st.info(
            "La comparación con persistencia pertenece a la versión corregida "
            "(v2). Selecciona v2 en la barra lateral para verla."
        )

# ---------------------------------------------------------------------------
# 7. Clasificación
# ---------------------------------------------------------------------------
with pestañas[6]:
    st.subheader("Clasificación de consumo alto / bajo")
    if not models.tf_available():
        st.warning(models.tensorflow_message())
    else:
        st.caption(
            "Celdas 5 y 12: objetivo binario según la mediana de "
            "`Consumo_Diario_Total`, usando `Horas_Uso_Dia` como variable explicativa."
        )
        with st.spinner("Entrenando el clasificador…"):
            resultado_clf = models.train_classifier_model(
                df, epochs=parametros["clf_epochs"], batch_size=parametros["clf_batch"]
            )
        res_clf = analysis.classification_results(
            resultado_clf.y_test, resultado_clf.y_prob
        )
        metrics.render_classification_metrics(res_clf)
        c1, c2 = st.columns(2)
        with c1:
            charts.matriz_confusion(res_clf["matriz"])
        with c2:
            charts.curva_roc(res_clf["fpr"], res_clf["tpr"], res_clf["auc"])
        with st.expander("Reporte de clasificación", expanded=False):
            reporte = pd.DataFrame(res_clf["reporte"]).T
            tables.render_table(reporte, "Métricas por clase", "reporte_clf", busqueda=False)
        st.divider()
        st.markdown("### Estimar clase para nuevas horas de uso")
        horas_nuevas = st.slider(
            "Horas de uso al día", 0.0, 24.0, 5.0, 0.5, key="clf_horas_nuevas"
        )
        probabilidad = models.predict_class_probability(resultado_clf, horas_nuevas)
        clase = "Alto consumo" if probabilidad >= 0.5 else "Bajo consumo"
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad de alto consumo", f"{probabilidad:.4f}")
            st.progress(min(1.0, max(0.0, probabilidad)))
        with c2:
            st.info(f"Clase asignada con umbral 0.5: **{clase}**.")
            st.caption(
                f"El umbral de la variable objetivo se calculó con la mediana del "
                f"dataset: {resultado_clf.mediana:.3f} kWh."
            )

# ---------------------------------------------------------------------------
# 8. Energía
# ---------------------------------------------------------------------------
with pestañas[7]:
    st.subheader("Perfil horario y balance energético")
    perfil, resumen_horario = processing.build_hourly_profile()
    with st.expander("Perfil horario del notebook (celda 18)", expanded=False):
        tables.render_table(perfil, "Distribución por electrodoméstico", "perfil_horario", busqueda=False)
    resumen_df = resumen_horario.rename("Consumo_kWh").reset_index()
    resumen_df.columns = ["Franja", "Consumo_kWh"]
    charts.barras(resumen_df, "Franja", "Consumo_kWh", "Consumo estimado por franja")

    escenario = st.segmented_control(
        "Escenario",
        options=["Cargas esenciales", "Vivienda completa"],
        default="Cargas esenciales",
        key="energia_escenario",
    )
    if escenario == "Cargas esenciales":
        consumo = ESCENARIO_1_CONSUMO
        consumo_diario = ESCENARIO_1_CONSUMO_DIARIO
        factores = ESCENARIO_1_GENERACION_FACTORES
        bateria = processing.size_battery(
            ESCENARIO_1_CONSUMO_NOCHE,
            ESCENARIO_1_MADRUGADA_2H,
            profundidad_descarga=parametros["dod"],
            eficiencia_inversor=parametros["eta_inv"],
        )
    else:
        consumo = ESCENARIO_2_CONSUMO
        consumo_diario = ESCENARIO_2_CONSUMO_DIARIO
        factores = ESCENARIO_2_GENERACION_FACTORES
        bateria = processing.size_battery(
            consumo["Noche"],
            consumo["Madrugada"],
            profundidad_descarga=parametros["dod"],
            eficiencia_inversor=parametros["eta_inv"],
        )

    dimensionamiento = processing.size_pv(
        consumo_diario,
        parametros["radiacion"],
        parametros["rendimiento"],
        parametros["panel_w"],
    )
    generacion = processing.build_generation(dimensionamiento["generacion_diaria"], factores)
    metrics.render_energy_metrics(dimensionamiento, bateria)

    st.divider()
    st.markdown("### Balance híbrido del primer día con batería inicialmente descargada")
    balance_vacio, soc_final, totales = processing.simulate_hybrid_balance(
        consumo,
        generacion,
        bateria["capacidad_bateria"],
        parametros["dod"],
        parametros["eta_inv"],
        parametros["eta_carga"],
        estado_inicial=0.0,
    )
    tables.render_table(balance_vacio, "Despacho por franja (celda 22)", "balance_vacio", busqueda=False)
    charts.areas_apiladas(
        balance_vacio,
        "Franja",
        ["Solar_directa_kWh", "Descarga_bateria_AC_kWh", "Red_kWh"],
        "Cobertura de la demanda por fuente",
    )
    st.caption(
        f"Consumo total: {totales['consumo_total']:.3f} kWh · "
        f"Generación solar: {totales['generacion_solar']:.3f} kWh · "
        f"Energía de la red: {totales['energia_red']:.3f} kWh · "
        f"Estado final de batería: {soc_final:.3f} kWh"
    )

    if escenario == "Cargas esenciales":
        with st.expander("Primer día con batería inicialmente cargada (celda 21)", expanded=False):
            balance_lleno, _, totales_lleno = processing.simulate_hybrid_balance(
                consumo,
                generacion,
                bateria["capacidad_bateria"],
                parametros["dod"],
                parametros["eta_inv"],
                parametros["eta_carga"],
            )
            tables.render_table(balance_lleno, "Despacho con batería llena", "balance_lleno", busqueda=False)
            st.caption(
                f"Energía de la red: {totales_lleno['energia_red']:.3f} kWh"
            )

    st.divider()
    st.markdown(f"### Simulación de {SIMULATION_DAYS} días")
    tabla_dias = processing.simulate_multiday(
        consumo,
        generacion,
        bateria["capacidad_bateria"],
        dias=SIMULATION_DAYS,
        profundidad_descarga=parametros["dod"],
        eficiencia_inversor=parametros["eta_inv"],
        eficiencia_carga=parametros["eta_carga"],
    )
    tables.render_table(tabla_dias, "Resultados diarios", "simulacion_dias", busqueda=False)
    charts.lineas_flujo(
        tabla_dias, "Dia", "SOC_final_kWh", "Dia", "Estado de carga final de la batería"
    )
    st.caption(
        f"Compra de red en {SIMULATION_DAYS} días: {tabla_dias['Red_kWh'].sum():.3f} kWh · "
        f"Promedio diario: {tabla_dias['Red_kWh'].mean():.3f} kWh/día"
    )

# ---------------------------------------------------------------------------
# 9. Finanzas
# ---------------------------------------------------------------------------
with pestañas[8]:
    st.subheader("Evaluación financiera a 25 años")
    escenarios = {
        nombre: {**valores, "tarifa": parametros["tarifa"]}
        for nombre, valores in FINANCIAL_SCENARIOS.items()
    }
    simple = analysis.compute_simple_financials(
        escenarios, parametros["tasa"], parametros["degradacion"], parametros["anios"]
    )
    con_reposicion, flujos = analysis.compute_financials_with_replacements(
        escenarios,
        parametros["tasa"],
        parametros["degradacion"],
        parametros["anios"],
        parametros["anio_bateria"],
        parametros["pct_bateria"],
        parametros["anio_inversor"],
        parametros["pct_inversor"],
    )
    sensibilidad = analysis.compute_sensitivity(
        escenarios,
        parametros["tasa"],
        parametros["degradacion"],
        parametros["anios"],
        FIN_TARIFAS_SENSIBILIDAD,
        FIN_FACTORES_INVERSION,
        FIN_ANIOS_BATERIA_SENSIBILIDAD,
        parametros["pct_bateria"],
        parametros["anio_inversor"],
        parametros["pct_inversor"],
    )
    resumen_sens, porcentaje_sens = analysis.summarize_sensitivity(sensibilidad)

    st.caption(
        "Todos los supuestos de inversión, ahorro y mantenimiento proceden de las "
        "celdas 25 a 30 del notebook; la tarifa y el resto de parámetros son ajustables."
    )
    metrics.render_financial_metrics(con_reposicion)
    st.divider()
    with st.expander("Resultados simples, sin reposiciones (celdas 25 y 26)", expanded=False):
        tables.render_table(simple, "VPN y recuperación", "fin_simple", busqueda=False)
    with st.expander("Resultados con reposición de componentes (celdas 27 y 30)", expanded=True):
        tables.render_table(con_reposicion, "Resumen financiero", "fin_reposicion", busqueda=False)
        tables.render_table(flujos, "Flujos detallados", "fin_flujos", altura=360)
        flujos_grafico = flujos.copy()
        flujos_grafico["Flujo_acumulado_COP"] = flujos_grafico.groupby("Escenario")[
            "Flujo_neto_COP"
        ].cumsum()
        charts.lineas_flujo(
            flujos_grafico, "Año", "Flujo_acumulado_COP", "Escenario",
            "Flujo acumulado por escenario (sin descontar)",
        )
    with st.expander("Análisis de sensibilidad (celdas 28 y 29)", expanded=False):
        c1, c2 = st.columns([3, 1])
        with c1:
            tables.render_table(resumen_sens, "Resumen por escenario", "fin_resumen_sens", busqueda=False)
        with c2:
            porcentaje_df = porcentaje_sens.rename("VPN positivo (%)").reset_index()
            tables.render_table(porcentaje_df, "Escenarios favorables", "fin_porcentaje", busqueda=False, descargar=False)
        tables.render_table(sensibilidad, "Combinaciones evaluadas", "fin_sensibilidad", altura=360)

# ---------------------------------------------------------------------------
# 10. Conclusiones y trazabilidad
# ---------------------------------------------------------------------------
with pestañas[9]:
    st.subheader("Conclusiones del notebook")
    st.markdown(NOTEBOOK_CONCLUSIONS)
    st.divider()
    st.markdown("### Trazabilidad notebook → aplicación")
    trazabilidad = pd.DataFrame(
        [
            ("Celdas 0-1", "Carga y EDA", "Resumen · Datos y filtros · EDA"),
            ("Celda 2", "Regresión con Consumo_Teorico", "Regresión"),
            ("Celdas 3, 13-14", "K-Means y perfiles", "Clustering"),
            ("Celda 4", "RNN v1 con fechas simuladas", "Series temporales (v1)"),
            ("Celdas 5, 12", "Clasificación, matriz de confusión y ROC", "Clasificación"),
            ("Celda 6", "Interfaz Gradio y simulador", "Simulador de regresión · Conclusiones"),
            ("Celdas 7-8", "Serie diaria consolidada y continuidad", "Resumen · Series temporales"),
            ("Celdas 9-10", "RNN corregida y persistencia", "Series temporales (v2)"),
            ("Celdas 11, 16-17", "Comparación e indicadores de consumo", "Series temporales · Resumen · Datos"),
            ("Celdas 18-24", "Perfil horario, dimensionamiento y simulación", "Energía"),
            ("Celdas 25-30", "VPN, reposiciones y sensibilidad", "Finanzas"),
        ],
        columns=["Celda", "Contenido original", "Sección en la aplicación"],
    )
    tables.render_table(trazabilidad, "Mapa de correspondencia", "trazabilidad", busqueda=False)

    st.markdown("### Valores reportados en el notebook")
    referencias = pd.DataFrame(
        [
            {"Bloque": bloque, "Métrica": clave, "Valor": valor}
            for bloque, elementos in NOTEBOOK_REFERENCE.items()
            for clave, valor in elementos.items()
        ]
    )
    tables.render_table(referencias, "Referencia histórica", "referencias", busqueda=False)

    st.warning(
        "El notebook presenta dos cifras distintas para el R² de la regresión: "
        "la celda de entrenamiento imprimió **0.115809**, mientras el texto de "
        "conclusiones de Gradio menciona **0.0884**. Esta aplicación muestra ambas "
        "tal como aparecen y calcula sus propias métricas en vivo; no las fusiona."
    )
    st.caption(
        "Supuestos documentados: el dataset original no se distribuye con el código; "
        "se carga desde archivo local, ruta configurada o carga del usuario. "
        "TensorFlow es opcional y se instala por separado por su tamaño. "
        "En Vercel, Streamlit requiere un proceso persistente: consulta el README."
    )
