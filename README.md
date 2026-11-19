# ⚡ Análisis de Carga Eléctrica y Modelos Predictivos

Aplicación Streamlit que convierte el notebook `Copia_de_proyecto11.ipynb` en una
herramienta interactiva en español. Permite cargar datos, explorarlos, entrenar
los modelos originales, examinar series temporales y simular escenarios
energéticos y financieros sin necesidad de Python ni Jupyter.

## Funcionalidades

- Carga de CSV, Excel, JSON y Parquet, con validación de esquema y tipos.
- Filtros por rango de fechas, electrodoméstico, rangos numéricos y búsqueda de texto.
- Métricas con `st.metric`, tablas con búsqueda/orden/descarga CSV y Excel.
- Gráficos interactivos con Altair: distribución, dispersión, correlación, series,
  matriz de confusión, curva ROC y compositiones energéticas.
- Red densa de regresión con `Consumo_Teorico`, simulador individual de predicción.
- K-Means con perfiles por clúster y parámetro de número de clústeres.
- RNN original (v1) y RNN corregida (v2), comparación con baseline de persistencia.
- Clasificador binario de alto/bajo consumo con matriz de confusión, ROC y simulador.
- Perfil horario, dimensionamiento FV/batería y simulación de 7 días para dos escenarios.
- VPN, recuperación descontada, reposición de componentes y análisis de sensibilidad.
- Reinicio de filtros y parámetros, estados vacíos y mensajes de error claros.

## Requisitos

- Python 3.10 o superior.
- El dataset original o cualquier archivo con la estructura descrita en
  [`data/README.md`](data/README.md).
- Opcional: TensorFlow para entrenar las redes neuronales.

## Instalación y ejecución local

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Para habilitar las pestañas de modelos neuronales:

```bash
pip install -r requirements-ml.txt
streamlit run streamlit_app.py
```

TensorFlow se entrega aparte porque es una dependencia pesada: así el resto de
la aplicación (datos, EDA, clustering, energía y finanzas) funciona con un
despliegue ligero. Si no está instalado, las pestañas correspondientes muestran
una instrucción clara en lugar de fallar.

## Cargar los datos

1. Selecciona **Subir archivo** en la barra lateral y elige tu archivo; o
2. copia `inventario_carga_.xlsx` en `data/` y elige **Ruta local**; o
3. crea un `.env` a partir de `.env.example` y define `INVENTARIO_DATA_PATH`.

Columnas obligatorias: `Fecha`, `Mes`, `Electrodomestico`, `Potencia_W`,
`Horas_Uso_Dia`, `Consumo_kWh`, `H_activa`, `Consumo_Diario_Total`.

## Estructura del proyecto

```text
.
├── streamlit_app.py          # Punto de entrada y pestañas
├── requirements.txt          # Dependencias ligeras de producción
├── requirements-ml.txt       # TensorFlow opcional
├── .env.example              # Variable de ruta del dataset
├── README.md
├── vercel.json               # Configuración de Vercel
├── .vercelignore
├── api/index.py              # Entrada ASGI para Vercel
├── config/settings.py        # Constantes y parámetros del notebook
├── core/
│   ├── data_loader.py        # Carga y validación
│   ├── processing.py         # Transformaciones y simulaciones
│   ├── analysis.py           # Métricas, clustering y finanzas
│   └── models.py             # Redes neuronales con carga diferida
├── components/
│   ├── sidebar.py
│   ├── metrics.py
│   ├── charts.py
│   └── tables.py
├── assets/styles.css
├── data/README.md            # Esquema del dataset
└── .streamlit/config.toml    # Tema sobrio
```

## Trazabilidad y supuestos

- Cada bloque del notebook está identificado en la pestaña **Conclusiones**;
  las fórmulas, arquitecturas e hiperparámetros se conservan sin alteración.
- Las métricas se calculan en vivo con los datos y parámetros seleccionados.
  Los valores impresos por el notebook se muestran sólo como **referencia
  histórica** y nunca se presentan como resultados de la ejecución actual.
- El notebook reporta dos valores distintos de R² para la regresión
  (`0.115809` en la celda de entrenamiento y `0.0884` en el texto de
  conclusiones de Gradio). La aplicación muestra ambos con su procedencia.
- La RNN v1 usa fechas simuladas y escala toda la serie, tal como el notebook;
  la v2 usa la serie diaria real consolidada, split cronológico y escalado
  únicamente con entrenamiento.
- Los datos son sintéticos y los modelos son ejercicios experimentales: no
  demuestran ahorros energéticos ni viabilidad fotovoltaica.

## Despliegue en Streamlit Community Cloud (recomendado)

1. Sube el proyecto a un repositorio de GitHub (público o privado).
2. En [share.streamlit.io](https://share.streamlit.io) selecciona **New app**.
3. Indica el repositorio, la rama y el archivo principal: `streamlit_app.py`.
4. En **Advanced settings → Secrets** configura, si aplica:
   `INVENTARIO_DATA_PATH` con la ruta válida dentro del entorno de despliegue.
5. No subas el dataset si contiene datos privados; usa el uploader de la app o
   almacenamiento externo.
6. Pulsa **Deploy** y verifica la URL asignada.

Para redes neuronales en Community Cloud, añade `tensorflow-cpu` a las
dependencias del entorno (por ejemplo, usando `requirements-ml.txt` como archivo
de requisitos) y ten en cuenta el mayor tiempo de arranque y tamaño del paquete.

## Despliegue en Vercel

### Arquitectura elegida y limitación conocida

Se entregan `vercel.json` y `api/index.py` con un objeto ASGI `app = st.App(...)`
compatible con `@vercel/python`. **Streamlit necesita un proceso persistente y
una conexión WebSocket**, mientras que Vercel Functions son efímeras, se
apagan entre solicitudes y tienen límite de duración (60 s en la configuración
incluida). Por ello, Vercel **no es el entorno recomendado para la interfaz
completa**: la opción viable y documentada es el **Opción C**, desplegar la UI
en un servicio con proceso persistente (Streamlit Community Cloud, Railway,
Render, Fly.io o Google Cloud Run) y mantener estos archivos para integraciones
o pruebas de arranque en Vercel. El wrapper no modifica la lógica analítica.

Tamaño: `requirements.txt` excluye TensorFlow deliberadamente. Si necesitas los
modelos en Vercel, calcula el tamaño del bundle y considera Vercel Pro o un
servicio alternativo con más memoria. CORS no requiere configuración adicional
porque no se exponen endpoints propios aparte de la app.

### Desde la CLI

```bash
npm install -g vercel
vercel login
vercel link
vercel dev
vercel --prod
```

`vercel dev` valida la configuración en local; `vercel --prod` publica en
producción. Para ver logs y errores:

```bash
vercel logs <deployment-url> --follow
```

### Desde GitHub

1. Sube el proyecto a un repositorio público o privado de GitHub.
2. Entra a Vercel y selecciona **Add New Project**.
3. Importa el repositorio. Vercel detectará `vercel.json` y `@vercel/python`.
4. En **Settings > Environment Variables** configura `INVENTARIO_DATA_PATH` si
   usas una ruta configurada. No guardes credenciales en el código.
5. Pulsa **Deploy**.
6. Verifica que el dominio asignado por Vercel cargue correctamente la
   aplicación; si la sesión se corta o el WebSocket falla, aplica la alternativa
   de proceso persistente descrita arriba.

### Rollback y redeploy

En **Deployments**, abre un despliegue anterior y pulsa **Promote to
Production** para volver a él. Para reconstruir sin cambios de código, usa
**Redeploy** en el despliegue deseado o ejecuta `vercel --prod` de nuevo.

## Validación final

- `vercel.json` usa el builder estándar `@vercel/python` y un entrypoint ASGI.
- `api/index.py` expone `app` y resuelve el script de forma relativa al archivo.
- `requirements.txt` contiene únicamente dependencias usadas por la aplicación.
- No hay rutas absolutas ni secretos en el código; la ruta del dataset se lee
  desde entorno o se elige en la interfaz.
- Los archivos de datos grandes están excluidos en `.gitignore` y `.vercelignore`.
- Ejecución local verificable con `streamlit run streamlit_app.py`.
- Despliegue con `vercel --prod` disponible, con las limitaciones de WebSockets,
  sesión persistente, duración y tamaño de paquete explicadas arriba.
