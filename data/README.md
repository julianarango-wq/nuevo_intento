# Datos

Este directorio no incluye el dataset original: el notebook utiliza registros
sintéticos y el archivo se mantiene fuera del repositorio para no versionar
datos privados ni archivos grandes.

## Cómo cargar el dataset

1. **Desde la interfaz:** en la barra lateral, selecciona *Subir archivo* y elige
   un CSV, Excel (`.xlsx`/`.xls`), JSON o Parquet.
2. **Desde una ruta local:** copia `inventario_carga_.xlsx` en esta carpeta y
   selecciona *Ruta local* en la barra lateral.
3. **Desde una variable de entorno:** copia `.env.example` a `.env` y define
   `INVENTARIO_DATA_PATH=./data/inventario_carga_.xlsx`.

## Estructura esperada

Columnas obligatorias (misma estructura del notebook original):

| Columna | Tipo | Descripción |
| --- | --- | --- |
| `Fecha` | fecha | Fecha del registro |
| `Mes` | fecha | Mes del registro |
| `Electrodomestico` | texto | Categoría del equipo |
| `Potencia_W` | numérico | Potencia en vatios |
| `Horas_Uso_Dia` | numérico | Horas de uso diario |
| `Consumo_kWh` | numérico | Consumo unitario en kWh |
| `H_activa` | numérico | Horas activas |
| `Consumo_Diario_Total` | numérico | Total diario (se repite en cada equipo de la misma fecha) |

El dataset original tiene 9.600 filas y 8 columnas. La aplicación valida la
presencia de todas las columnas y convierte fechas y numéricos automáticamente.
No elimina filas: los filtros se aplican en la interfaz y se pueden reiniciar.
