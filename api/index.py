"""Entrada ASGI para Vercel.

Expone la aplicación Streamlit como objeto ``app``, que es lo que espera el
builder ``@vercel/python``. Streamlit necesita un proceso persistente y
WebSockets; Vercel Functions son efímeras y tienen límite de duración. Por eso
este archivo se entrega como puente técnico y el despliegue recomendado para la
interfaz completa es Streamlit Community Cloud, Railway, Render, Fly.io o Cloud
Run. Consulta la sección "Despliegue en Vercel" del README.
"""

from pathlib import Path

import streamlit as st

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"

app = st.App(str(SCRIPT_PATH))
