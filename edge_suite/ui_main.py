"""Streamlit-Einstiegs-Script."""

import streamlit as st

st.set_page_config(
    page_title="Edge Suite",
    page_icon="🌊",
    layout="wide",
)

st.sidebar.title("🖼️ Edge-Detection Suite")
st.sidebar.markdown(
    """
**1 Bilder anwenden** – einzelnes Bild / Ordner verarbeiten  
**2 Modelle verwalten** – Gewichte downloaden / hochladen  
**3 Auswahl & Start** – Modelle wählen und Batch starten
"""
)

# Standard-Seite öffnen
# Der Name entspricht exakt dem Dateinamen der Page (ohne Pfad).
st.switch_page("1_📂_Bilder_anwenden.py")
