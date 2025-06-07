import streamlit as st
from pathlib import Path
from .config import BASE_DIR

# Streamlit findet alle Dateien in edge_suite/pages automatisch
st.set_page_config(page_title="Edge Suite", page_icon="🌊", layout="wide")

st.sidebar.title("🖼️ Edge-Detection Suite")
st.sidebar.markdown(
    """
    **1 Bilder anwenden** – einzelne Bilder oder Ordner verarbeiten  
    **2 Modelle verwalten** – Weights downloaden / hochladen  
    **3 Auswahl & Start** – Modelle wählen und Batch starten
    """
)
st.sidebar.divider()
st.sidebar.caption("Backend-Pfad: `%s`" % BASE_DIR)
st.switch_page("edge_suite/pages/1_📂_Bilder_anwenden.py")  # Standard-Seite
