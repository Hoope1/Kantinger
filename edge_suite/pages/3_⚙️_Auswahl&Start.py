"""Streamlit-Seite 3 – Batch-Job mit mehreren Bildern und Modellen."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import streamlit as st

from edge_suite.inference import run_edge
from edge_suite.model_manager import list_models_status

st.header("⚙️ Batch-Job starten")

# Eingabe-Ordner
src_dir = Path(st.text_input("Bild-Ordner", "samples"))
valid_ext = (".png", ".jpg", ".jpeg")
img_files: Sequence[Path] = (
    [p for p in src_dir.glob("**/*") if p.suffix.lower() in valid_ext]
    if src_dir.exists()
    else []
)
st.write(f"• {len(img_files)} Bilder gefunden")

# Modell-Auswahl
models_available = [name for name, ok in list_models_status() if ok]
selected = st.multiselect(
    "Modelle wählen", options=models_available, default=models_available[:3]
)

# Ziel-Ordner
out_root = Path(st.text_input("Ziel-Ordner", "results_batch"))
wipe = st.checkbox("Ziel-Ordner vor Start löschen")

# Parameter
resize = st.number_input("Max. Kantenlänge (px) – 0 = Original", 0, 4096, 0)

if st.button("🚀 Los") and img_files and selected:
    if wipe and out_root.exists():
        shutil.rmtree(out_root)

    run_edge(selected, list(img_files), out_root, size=resize or None)
    st.success("Batch abgeschlossen")
