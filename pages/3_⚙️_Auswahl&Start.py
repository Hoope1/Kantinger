import streamlit as st, shutil
from pathlib import Path
from ..model_manager import list_models_status
from ..inference import run_edge

st.header("⚙️ Batch-Job starten")

root = Path(st.text_input("Bild-Ordner", "samples"))
files = sorted(list(root.glob("*.png")) + list(root.glob("*.jpg")))
st.write(f"{len(files)} Bilder gefunden.")

models = [n for n,p in list_models_status() if p]
sel = st.multiselect("Modelle", models, default=models[:3])
out_root = Path(st.text_input("Ziel-Dir", "results_batch"))
clear = st.checkbox("Ziel-Dir vorab löschen")

if st.button("🚀 Los"):
    if clear and out_root.exists():
        shutil.rmtree(out_root)
    run_edge(sel, files, out_root)
    st.success("Batch fertig.")
