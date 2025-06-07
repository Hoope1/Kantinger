import streamlit as st
from pathlib import Path
from ..inference import run_edge
from ..model_manager import list_models_status
from ..config import BASE_DIR

st.header("📂 Bilder verarbeiten")

# Eingabe-Quelle: einzelne Datei oder Ordner
src_mode = st.radio("Quelle auswählen", ["Einzelnes Bild", "Bild-Ordner"])
if src_mode == "Einzelnes Bild":
    uploaded = st.file_uploader("Bild wählen", type=["png","jpg","jpeg"])
    if uploaded:
        tmp = Path("tmp") / uploaded.name
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_bytes(uploaded.getbuffer())
        files = [tmp]
else:
    folder = st.text_input("Pfad zu Bild-Ordner", "samples")
    files = list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.jpg"))

# Ziel-Ordner
out_dir = Path(st.text_input("Ziel-Ordner", "results"))
size = st.number_input("Maximale Kantenlänge (px) – 0 = original", 0, 2048, 0)

# Modell-Checkboxen (grau = nicht vorhanden)
st.subheader("Modelle wählen")
to_use = []
for name, present in list_models_status():
    chk = st.checkbox(name, value=present, disabled=not present)
    if chk and present:
        to_use.append(name)

if st.button("🚀 Start") and files and to_use:
    st.write(f"Verarbeite {len(files)} Bilder mit {to_use} ...")
    run_edge(to_use, files, out_dir, int(size) or None)
    st.success("Fertig!")
