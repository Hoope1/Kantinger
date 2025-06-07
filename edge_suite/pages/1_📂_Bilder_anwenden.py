import streamlit as st
from pathlib import Path

from edge_suite.inference import run_edge
from edge_suite.model_manager import list_models_status

st.header("📂 Bilder verarbeiten")

SRC_TYPES = ("png", "jpg", "jpeg")

src_mode = st.radio("Quelle wählen", ["Einzelbild", "Ordner"])
if src_mode == "Einzelbild":
    uploaded = st.file_uploader("Bild auswählen", type=SRC_TYPES)
    files: list[Path] = []
    if uploaded:
        tmp = Path("tmp") / uploaded.name
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_bytes(uploaded.getbuffer())
        files.append(tmp)
else:
    folder = Path(st.text_input("Ordnerpfad", "samples"))
    files = [*folder.rglob("*") if folder.exists() else []]
    files = [f for f in files if f.suffix[1:].lower() in SRC_TYPES]

out_dir = Path(st.text_input("Ziel-Ordner", "results"))
size = st.number_input("Max. Kantenlänge (px) – 0 = Original",
                       0, 4096, 0)

st.subheader("Modelle")
chosen: list[str] = []
for name, ok in list_models_status():
    chk = st.checkbox(name, ok and name in ("TEED", "EdgeNAT"), disabled=not ok)
    if chk and ok:
        chosen.append(name)

if st.button("🚀 Start") and files and chosen:
    st.info(f"{len(files)} Bilder, Modelle: {', '.join(chosen)}")
    run_edge(chosen, files, out_dir, size or None)
    st.success("Fertig!")
