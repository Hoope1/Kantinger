"""Streamlit-Seite 2 – Gewichte downloaden / hochladen."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from edge_suite.model_manager import ensure_weight, list_models_status
from edge_suite.config import MODELS, WEIGHTS_DIR

st.header("⬇️ Modelle und Gewichte verwalten")

for name, available in list_models_status():
    mi = MODELS[name]

    col_meta, col_status, col_dl, col_up = st.columns([3, 1, 1, 1])
    col_meta.markdown(
        f"**{name}**   · {mi.kind.capitalize()}   "
        f"{mi.params_m:.2f} M Param   "
        f"ODS {mi.ods:.3f}"
    )

    col_status.success("✓" if available else "✗")

    if col_dl.button("Download", key=f"dl_{name}", disabled=available):
        ensure_weight(mi)
        st.experimental_rerun()

    upload = col_up.file_uploader(
        "Upload", type=("pth", "pt"), key=f"up_{name}", label_visibility="collapsed"
    )
    if upload is not None:
        target = WEIGHTS_DIR / mi.default_weight
        target.write_bytes(upload.getbuffer())
        st.success(f"{upload.name} gespeichert als {target.name}")
        st.experimental_rerun()

st.caption(
    "Downloads kommen direkt aus den Original-Repos "
    "(HTTP, Google-Drive oder Zenodo). "
    "Eigene Fine-Tuned-Gewichte können per Upload ersetzt werden."
)
