import streamlit as st
from ..model_manager import ensure_weight, list_models_status
from ..config import MODELS, WEIGHTS_DIR

st.header("⬇️ Modelle & Gewichte verwalten")

for name, available in list_models_status():
    mi = MODELS[name]
    cols = st.columns([2,1,1,1])
    cols[0].markdown(f"**{name}**  \n{mi.kind.capitalize()} – {mi.params_m:.2f} M Param – ODS {mi.ods:.3f}")
    if available:
        cols[1].success("✓")
    else:
        cols[1].error("✗")
    if cols[2].button("Download", key=f"dl_{name}") and not available:
        ensure_weight(mi, force=False)
        st.rerun()
    up = cols[3].file_uploader("Upload", type=["pth","pt"], key=f"up_{name}")
    if up:
        dest = WEIGHTS_DIR / mi.default_weight
        dest.write_bytes(up.getbuffer())
        st.rerun()
