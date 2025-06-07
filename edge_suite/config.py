"""
Globale Metadaten zu allen 10 Modellen.
Füge hier weitere hinzu oder ändere Links nach Belieben.
"""
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

@dataclass
class ModelInfo:
    name: str
    kind: str             # "light" oder "heavy"
    params_m: float
    ods: float
    weight_url: str       # Direkt- oder Google-Drive-Link
    default_weight: str   # Dateiname unter WEIGHTS_DIR
    extra_pkgs: list[str] # Pakete, die _zusätzlich_ gebraucht werden

MODELS: dict[str, ModelInfo] = {
    # ------------------- Leichtgewichte ---------------------
    "TEED":     ModelInfo("TEED","light",0.058,0.806,
        "https://github.com/TEED/weights/teed_bsds.pth","teed.pth",[]),
    "PiDiNet":  ModelInfo("PiDiNet","light",0.78,0.798,
        "https://github.com/xavysp/weighted_models/pidinet_small.pth","pidinet.pth",[]),
    "FINED":    ModelInfo("FINED","light",0.68,0.792,
        "https://github.com/FinedNet/bsds_fined.pth","fined.pth",[]),
    "DexiNed":  ModelInfo("DexiNed","light",2.6,0.788,
        "https://github.com/xavysp/DexiNed/releases/download/v2.0/dexined.pth","dexined.pth",[]),
    "CATS":     ModelInfo("CATS","light",3.1,0.800,
        "https://huggingface.co/cats-lite/resolve/main/cats_lite.pth","cats.pth",[]),
    # ------------------- Schwergewichte ---------------------
    "EdgeNAT":  ModelInfo("EdgeNAT","heavy",45,0.849,
        "https://zenodo.org/record/123456/files/edgenat_l.pth?download=1","edgenat.pth",["einops"]),
    "DiffEdge": ModelInfo("DiffEdge","heavy",112,0.834,
        "https://huggingface.co/diffedge/resolve/main/diffedge_swin.pth","diffedge.pth",
        ["einops","transformers","xformers"]),
    "UAED":     ModelInfo("UAED","heavy",68,0.829,
        "https://drive.google.com/uc?id=1a2b3UAED","uaed.pth",["einops"]),
    "BDCN":     ModelInfo("BDCN","heavy",63,0.828,
        "https://github.com/pkuCactus/BDCN/releases/download/v1.0.0/bdcn_sem.pth","bdcn.pth",[]),
    "EDTER":    ModelInfo("EDTER","heavy",96,0.824,
        "https://mmseg.org/checkpoints/edter_bsds.pth","edter.pth",["einops"]),
}
