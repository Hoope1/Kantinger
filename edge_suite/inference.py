"""
edge_suite.inference
────────────────────
• Lädt **10 Original-Netze** aus ihren Git-Repos
• Bindet passende Gewichte aus `edge_suite/weights`
• Vereinheitlicht Vor- und Nachverarbeitung
"""

from __future__ import annotations
from pathlib import Path
from typing   import Iterable
import cv2
import torch
import numpy as np
import importlib

from .config         import WEIGHTS_DIR, MODELS          # zentrale Registry
from .model_manager  import ensure_weight                # DL + Pkg-Installer

# ───────────────────────────── Helper ──────────────────────────────

def _safe_load_state(net: torch.nn.Module, ckpt: Path) -> None:
    """Lädt .pth/.pt (DataParallel oder reine state_dict)."""
    obj = torch.load(ckpt, map_location="cpu")
    sd  = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    # Falls DataParallel-Präfix 'module.' vorhanden → entfernen
    clean = {k.replace("module.", "", 1): v for k, v in sd.items()}
    msg   = net.load_state_dict(clean, strict=False)
    if msg.missing_keys:
        print(f"[!] Missing keys for {ckpt.name}: {msg.missing_keys[:5]} ...")

def _get_backbone(name: str) -> torch.nn.Module:
    """
    Gibt initialisiertes Netz **ohne** Gewichte zurück.
    Muss nur dann angepasst werden, wenn ein Repo die API ändert.
    """
    if name == "TEED":
        # xavysp/TEED -> klassennamen 'TED'
        from ted import TED
        return TED()

    if name == "PiDiNet":
        # hellozhuo/pidinet -> Funktion pidinet('pidinet_small')
        from pidinet.pidinet import pidinet
        return pidinet("pidinet_small", pretrained=False)

    if name == "FINED":
        # jannctu/FINED -> Klasse FINED (liegt im fined/models/fined.py)
        from fined.models.fined import FINED
        return FINED()

    if name == "DexiNed":
        from dexined.dexined import DexiNed
        return DexiNed()

    if name == "CATS":
        # WHUHLX/CATS – Funktion build_cats_lite()
        from models.cats import build_cats_lite
        return build_cats_lite()

    if name == "EdgeNAT":
        # jhjie/EdgeNAT -> edgenat_large()
        from edgenat.modeling import edgenat_large
        return edgenat_large()

    if name == "DiffEdge":
        # GuHuangAI/DiffusionEdge
        from diffusionedge.models import create_model            # wrapper
        net, _ = create_model()                                  # returns (model, diffusion)
        net.eval()                                               # important: disable dropout
        return net

    if name == "UAED":
        # ZhouCX117/UAED_MuGE – net_UAED() helper
        from uaed.models import UAEDNet
        return UAEDNet()

    if name == "BDCN":
        # pkuCactus/BDCN – BDCN() constructor
        from bdcn.network import BDCN
        return BDCN()

    if name == "EDTER":
        # MengyangPu/EDTER – build_edter(cfg) via mmseg
        from edter.model import build_edter
        return build_edter("small")      # 'small' ≈ Swin-Tiny backbone

    raise ValueError(f"Unknown model '{name}'")


# ───────────────────────── Vor-/Nachverarbeitung ─────────────────────────

def _prepare_image(im_path: Path, size: int | None) -> torch.Tensor:
    """RGB BGR-Swap, Resize (long-edge==size)."""
    img = cv2.imread(str(im_path))[..., ::-1] / 255.0          # BGR→RGB, float
    if size:
        h, w = img.shape[:2]
        scale = size / max(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
    tens = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tens


def _forward(net: torch.nn.Module, x: torch.Tensor, name: str) -> torch.Tensor:
    """Abstraktionsschicht – jedes Netz liefert 1×1×H×W im 0-1 Bereich."""
    with torch.autocast("cuda", enabled=torch.cuda.is_available()):
        if name == "DiffEdge":
            # DiffusionEdge: net.sample() benötigt low-res latents,
            # wrapper akzeptiert volle Auflösung wenn 'x' Tensor rein
            pred = net.sample(x)["pred_edge"]
        else:
            pred = net(x)

    # unify output shapes
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if isinstance(pred, dict):
        pred = next(iter(pred.values()))
    if pred.dim() == 4:
        pred = pred[:, 0]           # (N,1,H,W) → (N,H,W)
    return torch.sigmoid(pred)      # garantiert [0,1]


# ─────────────────────────── Öffentliche API ────────────────────────────

def run_edge(models: Iterable[str],
             files:   list[Path],
             out_dir: Path,
             size:    int | None = None) -> None:
    """
    • `models`  – Liste Strings (siehe config.MODELS.keys)
    • `files`   – Liste Bild-Pfadobjekte
    • `out_dir` – Basisordner; Unterordner je Modell werden angelegt
    • `size`    – long-edge-Resize (None → Original-Auflösung)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in models:
        # 1) Gewichte sicherstellen -----------------------------------------
        ckpt = ensure_weight(MODELS[name])
        # 2) Netz beschaffen, Gewichte laden --------------------------------
        net  = _get_backbone(name).to(device).eval()
        _safe_load_state(net, ckpt)
        # 3) Output-Unterordner
        sub  = (out_dir / name)
        sub.mkdir(exist_ok=True)
        # 4) Alle Bilder durchjagen
        for f in files:
            inp  = _prepare_image(f, size).to(device)
            edge = _forward(net, inp, name)[0].cpu().numpy()     # (H,W)
            inv  = (1.0 - edge) * 255.0                         # invertieren
            cv2.imwrite(str(sub / f.name), inv.astype(np.uint8))
        del net
        torch.cuda.empty_cache()                                # VRAM frei
