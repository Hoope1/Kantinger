"""
Einheitliche Inferenz­pipeline für die zehn Edge-Detector.

Highlights
* torch.inference_mode()  → kein Gradienten-Tracking
* robuster cv2.imread-Check
* autocast (fp16) auf CUDA
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any

import cv2
import numpy as np
import torch

from .config import MODELS, WEIGHTS_DIR
from .model_manager import ensure_weight


# ───────────────────────── State-Dict Loader ─────────────────────────


def _safe_load_state(net: torch.nn.Module, ckpt_path: Path) -> None:
    obj: Dict[str, Any] | torch.Tensor = torch.load(
        ckpt_path, map_location="cpu"
    )
    state: Dict[str, Any] = (
        obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    )
    clean = {
        k.removeprefix("module."): v  # DataParallel-Präfix strippen
        for k, v in state.items()
    }
    missing = net.load_state_dict(clean, strict=False).missing_keys
    if missing:
        print(f"[!] Warnung: {len(missing)} Keys passen nicht → {missing[:4]} …")


# ───────────────────── Backbone-Factory pro Modell ────────────────────


def _get_backbone(name: str) -> torch.nn.Module:
    """Neues, unge­wichtetes Modell aus dem jeweiligen Repo."""
    if name == "TEED":
        from ted import TED

        return TED()

    if name == "PiDiNet":
        from pidinet.pidinet import pidinet

        return pidinet("pidinet_small", pretrained=False)

    if name == "FINED":
        from fined.models.fined import FINED

        return FINED()

    if name == "DexiNed":
        from dexined.dexined import DexiNed

        return DexiNed()

    if name == "CATS":
        from models.cats import build_cats_lite

        return build_cats_lite()

    if name == "EdgeNAT":
        from edgenat.modeling import edgenat_large

        return edgenat_large()

    if name == "DiffEdge":
        from diffusionedge.models import create_model

        net, _ = create_model()  # returns (model, diffusion)
        return net

    if name == "UAED":
        from uaed.models import UAEDNet

        return UAEDNet()

    if name == "BDCN":
        from bdcn.network import BDCN

        return BDCN()

    if name == "EDTER":
        from edter.model import build_edter

        return build_edter("small")

    raise ValueError(f"Unbekanntes Modell: {name}")


# ─────────────────────── Helper: Bild I/O etc. ────────────────────────


def _prepare_image(path: Path, size: int | None) -> torch.Tensor:
    """Lädt Bild als Float-Tensor, RGB, [0..1]."""
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Kann Bild nicht laden: {path}")

    img = img_bgr[..., ::-1] / 255.0  # BGR→RGB

    if size:
        h, w = img.shape[:2]
        scale = size / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def _forward(
    net: torch.nn.Module, x: torch.Tensor, name: str
) -> torch.Tensor:
    """Gibt 1×H×W-Tensor mit Werten 0–1 zurück."""
    with torch.inference_mode(), torch.autocast(
        "cuda", enabled=torch.cuda.is_available()
    ):
        if name == "DiffEdge":
            edge = net.sample(x)["pred_edge"]  # type: ignore[attr-defined]
        else:
            edge = net(x)  # type: ignore[operator]

    if isinstance(edge, dict):
        edge = next(iter(edge.values()))
    if isinstance(edge, (list, tuple)):
        edge = edge[0]

    if edge.dim() == 4:
        edge = edge[:, 0]

    return torch.sigmoid(edge)  # (N,H,W)


# ───────────────────────── Öffentliche API ────────────────────────────


def run_edge(
    models: Iterable[str],
    files: list[Path],
    out_dir: Path,
    size: int | None = None,
) -> None:
    """Batch-Inferenz über mehrere Modelle & Dateien."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for name in models:
        mi = MODELS[name]
        ckpt = ensure_weight(mi)

        net = _get_backbone(name).to(device).eval()  # type: ignore[arg-type]
        _safe_load_state(net, ckpt)

        sub = out_dir / name
        sub.mkdir(exist_ok=True)

        for img_path in files:
            x = _prepare_image(img_path, size).to(device)
            y = _forward(net, x, name)[0].cpu().numpy()

            out = (1.0 - y) * 255.0  # invertiert: weißer BG
            if not cv2.imwrite(str(sub / img_path.name), out.astype(np.uint8)):
                print(f"[!] Konnte Ausgabedatei nicht schreiben: {img_path.name}")

        del net
        torch.cuda.empty_cache()
