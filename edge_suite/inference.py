from __future__ import annotations
import cv2, torch, numpy as np
from pathlib import Path
from typing import Iterable
from .config import WEIGHTS_DIR
from .model_manager import ensure_weight
# Für Demo nutzen wir _platzhalter_ Netz – echtes Netz hier einbinden ↓↓↓
from torchvision.models.segmentation import lraspp_mobilenet_v3_large as _dummy_net

def _load_model_stub(name: str):
    """
    Vollständige Modell-Initialisierung ist repo-spezifisch.
    Für TEED/PiDiNet/etc. musst du hier das entspr. Backbone ersetzen.
    """
    net = _dummy_net(weights=None, num_classes=1)
    weight_path = WEIGHTS_DIR / f"{name.lower()}.pth"
    if weight_path.exists():
        state = torch.load(weight_path, map_location="cpu")
        net.load_state_dict(state, strict=False)
    return net

def prepare_image(im_path: Path, size: int|None=None) -> torch.Tensor:
    img = cv2.imread(str(im_path))[..., ::-1] / 255.0
    if size:
        h,w = img.shape[:2]
        scale = size / max(h,w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    img = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)
    return img

def run_edge(models: Iterable[str], files: list[Path], out_dir: Path, size: int|None=None):
    out_dir.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in models:
        sub = out_dir / model_name
        sub.mkdir(exist_ok=True)
        net = _load_model_stub(model_name).to(device).eval()
        for f in files:
            inp = prepare_image(f, size).to(device)
            with torch.no_grad():
                pred = net(inp)["out"].sigmoid()[0,0].cpu().numpy()
            # invert: weißer Hintergrund, dunkle Linien
            inv = (1.0-pred) * 255.0
            cv2.imwrite(str(sub/f.name), inv.astype(np.uint8))
