import subprocess, sys
from pathlib import Path
import requests, gdown, shutil
from .config import MODELS, WEIGHTS_DIR, ModelInfo
from tqdm import tqdm

def _install_missing_packages(pkgs: list[str]):
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ModuleNotFoundError:
            print(f"[+] Installing extra pkg '{pkg}' ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def _download_file(url: str, dest: Path):
    dest.parent.mkdir(exist_ok=True)
    if url.startswith("https://drive.google.com"):
        gdown.download(url, str(dest), quiet=False)
    else:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

def ensure_weight(mi: ModelInfo, force: bool=False):
    target = WEIGHTS_DIR / mi.default_weight
    if target.exists() and not force:
        return target
    print(f"[+] Downloading weights for {mi.name} ...")
    _install_missing_packages(mi.extra_pkgs)
    _download_file(mi.weight_url, target)
    return target

def list_models_status():
    """Returns [(name, available_bool)]"""
    return [(n, (WEIGHTS_DIR/mi.default_weight).exists()) for n,mi in MODELS.items()]
