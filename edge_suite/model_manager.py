"""Download & Package-Handling für Gewichte."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

import requests
import gdown
from tqdm import tqdm

from .config import MODELS, WEIGHTS_DIR, ModelInfo


# ────────────────────────── Hilfsfunktionen ──────────────────────────


def _install_missing_packages(pkgs: Iterable[str]) -> None:
    """Pip-Install einzelner Pakete, falls noch nicht vorhanden."""
    for pkg in pkgs:
        try:
            __import__(pkg)
        except ModuleNotFoundError:
            print(f"[+] pip install {pkg} …")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def _download_file(url: str, dest: Path) -> None:
    """Einfacher Downloader – unterstützt HTTP(S) und Google-Drive."""
    dest.parent.mkdir(exist_ok=True)

    if url.startswith("https://drive.google.com"):
        gdown.download(url, str(dest), quiet=False)
        return

    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)
                bar.update(len(chunk))


# ───────────────────────── Öffentliche API ───────────────────────────


def ensure_weight(mi: ModelInfo, force: bool = False) -> Path:
    """Lädt Gewichte in `edge_suite/weights/`, falls noch nicht vorhanden."""
    target = WEIGHTS_DIR / mi.default_weight
    if target.exists() and not force:
        return target

    print(f"[+] Lade Gewichte für {mi.name} …")
    _install_missing_packages(mi.extra_pkgs)
    _download_file(mi.weight_url, target)
    return target


def list_models_status() -> list[tuple[str, bool]]:
    """Liste (Modell­name, liegt Gewichts-Datei vor?)."""
    return [
        (name, (WEIGHTS_DIR / mi.default_weight).exists())
        for name, mi in MODELS.items()
    ]
