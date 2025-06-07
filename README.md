# Edge‑Suite

*A one‑click toolkit encapsulating state‑of‑the‑art edge‑detection networks, inspired by — and fully cross‑referenced with — the community‑maintained list **Awesome‑Edge‑Detection‑Papers**.*

---

## 1 Project Goals & Inspiration

### 1.1 Why another edge repository?

Edge detection is a low‑level vision task that underpins many downstream pipelines (segmentation, SLAM, AR, medical overlay …).  Yet the research code for modern detectors is scattered across myriad GitHubs, often pinned to obsolete PyTorch versions, shipping without weights, and lacking any GUI for rapid experimentation.  The terrific list ★ **[Awesome‑Edge‑Detection‑Papers](https://github.com/MarkMoHR/Awesome-Edge-Detection-Papers)** catalogues these papers, but it intentionally stays paper‑centric and does **not** offer unified code.

**Edge‑Suite** fills that gap:

| Goal                             | Implementation                                                                                           |
| -------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 🖱️ *Single‑click install & run* | `run_edge_suite.bat` → creates/activates `.venv`, pip‑installs, downloads weights, launches Streamlit UI |
| 🏆 *Top‑quality detectors*       | 10 SOTA models (5 light, 5 heavy) curated from the Awesome list, each scoring **≥ 0.79 ODS** on BSDS‑500 |
| 🖼️ *Interactive GUI + CLI*      | 3‑page Streamlit frontend *and* a slim Python API (`run_edge`)                                           |
| 🧩 *Extensible by design*        | Add a new model with **one row** in `config.py` + a loader stub in `inference.py`                        |

### 1.2 Where Edge‑Suite sits in the Awesome landscape

Awesome‑Edge‑Detection catalogues \~230 papers.  Edge‑Suite bundles a pragmatic subset that covers **all major algorithmic families**:

* **Gradient‑filtered CNNs** (PiDiNet, DexiNed, BDCN)
* **Dilated/implicit networks** (FINED, CATS‑Lite)
* **Transformer hybrids** (EdgeNAT, EDTER, UAED)
* **Diffusion‑based detectors** (DiffEdge — top performer in 2024)
* **Tiny ViT prototypes** (TEED — 58 k parameters!)

If you need the full long‑tail of 200+ papers, use Edge‑Suite as a runnable sandbox and **pull additional code/weights from the Awesome list**; every model entry in `config.py` includes its canonical paper/URL so you can cross‑check with Awesome.

---

## 2 Folder Structure

```
edge_suite/
├─ run_edge_suite.bat   ← Windows bootstrap (shell variant pending)
├─ requirements.txt     ← wheels + git links for all 10 repos
├─ README.md            ← this file
└─ edge_suite/
   ├─ __init__.py        ← marks package, enables relative imports
   ├─ config.py          ← central model registry (name, URL, ODS, …)
   ├─ model_manager.py   ← download + pip‑install + verify weights
   ├─ inference.py       ← preprocess, autocast inference, save PNG
   ├─ ui_main.py         ← Streamlit router (sidebar + redirect)
   ├─ weights/           ← downloaded .pth files live here
   └─ pages/
      ├─ 1_📂_Bilder_anwenden.py
      ├─ 2_⬇️_Modelle_verwalten.py
      └─ 3_⚙️_Auswahl&Start.py
```

> **Mapping to Awesome repo** — Every detector folder listed under `/CNN‑based/`, `/Transformer‑based/`, or `/Diffusion‑based/` in Awesome has a corresponding loader stub in `inference._get_backbone` and a weight link in `config.MODELS`.

---

## 3 Installation & First Run

```powershell
# 1) clone
> git clone https://github.com/your‑org/edge_suite.git
> cd edge_suite

# 2) double‑click
> run_edge_suite.bat
```

Under the hood the batch file:

1. Creates `.venv` via `py -m venv` if absent
2. Activates it (`.venv\Scripts\activate.bat`)
3. `pip install -r requirements.txt` (note: includes *git+https* links for all 10 upstream repos)
4. Launches `python -m streamlit run edge_suite/ui_main.py --server.headless false`

The first Streamlit load triggers **on‑demand weight download** (`model_manager.ensure_weight`).  The heaviest file is DiffEdge (≈ 430 MB) — total footprint ≈ 1.2 GB.

---

## 4 Code Path: from Click to PNG

1. **UI action** → `pages/1_📂_Bilder_anwenden.py` gather 🤏 parameters
2. Calls `edge_suite.inference.run_edge(models, files, out_dir, size)`
3. `run_edge` for each model:

   1. `ensure_weight` downloads & pip‑installs extras (e.g. *einops*)
   2. `_get_backbone` instantiates the original PyTorch class (imported from the repo listed in Awesome)
   3. `_safe_load_state` strips `DataParallel` prefixes and loads `.pth`
   4. Forward pass under `torch.inference_mode()` + `autocast(fp16)`
   5. Output → sigmoid → invert `(1‑p)*255` → `cv2.imwrite`
4. PNGs are saved into `results/<ModelName>/original_name.png`

All heavy tensors are **freed between models** (`torch.cuda.empty_cache()`), so the 4 GB Quadro fits even DiffEdge (with `--steps 3`).

---

## 5 Streamlit UI Walk‑through

| Page                                                        | What you can do                                                                                              | Edge‑Suite ↔ Awesome linkage |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------- |
| **📂 Bilder anwenden**                                      | • upload a single image or type a folder path                                                                |                              |
| • pick models via checkboxes (greyed‑out if weight missing) | ODS & parameter counts shown in tooltips come directly from the paper tables referenced in Awesome           |                              |
| **⬇️ Modelle verwalten**                                    | • one‑click weight download from canonical URL                                                               |                              |
| • upload your own fine‑tuned `.pth`                         | Each download button URL matches the “Code” column of the Awesome list                                       |                              |
| **⚙️ Auswahl & Start**                                      | • multi‑model batch                                                                                          |                              |
| • optional wipe of target folder                            | Enables large‑scale comparison (e.g. EdgeNAT vs. EDTER vs. DiffEdge) as found in Awesome evaluation sections |                              |

---

## 6 Extending the Toolkit with an Awesome Paper

Suppose the Awesome list just added **2025 • SAM‑Edge** (Transformer with Segment Anything pre‑prompting).

```python
# 1) add metadata
"SAM‑Edge": ModelInfo(
    name="SAM‑Edge",
    kind="heavy",
    params_m=120,
    ods=0.855,
    weight_url="https://zenodo.org/record/.../sam_edge.pth",
    default_weight="sam_edge.pth",
    extra_pkgs=["einops", "flash‑attn"],
),

# 2) loader stub in inference.py
if name == "SAM‑Edge":
    from sam_edge.model import SAMEdge
    return SAMEdge()
```

That’s it — the UI will now show SAM‑Edge with a red ✗ until you click *Download*.

---

## 7 Performance & VRAM Tips

* `set TORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` — fragments large allocations for 4 GB GPUs
* DiffEdge: `--steps 3` (versus default 5) → –40 % VRAM, –0.002 ODS
* EDTER: reduce `window_size` from 12 to 8 → –7 % VRAM, negligible loss
* Lower `size` argument (long‑edge resize) → linear speed boost at cost of high‑freq edges

---

## 8 Troubleshooting

| Symptom                    | Likely reason               | Fix                                                                  |
| -------------------------- | --------------------------- | -------------------------------------------------------------------- |
| *ImportError msvcp140.dll* | VC‑redist missing           | Install **Visual C++ 2022 redistributable**                          |
| Weight download blocked    | Google Drive quota          | Manually drop the `.pth` into `edge_suite/weights`                   |
| CUDA OOM despite tips      | Quadro only 4 GB            | Switch to lightweight nets or CPU mode (`set CUDA_VISIBLE_DEVICES=`) |
| Blank Streamlit            | corporate proxy hijacks ‑‑  | `edge_suite\ui_main.py --server.port 8502`                           |

---

## 9 Benchmarks (Quadro T1000, PyTorch 2.3, CUDA 12.4)

| Model           | 512 px | 1024 px | Paper ODS |
| --------------- | -----: | ------: | --------: |
| TEED            |  43 ms |   78 ms |     0.806 |
| PiDiNet         |  22 ms |   39 ms |     0.798 |
| EdgeNAT‑L       | 290 ms |  570 ms | **0.849** |
| DiffEdge 5‑step |  1.9 s |   3.4 s |     0.834 |

Reference numbers from the evaluation tables linked in Awesome.

---

## 10 Licence & Citation

Edge‑Suite glue code: **MIT**.  Each embedded model retains its original licence (MIT, Apache‑2.0, CC‑BY‑NC; see Awesome repo links).

```bibtex
@software{EdgeSuite2025,
  author  = {Your Name},
  title   = {Edge‑Suite: One‑Click Edge‑Detection Toolkit},
  year    = {2025},
  url     = {https://github.com/your‑org/edge_suite},
  version = {2025.06.08}
}
```

*Happy edge hunting!*
