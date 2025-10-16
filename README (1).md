# Echocardiography PanEcho Inference

> A production-minded workflow for running the **PanEcho** echocardiography model on cine **DICOM** files in **Google Colab** (and locally), featuring a robust **DICOM → NumPy → PyTorch** **pipeline, full model outputs, and visual QA—**

---

## Status & Links

-  
- **Upstream model:** `CarDS-Yale/PanEcho` (loaded via `torch.hub`)  
- **Owners:** Imaging/ML Platform Team

---

## Table of Contents

- [Highlights]
- [Overview]
- [Architecture]
- [Tech Stack]
- [Getting Started]
  - [Prerequisites]
  - [Quick Start (Colab)]
  - [Local Setup]
  - [Configuration]
- [Usage]
  - [Single-Study Inference]
  - [Outputs & Head Types]
  - [Visual QA]
- [Troubleshooting]
- [Contributing]
- [Security & Privacy]
- [Roadmap]
- [Acknowledgements]

---

## Highlights

- ** PanEcho:** load the official model via `torch.hub`.
- **Robust DICOM reader:** compressed syntaxes, VOI LUT/windowing, YBR→RGB, and **MONOCHROME1** inversion.
- **Model-ready tensors:** uniform temporal sampling, 224×224 resize, grayscale→3-channel, **ImageNet** normalization → `(1, 3, T, 224, 224)`.
- **Visual QA built-in:** first/middle/last frames, a montage of sampled frames, and an **MP4 cine**.
- **Full outputs (no truncation):** print every task head for inspection and debugging.
- **Self-contained Colab experience:** Drive mount + a single path variable (`DICOM_PATH`), CPU/GPU compatible.

---

## Overview

This repository provides a clean, reproducible notebook + helper code to run the **PanEcho** model on echocardiography cine **DICOM** files. The key value is a hardened input pipeline that turns real-world DICOMs into the exact video tensor PanEcho expects—without working or editing the upstream repo.

**What you get**
- A direct path (in Colab) from **DICOM → tensor → PanEcho outputs**.
- Strong handling of common DICOM quirks (compression, VOI LUT, color space, MONOCHROME1 polarity).
- Visual checks to confirm the model saw the right frames.

---

## Architecture

```
flowchart LR
  U[User (Colab / Local)] --> D[Google Drive / Filesystem]
  D --> F[DICOM (.dcm)]
  F -->|pydicom + pylibjpeg/GDCM| P[NumPy frames (T,H,W)]
  P -->|VOI LUT + YBR→RGB + MONOCHROME1 fix| C[Cleaned frames (uint8)]
  C -->|Uniform sample to 16| S[Sampled clip]
  S -->|Resize 224×224 + 3ch + ImageNet norm| T[(Tensor (1,3,T,224,224))]
  T --> M[PanEcho (torch.hub)]
  M --> O[Multi-task outputs (dict)]
  S --> V[Visualization (montage + MP4)]
```

**Data flow (condensed)**  
DICOM → `pixel_array` (auto-decompress) → color/VOI/polarity fixes → per-frame min–max → uniform sampling → resize + 3ch + ImageNet norm → PanEcho inference → outputs + visual QA.

---

## Tech Stack

- **Languages:** Python  
- **ML/CV:** PyTorch, NumPy, OpenCV  
- **DICOM:** pydicom, pylibjpeg, GDCM (CLI tools; optional Python bindings)  
- **Visualization:** Matplotlib, HTML video in Colab  
- **Runtime:** Google Colab (CPU/GPU), optional local execution

---

## Getting Started

### Prerequisites
- **Colab:** none (internet access enabled; GPU recommended).  
- **Local:** Python ≥ 3.9, PyTorch install (ideally with CUDA), and GDCM tools for `gdcmconv`.

### Quick Start (Colab)

1. Open the notebook:  
   
2. Run **Cell 1** to mount Google Drive and install dependencies.
3. Set the path to your file:
   ```python
   DICOM_PATH = "/content/drive/MyDrive/path/to/your_echo.dcm"
   ```
4. Run the remaining cells to:
   - Read DICOM → NumPy frames
   - Build `(1,3,T,224,224)` tensor (default `T=16`)
   - Load PanEcho via `torch.hub`
   - Infer, print outputs, and visualize

### Local Setup

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
python -m pip install --upgrade pip
pip install pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg opencv-python numpy pillow tqdm gdcm torch torchvision torchaudio

# 3) (Debian/Ubuntu) Install GDCM CLI (for gdcmconv)
sudo apt-get update && sudo apt-get -y install libgdcm-tools
```

### Configuration

Set these variables in the notebook (or your script):

```python
# Required: path to your cine DICOM
DICOM_PATH = "/content/drive/MyDrive/path/to/your_echo.dcm"

# Optional: preprocessing defaults
CLIP_LEN = 16
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

---

## Usage

### Single-Study Inference

```python
import numpy as np, torch, cv2, pydicom

# (1) Read & normalize DICOM → (T,H,W) uint8
frames = read_echo_dicom_frames(DICOM_PATH)

# (2) Convert → PanEcho tensor (1,3,T,224,224)
x = frames_to_panecho_tensor(frames, clip_len=16, size=224)

# (3) Load PanEcho from the official repo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = torch.hub.load("CarDS-Yale/PanEcho", "PanEcho", clip_len=16).to(device).eval()

# (4) Inference
with torch.no_grad():
    outputs = model(x.to(device))
```

### Outputs & Head Types

PanEcho is **multi-task**. Typical head types include:
- **Multi-class** (e.g., view classification): logits/probabilities over classes
- **Binary** (e.g., presence/absence): sigmoid/logits or 2-class scores
- **Regression** (e.g., EF): continuous value(s)

Inspect everything (no truncation):

```python
def to_np(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t

if isinstance(outputs, dict):
    print(f"Tasks returned: {len(outputs)}")
    for name, tensor in outputs.items():
        arr = to_np(tensor)
        print(f"{name:<28} shape={getattr(arr, 'shape', None)}")
        print(arr)  # beware: can be verbose
else:
    print(to_np(outputs))
```

### Visual QA

The notebook shows:
- **Key frames:** first / middle / last  
- **Montage:** the exact sampled frames fed to the model  
- **MP4 cine:** quick inline player for verification

---



## Troubleshooting

- **“Number of bytes of pixel data is less than expected … transfer syntax may be incorrect.”**  
  Some DICOMs have header/pixel mismatches. Run a one-shot repair:
  ```bash
  gdcmconv -w "in.dcm" "decompressed.dcm"
  ```
  Then point `DICOM_PATH` to `decompressed.dcm` and re-run.

- **DICOM path not found**  
  After mounting Drive in Colab, files live under `/content/drive/MyDrive/...`. Check:
  ```python
  import os; print(os.path.exists(DICOM_PATH), DICOM_PATH)
  ```

- **Contrast/polarity looks wrong**  
  Keep `apply_voi_lut` enabled; ensure **MONOCHROME1** inversion is applied when indicated by the header.

- **Slow on CPU**  
  In Colab, set **Runtime → Change runtime type → GPU**.

---

## Contributing

We welcome improvements (robustness, docs, examples).

1. **Open an issue** describing the change/problem.  
2. **Create a branch:**
   ```bash
   git checkout -b feat/short-description
   ```
3. **Standards:** add/adjust tests where applicable; keep style consistent.  
4. **Open a PR** with a clear summary (screenshots encouraged if UI/plots change).

---



## Roadmap

- Batch inference CLI for directories of DICOMs  
- Dataset-level evaluation harness + reports  
- Package DICOM helpers as a reusable Python module

---

## Acknowledgements

- **PanEcho** — model & code loaded via `torch.hub` from the official repository  
- **pydicom**, **pylibjpeg**, **GDCM**, **OpenCV**, **PyTorch** — indispensable tooling
