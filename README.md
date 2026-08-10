# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**KLA Hackathon 2026 — Track 1**
**Team:** _[Add your team name here]_
**Members:** _[Add teammate names here]_ — Viroshan (Reg. No. 212224060304), _[Add others]_

---

## Problem Statement

Semiconductor inspection images are frequently degraded by noise (speckle + Gaussian) and reduced resolution during acquisition. This project restores such degraded images — removing noise and upscaling resolution simultaneously — using a deep learning model trained on paired noisy/clean image data.

- **Input:** Noisy, low-resolution image (128×128)
- **Output:** Clean, full-resolution image (256×256)
- **Task type:** Joint denoising + 2× super-resolution

---

## Approach

### Model Architecture

A lightweight residual CNN (~666K parameters) designed for fast inference:

1. **Entry convolution** — extracts initial features from the noisy input
2. **6 residual blocks** — learn to separate noise from real signal via skip connections
3. **PixelShuffle upsampling** — reconstructs 128×128 features into 256×256 output without the checkerboard artifacts that naive upsampling/deconvolution can introduce
4. **Exit convolution** — refines and collapses back to a single-channel restored image

### Loss Function

A combined **L1 + SSIM loss** (50/50 weighted):
- L1 minimizes average pixel-wise error
- SSIM (Structural Similarity) directly optimizes for perceptual/structural accuracy — chosen because SSIM is one of the metrics this challenge is scored on

### Training Details

| | |
|---|---|
| Dataset | 3,200 paired GT/NoisyLR samples |
| Train / Validation split | 2,900 / 300 (seed=42, reproducible) |
| Epochs | 25 |
| Optimizer | Adam, lr=1e-3, ReduceLROnPlateau scheduler |
| Hardware | Google Colab, Tesla T4 GPU |
| Training time | ~26 minutes (25 epochs × ~63s) |

---

## Results

Evaluated on the **held-out validation set** (300 samples never seen during training):

| Metric | Score |
|---|---|
| PSNR | 28.12 dB |
| SSIM | 0.7784 |
| Inference speed (GPU) | 5.88 ms/image |
| Inference speed (CPU) | ~707 ms/image |

### Visual Results

See `sample_results.png` for side-by-side comparisons of Noisy Input / Model Output / Ground Truth.

### Known Limitations

- **Fine-texture content** (e.g. sand, grain-heavy surfaces) shows visible streaking artifacts in the model output rather than accurately reconstructed texture. This is a capacity limitation of the current lightweight architecture on high-frequency detail, not a training or data issue.
- Outputs are somewhat softer than ground truth on average, consistent with the SSIM score — there is headroom for improvement with a larger model or longer training.

---

## Dataset

The `dataset/` folder (3,200 paired GT/NoisyLR `.npy` files) is **not included in this repository** to keep it lightweight — it's the official KLA hackathon dataset. To reproduce training or evaluation:

1. Obtain `train.zip` from the official hackathon dataset source
2. Extract it so you have `dataset/GT/` and `dataset/NoisyLR/`, each containing matching `.npy` files
3. Place the `dataset/` folder in the project root before running `train.py` or `evaluate.py`

---

## Repository Structure

```
restoration_project/
├── train.py              # Training script (reproduces best_model.pt from scratch)
├── evaluate.py            # Standalone evaluation script (no training dependencies)
├── model.py                # Model architecture (RestorationNet)
├── best_model.pt            # Trained model weights
├── requirements.txt          # Python dependencies
├── sample_results.png         # Visual comparison grid
├── dataset/
│   ├── GT/                     # Ground truth images (.npy, 256x256)
│   └── NoisyLR/                # Noisy low-res inputs (.npy, 128x128)
└── README.md
```

---

## How to Run

### Setup

```bash
pip install -r requirements.txt
```

### Evaluate the trained model (reproduce reported scores)

```bash
python evaluate.py --mode val --weights best_model.pt --gt_dir dataset/GT --lr_dir dataset/NoisyLR
```

This evaluates only on the 300 held-out validation images (same split used during training, reproduced via fixed random seed) and prints PSNR, SSIM, and inference speed.

### Run inference on new/unlabeled test data

```bash
python evaluate.py --mode test --weights best_model.pt --lr_dir path/to/test_images --out_dir predictions/
```

Saves restored `.npy` outputs for every input image, without requiring ground truth.

### Retrain from scratch

```bash
python train.py
```

---

## Tech Stack

- **Framework:** PyTorch
- **Training environment:** Google Colab (Tesla T4 GPU)
- **Key libraries:** `torch`, `numpy` — both `train.py` and `evaluate.py` use a self-contained, hand-verified windowed SSIM implementation (matches the `pytorch-msssim` reference library to within 0.0007), so neither script depends on external SSIM packages

---

## Team Notes

_[Add anything else your team wants judges to know — challenges faced, what you'd improve with more time, individual contributions, etc.]_
