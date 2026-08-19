"""
run.py

Required entry point per the official i4C Hackathon submission spec
(KLA Problem Statement -- AI-Based Restoration of Degraded Images).

Usage (positional arguments only, no flags):
  python run.py <input-dir> <output-dir>

Loads the final trained model (HighResSemiconductorNet, weights at
models/best_model_v3.pth), runs inference on every .npy file found in
<input-dir>, and writes restored .npy outputs to <output-dir> -- with
no manual configuration, no internet access, and no user interaction
required.

Inputs: .npy files, float32, shape (128, 128), values roughly in [0, ~1.4]
        (noisy/low-res, may exceed 1.0 due to noise overshoot).
Outputs: .npy files, float32, shape (256, 256), values clamped to [0, 1],
         no NaN/Inf, same filenames as input.
"""

import os
import sys
import time

import numpy as np
import torch

from model_architecture import HighResSemiconductorNet

# Hardcoded per spec: no manual configuration, no flags.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(_SCRIPT_DIR, "models", "best_model_v3.pth")
USE_TTA = True


def load_model(weights_path, device):
    model = HighResSemiconductorNet().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def restore_image(model, lr_array, device, tta=True):
    """Runs inference on a single (128,128) noisy/low-res array,
    returns a (256,256) restored array clamped to [0,1] with no NaN/Inf."""
    lr_tensor = torch.from_numpy(lr_array).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        if tta:
            out1 = model(lr_tensor)
            out2 = torch.flip(model(torch.flip(lr_tensor, dims=[3])), dims=[3])
            pred = (out1 + out2) / 2.0
        else:
            pred = model(lr_tensor)

    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    pred = torch.clamp(pred, 0, 1)
    return pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: No GPU detected. Inference will be slower.")

    print(f"Loading model weights from: {WEIGHTS_PATH}")
    model = load_model(WEIGHTS_PATH, device)

    filenames = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy") and not f.startswith("._"))
    if not filenames:
        raise RuntimeError(f"No .npy files found in input directory: {input_dir}")

    print(f"Found {len(filenames)} input images in: {input_dir}")
    print(f"Writing restored outputs to: {output_dir}")
    print(f"TTA (test-time flip averaging): ON")
    print()

    model_only_times = []
    pipeline_times = []
    for i, fname in enumerate(filenames, start=1):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        pipeline_start = time.time()

        lr_array = np.load(in_path).astype(np.float32)
        if lr_array.ndim != 2:
            raise ValueError(f"Expected a 2D array in {fname}, got shape {lr_array.shape}")

        model_start = time.time()
        restored = restore_image(model, lr_array, device, tta=USE_TTA)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_only_times.append(time.time() - model_start)

        if np.isnan(restored).any() or np.isinf(restored).any():
            raise RuntimeError(f"NaN/Inf detected in output for {fname} -- aborting.")

        np.save(out_path, restored)
        pipeline_times.append(time.time() - pipeline_start)

        if i % 50 == 0 or i == len(filenames):
            print(f"  Processed {i}/{len(filenames)} images...")

    avg_model_ms = (sum(model_only_times) / len(model_only_times)) * 1000
    avg_pipeline_ms = (sum(pipeline_times) / len(pipeline_times)) * 1000
    print()
    print("=" * 50)
    print(f"Done. Restored {len(filenames)} images.")
    print(f"Batch size: 1 (single-image inference loop)")
    print(f"Average model-only inference time: {avg_model_ms:.2f} ms/image")
    print(f"Average end-to-end pipeline time (load + inference + save): {avg_pipeline_ms:.2f} ms/image")
    print(f"Outputs written to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
