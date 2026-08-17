"""
evaluate_final.py

Standalone evaluation/inference script for the final submitted model
(HighResSemiconductorNet, trained via train_teammate_arch_v3.py,
weights: best_model_v3.pth).

Per KLA hackathon submission requirements: this script accepts a path
to a test images directory and a path to an output directory. It loads
the trained model, runs inference on every .npy input in the test
directory, and writes restored (denoised + 2x super-resolved) outputs
to the output directory -- with no manual edits required.

Usage:
  python evaluate_final.py --input_dir path/to/NoisyLR --output_dir path/to/restored_outputs
  python evaluate_final.py --input_dir path/to/NoisyLR --output_dir path/to/restored_outputs --tta
  python evaluate_final.py --input_dir path/to/NoisyLR --output_dir path/to/restored_outputs --weights custom_weights.pth

Inputs: .npy files, float32, shape (128, 128), values roughly in [0, ~1.4]
        (noisy/low-res, may exceed 1.0 due to noise overshoot).
Outputs: .npy files, float32, shape (256, 256), values clamped to [0, 1]
         (restored, denoised, 2x super-resolved), same filenames as input.
"""

import argparse
import os
import time

import numpy as np
import torch

from model_architecture import HighResSemiconductorNet


def load_model(weights_path, device):
    model = HighResSemiconductorNet().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def restore_image(model, lr_array, device, tta=False):
    """Runs inference on a single (128,128) noisy/low-res array,
    returns a (256,256) restored array clamped to [0,1]."""
    lr_tensor = torch.from_numpy(lr_array).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        if tta:
            out1 = model(lr_tensor)
            out2 = torch.flip(model(torch.flip(lr_tensor, dims=[3])), dims=[3])
            pred = (out1 + out2) / 2.0
        else:
            pred = model(lr_tensor)

    pred = torch.clamp(pred, 0, 1)
    return pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with the final restoration model on a directory of noisy/low-res .npy images."
    )
    parser.add_argument("--input_dir", required=True,
                         help="Path to directory containing noisy low-res .npy input images.")
    parser.add_argument("--output_dir", required=True,
                         help="Path to directory where restored .npy outputs will be written.")
    parser.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model_v3.pth"),
                         help="Path to trained model weights (default: best_model_v3.pth alongside this script).")
    parser.add_argument("--tta", action="store_true",
                         help="Enable test-time augmentation (flip + average) for slightly improved output quality.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Model weights not found: {args.weights}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: No GPU detected. Inference will be slower.")

    print(f"Loading model weights from: {args.weights}")
    model = load_model(args.weights, device)

    filenames = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy") and not f.startswith("._"))
    if not filenames:
        raise RuntimeError(f"No .npy files found in input directory: {args.input_dir}")

    print(f"Found {len(filenames)} input images in: {args.input_dir}")
    print(f"Writing restored outputs to: {args.output_dir}")
    print(f"TTA (test-time flip averaging): {'ON' if args.tta else 'OFF'}")
    print()

    model_only_times = []
    pipeline_times = []
    for i, fname in enumerate(filenames, start=1):
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        pipeline_start = time.time()

        lr_array = np.load(in_path).astype(np.float32)
        if lr_array.ndim != 2:
            raise ValueError(f"Expected a 2D array in {fname}, got shape {lr_array.shape}")

        model_start = time.time()
        restored = restore_image(model, lr_array, device, tta=args.tta)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_only_times.append(time.time() - model_start)

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
    print(f"Outputs written to: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
