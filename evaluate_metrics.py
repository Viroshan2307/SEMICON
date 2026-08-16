"""
evaluate_metrics.py

Independently verifies the teammate's trained model against OUR exact
held-out validation split (same seed=42, same 300 images used to score
our own baseline/exp1/exp2), using OUR verified windowed SSIM/PSNR --
NOT the teammate script's own train/val split or metrics.

This matters because the teammate's script splits data by taking the
LAST 10% of sorted filenames (not a random shuffle), which is a
different, non-comparable validation set. Without re-evaluating on our
own split, "performed well" can't be honestly compared to our own
28.36 dB / 0.7762 baseline.

Usage:
  python evaluate_metrics.py --weights best_model.pth --gt_dir dataset/GT --lr_dir dataset/NoisyLR
  python evaluate_metrics.py --weights best_model.pth --gt_dir dataset/GT --lr_dir dataset/NoisyLR --tta
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model_architecture import HighResSemiconductorNet


def get_val_filenames(gt_dir, val_size=300, seed=42):
    """EXACT same split logic as our train.py/evaluate.py -- must match
    for the comparison to be fair."""
    all_filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy") and not f.startswith("._"))
    rng = np.random.default_rng(seed=seed)
    shuffled = all_filenames.copy()
    rng.shuffle(shuffled)
    return shuffled[:val_size]


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def compute_ssim(pred, target, window_size=11):
    """Our own verified windowed SSIM -- matches pytorch-msssim to
    within 0.0007 (checked earlier in this project)."""
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    device = pred.device
    window = gaussian_window(window_size).to(device).unsqueeze(0).unsqueeze(0)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad)
    mu_target = F.conv2d(target, window, padding=pad)
    mu_pred_sq, mu_target_sq = mu_pred ** 2, mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    return ssim_map.mean().item()


def main():
    parser = argparse.ArgumentParser(description="Verify teammate's model on OUR held-out validation set.")
    parser.add_argument("--weights", default="/kaggle/working/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--gt_dir", default="/kaggle/input/datasets/viroshans/kla-hackathon/train/GT")
    parser.add_argument("--lr_dir", default="/kaggle/input/datasets/viroshans/kla-hackathon/train/NoisyLR")
    parser.add_argument("--tta", action="store_true",
                         help="Enable Test-Time Augmentation (flip + average) as the teammate's own eval script does")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HighResSemiconductorNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Loaded weights from: {args.weights}")

    filenames = get_val_filenames(args.gt_dir)
    print(f"Evaluating on {len(filenames)} HELD-OUT validation samples (OUR seed=42 split, "
          f"same set used for baseline/exp1/exp2 -- directly comparable).")
    print(f"TTA (test-time flip averaging): {'ON' if args.tta else 'OFF'}")

    total_psnr, total_ssim = 0.0, 0.0
    inference_times = []

    with torch.no_grad():
        for fname in filenames:
            lr = np.load(os.path.join(args.lr_dir, fname)).astype(np.float32)
            gt = np.load(os.path.join(args.gt_dir, fname)).astype(np.float32)

            lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
            gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)

            start = time.time()

            if args.tta:
                out1 = model(lr_tensor)
                out2 = torch.flip(model(torch.flip(lr_tensor, dims=[3])), dims=[3])
                pred = (out1 + out2) / 2.0
            else:
                pred = model(lr_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.time() - start)

            pred_clamped = torch.clamp(pred, 0, 1)

            total_psnr += compute_psnr(pred_clamped, gt_tensor).item()
            total_ssim += compute_ssim(pred_clamped, gt_tensor)

    n = len(filenames)
    print()
    print("=" * 50)
    print(f"Samples evaluated: {n}")
    print(f"Average PSNR: {total_psnr / n:.2f} dB")
    print(f"Average SSIM: {total_ssim / n:.4f}")
    print(f"Average inference time: {(sum(inference_times) / n) * 1000:.2f} ms/image")
    print("=" * 50)
    print()
    print("FOR COMPARISON (same 300-image held-out set, our own model):")
    print("  Baseline:     PSNR 28.12 dB | SSIM 0.7686")
    print("  Experiment 1: PSNR 28.36 dB | SSIM 0.7762")


if __name__ == "__main__":
    main()
