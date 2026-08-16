"""
evaluate.py

Standalone evaluation script for the wafer/semiconductor image restoration model.
Does NOT depend on the training notebook -- can be run independently by anyone
who has the model weights and the model architecture.

Two modes:
  1. --mode val   : compute PSNR/SSIM against ground truth (needs paired GT/NoisyLR data)
  2. --mode test   : run inference on unlabeled test images, save restored outputs

Usage:
  python evaluate.py --mode val --gt_dir path/to/GT --lr_dir path/to/NoisyLR --weights best_model.pt
  python evaluate.py --mode test --lr_dir path/to/Test_NoisyLR --weights best_model.pt --out_dir predictions/
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Model architecture (must match training exactly)
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class RestorationNet(nn.Module):
    def __init__(self, base_channels=64, num_res_blocks=6):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)]
        )
        self.bridge = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.exit = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = self.entry(x)
        res = self.res_blocks(feat)
        res = self.bridge(res)
        feat = feat + res
        up = self.upsample(feat)
        out = self.exit(up)
        return out


# ============================================================
# Metrics
# ============================================================

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def compute_ssim(pred, target, window_size=11):
    """
    Proper windowed SSIM using a Gaussian-weighted local window, matching
    standard implementations (e.g. pytorch-msssim, skimage). A naive
    whole-image SSIM is too lenient and gives inflated scores -- this
    computes local structural similarity and averages it, like the
    metric KLA will actually score you on.
    """
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)  # outer product -> 2D window

    device = pred.device
    window = gaussian_window(window_size).to(device).unsqueeze(0).unsqueeze(0)

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    pad = window_size // 2
    mu_pred = torch.nn.functional.conv2d(pred, window, padding=pad)
    mu_target = torch.nn.functional.conv2d(target, window, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = torch.nn.functional.conv2d(pred * pred, window, padding=pad) - mu_pred_sq
    sigma_target_sq = torch.nn.functional.conv2d(target * target, window, padding=pad) - mu_target_sq
    sigma_pred_target = torch.nn.functional.conv2d(pred * target, window, padding=pad) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim_map.mean().item()


# ============================================================
# Modes
# ============================================================

def get_val_filenames(gt_dir, val_size=300, seed=42):
    """
    Reproduces the EXACT same train/val split used during training
    (Cell 2, seed=42), so evaluation only scores the model on data it
    never saw during training. Without this, scores would be inflated
    because the model has memorized the training portion.
    """
    all_filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy") and not f.startswith("._"))

    rng = np.random.default_rng(seed=seed)
    shuffled = all_filenames.copy()
    rng.shuffle(shuffled)

    return shuffled[:val_size]


def run_validation(model, device, gt_dir, lr_dir, val_only=True):
    if val_only:
        filenames = get_val_filenames(gt_dir)
        print(f"Evaluating on {len(filenames)} HELD-OUT validation samples "
              f"(excludes training data, matches training split).")
    else:
        filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy") and not f.startswith("._"))
        print(f"Evaluating on all {len(filenames)} samples (includes training data -- "
              f"scores will be optimistic, not a true generalization measure).")

    total_psnr, total_ssim = 0.0, 0.0
    inference_times = []

    model.eval()
    with torch.no_grad():
        for fname in filenames:
            lr = np.load(os.path.join(lr_dir, fname)).astype(np.float32)
            gt = np.load(os.path.join(gt_dir, fname)).astype(np.float32)

            lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
            gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)

            start = time.time()
            pred = model(lr_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.time() - start)

            pred_clamped = torch.clamp(pred, 0, 1)

            total_psnr += compute_psnr(pred_clamped, gt_tensor).item()
            total_ssim += compute_ssim(pred_clamped, gt_tensor)

    n = len(filenames)
    print()
    print("===================================")
    print(f"Samples evaluated: {n}")
    print(f"Average PSNR: {total_psnr / n:.2f} dB")
    print(f"Average SSIM: {total_ssim / n:.4f}")
    print(f"Average inference time: {(sum(inference_times) / n) * 1000:.2f} ms/image")
    print("===================================")


def run_test_predictions(model, device, lr_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filenames = sorted(f for f in os.listdir(lr_dir) if f.endswith(".npy") and not f.startswith("._"))
    print(f"Running inference on {len(filenames)} unlabeled test images...")

    model.eval()
    with torch.no_grad():
        for fname in filenames:
            lr = np.load(os.path.join(lr_dir, fname)).astype(np.float32)
            lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

            pred = model(lr_tensor)
            pred_clamped = torch.clamp(pred, 0, 1)

            output_array = pred_clamped.squeeze(0).squeeze(0).cpu().numpy()
            np.save(os.path.join(out_dir, fname), output_array)

    print(f"Saved {len(filenames)} restored images to: {out_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate the restoration model.")
    parser.add_argument("--mode", choices=["val", "test"], required=True,
                         help="'val' computes PSNR/SSIM against ground truth. "
                              "'test' runs inference on unlabeled data and saves outputs.")
    parser.add_argument("--weights", required=True, help="Path to trained model weights (.pt file)")
    parser.add_argument("--lr_dir", required=True, help="Path to folder of NoisyLR .npy files")
    parser.add_argument("--gt_dir", default=None, help="Path to folder of GT .npy files (required for --mode val)")
    parser.add_argument("--out_dir", default="predictions", help="Where to save outputs (--mode test only)")
    parser.add_argument("--all_data", action="store_true",
                         help="Evaluate on ALL data instead of just the held-out validation split. "
                              "Not recommended for reporting scores -- inflates results since it "
                              "includes data the model was trained on.")
    parser.add_argument("--base_channels", type=int, default=64, help="Must match the architecture used in training")
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Must match the architecture used in training")
    args = parser.parse_args()

    if args.mode == "val" and args.gt_dir is None:
        parser.error("--gt_dir is required when --mode val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RestorationNet(base_channels=args.base_channels, num_res_blocks=args.num_res_blocks).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from: {args.weights}")

    if args.mode == "val":
        run_validation(model, device, args.gt_dir, args.lr_dir, val_only=not args.all_data)
    else:
        run_test_predictions(model, device, args.lr_dir, args.out_dir)


if __name__ == "__main__":
    main()
