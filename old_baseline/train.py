"""
train.py

Standalone training script for the image restoration model.
Reproduces best_model.pt from scratch given the GT/NoisyLR dataset.

Usage:
  python train.py --gt_dir dataset/GT --lr_dir dataset/NoisyLR --epochs 25 --out best_model.pt

Requires a GPU for reasonable training time (~1 min/epoch on a T4).
Will run on CPU if no GPU is available, but expect it to be far slower.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import RestorationNet


# ============================================================
# Dataset
# ============================================================

class WaferRestorationDataset(Dataset):
    """
    Loads paired (NoisyLR, GT) samples.
    Does NOT clip/normalize away noise overshoot -- the model needs to
    see realistic noisy input, including values outside [0,1].
    """

    def __init__(self, gt_dir, lr_dir, filenames, augment=False):
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.filenames = filenames
        self.augment = augment  # only True for the training split, never validation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        if self.augment and np.random.rand() < 0.5:
            # Flip BOTH images the same way -- they must stay aligned as a pair
            lr = np.ascontiguousarray(lr[:, ::-1])
            gt = np.ascontiguousarray(gt[:, ::-1])

        return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)


def get_train_val_split(gt_dir, val_size=300, seed=42):
    """Fixed-seed split so training and evaluation always agree on
    which images are held out."""
    all_filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy") and not f.startswith("._"))

    rng = np.random.default_rng(seed=seed)
    shuffled = all_filenames.copy()
    rng.shuffle(shuffled)

    val_filenames = shuffled[:val_size]
    train_filenames = shuffled[val_size:]
    return train_filenames, val_filenames


# ============================================================
# Loss
# ============================================================

class CombinedLoss(nn.Module):
    """
    L1 loss: gets pixel values close on average.
    SSIM loss: keeps edges/structure sharp and matches human perception.
    Combined 50/50 -- SSIM is one of the metrics this challenge is scored on.
    """

    def __init__(self, l1_weight=0.5, ssim_weight=0.5, window_size=11):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size

    def _gaussian_window(self, size, sigma=1.5, device="cpu"):
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    def _ssim(self, pred, target):
        window = self._gaussian_window(self.window_size, device=pred.device)
        pad = self.window_size // 2
        C1, C2 = 0.01 ** 2, 0.03 ** 2

        mu_pred = nn.functional.conv2d(pred, window, padding=pad)
        mu_target = nn.functional.conv2d(target, window, padding=pad)
        mu_pred_sq, mu_target_sq = mu_pred ** 2, mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = nn.functional.conv2d(pred * pred, window, padding=pad) - mu_pred_sq
        sigma_target_sq = nn.functional.conv2d(target * target, window, padding=pad) - mu_target_sq
        sigma_pred_target = nn.functional.conv2d(pred * target, window, padding=pad) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - self._ssim(torch.clamp(pred, 0, 1), target)
        return self.l1_weight * l1_loss + self.ssim_weight * ssim_loss


# ============================================================
# Training
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train the image restoration model.")
    parser.add_argument("--gt_dir", required=True, help="Path to folder of GT .npy files")
    parser.add_argument("--lr_dir", required=True, help="Path to folder of NoisyLR .npy files")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out", default="best_model.pt", help="Where to save the best checkpoint")
    parser.add_argument("--val_size", type=int, default=300, help="Number of samples held out for validation")
    parser.add_argument("--base_channels", type=int, default=64, help="Model width")
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Model depth")
    parser.add_argument("--l1_weight", type=float, default=0.5, help="Weight for L1 loss component")
    parser.add_argument("--ssim_weight", type=float, default=0.5, help="Weight for SSIM loss component")
    parser.add_argument("--augment", action="store_true", help="Enable random horizontal flip augmentation (train split only)")
    parser.add_argument("--patience", type=int, default=7, help="Stop early if val loss doesn't improve for this many epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: No GPU detected. Training will be significantly slower.")

    train_filenames, val_filenames = get_train_val_split(args.gt_dir, val_size=args.val_size)
    print(f"Train samples: {len(train_filenames)} | Val samples: {len(val_filenames)}")

    train_dataset = WaferRestorationDataset(args.gt_dir, args.lr_dir, train_filenames, augment=args.augment)
    val_dataset = WaferRestorationDataset(args.gt_dir, args.lr_dir, val_filenames, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RestorationNet(base_channels=args.base_channels, num_res_blocks=args.num_res_blocks).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} (base_channels={args.base_channels}, num_res_blocks={args.num_res_blocks})")

    criterion = CombinedLoss(l1_weight=args.l1_weight, ssim_weight=args.ssim_weight)
    print(f"Loss weights: L1={args.l1_weight}, SSIM={args.ssim_weight}")
    print(f"Augmentation: {'ON (random horizontal flip)' if args.augment else 'OFF'}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print(f"\nTraining for up to {args.epochs} epochs (early stop after {args.patience} epochs without improvement)...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        for lr_batch, gt_batch in train_loader:
            lr_batch, gt_batch = lr_batch.to(device), gt_batch.to(device)

            optimizer.zero_grad()
            output = model(lr_batch)
            loss = criterion(output, gt_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * lr_batch.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for lr_batch, gt_batch in val_loader:
                lr_batch, gt_batch = lr_batch.to(device), gt_batch.to(device)
                output = model(lr_batch)
                loss = criterion(output, gt_batch)
                running_val_loss += loss.item() * lr_batch.size(0)

        val_loss = running_val_loss / len(val_dataset)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.out)
            marker = "  <- saved best"
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:2d}/{args.epochs} | Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | Time: {epoch_time:.1f}s{marker}")

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nTraining complete. Best model saved to: {args.out}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
