"""
train_teammate_arch_v3.py

Follow-up to train_teammate_arch.py (best_model_v2.pth). That run pushed
loss weighting to 0.2 L1 / 0.8 SSIM, which drove SSIM past 0.80 quickly
but left PSNR trailing (~25-26 dB, well short of the 28-30 dB target)
compared to the teammate's original 50/50-trained model re-verified on
our split (28.62-28.68 dB / 0.7959-0.7970).

This version dials the weighting back to a more balanced 0.35 L1 / 0.65 SSIM
-- still leaning toward SSIM (since it's a judged metric) but not as
aggressively -- to look for a better joint PSNR/SSIM outcome instead of
maximizing one at the expense of the other.

Everything else is identical to train_teammate_arch.py: same seed=42
split, same architecture (HighResSemiconductorNet), same early stopping
(patience=10), same "best checkpoint by SSIM" selection, same
compatibility with verify_teammate_model.py.

Usage:
  python train_teammate_arch_v3.py --gt_dir <path> --lr_dir <path> --out best_model_v3.pth
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from teammate_model import HighResSemiconductorNet


class WaferRestorationDataset(Dataset):
    def __init__(self, gt_dir, lr_dir, filenames, augment=False):
        self.gt_dir = gt_dir
        self.lr_dir = lr_dir
        self.filenames = filenames
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)

        if self.augment and np.random.rand() < 0.5:
            lr = np.ascontiguousarray(lr[:, ::-1])
            gt = np.ascontiguousarray(gt[:, ::-1])

        return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)


def get_val_filenames(gt_dir, val_size=300, seed=42):
    """EXACT same split as our train.py/evaluate.py/verify_teammate_model.py."""
    all_filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy") and not f.startswith("._"))
    rng = np.random.default_rng(seed=seed)
    shuffled = all_filenames.copy()
    rng.shuffle(shuffled)
    return shuffled[val_size:], shuffled[:val_size]  # train, val


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        mu1 = F.avg_pool2d(img1, self.window_size, 1, self.window_size // 2)
        mu2 = F.avg_pool2d(img2, self.window_size, 1, self.window_size // 2)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, 1, self.window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, 1, self.window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, 1, self.window_size // 2) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class HighFidelityLoss(nn.Module):
    def __init__(self, l1_weight=0.2, ssim_weight=0.8):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        return self.l1_weight * self.l1(pred, target) + self.ssim_weight * self.ssim(pred, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1_weight", type=float, default=0.35)
    parser.add_argument("--ssim_weight", type=float, default=0.65)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--out", default="best_model_v3.pth")
    parser.add_argument("--val_size", type=int, default=300)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_filenames, val_filenames = get_val_filenames(args.gt_dir, val_size=args.val_size)
    print(f"Train samples: {len(train_filenames)} | Val samples: {len(val_filenames)} (OUR seed=42 split)")

    train_dataset = WaferRestorationDataset(args.gt_dir, args.lr_dir, train_filenames, augment=args.augment)
    val_dataset = WaferRestorationDataset(args.gt_dir, args.lr_dir, val_filenames, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = HighResSemiconductorNet().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    criterion = HighFidelityLoss(l1_weight=args.l1_weight, ssim_weight=args.ssim_weight)
    print(f"Loss weights: L1={args.l1_weight}, SSIM={args.ssim_weight}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_ssim = 0.0
    epochs_without_improvement = 0

    print(f"\nTraining for up to {args.epochs} epochs (early stop after {args.patience} epochs without improvement)...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        for lr_batch, gt_batch in train_loader:
            lr_batch, gt_batch = lr_batch.to(device), gt_batch.to(device)
            optimizer.zero_grad()
            output = model(lr_batch)
            loss = criterion(output, gt_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_psnr, val_ssim = 0.0, 0.0
        ssim_metric = SSIMLoss()
        with torch.no_grad():
            for lr_batch, gt_batch in val_loader:
                lr_batch, gt_batch = lr_batch.to(device), gt_batch.to(device)
                output = torch.clamp(model(lr_batch), 0.0, 1.0)
                val_psnr += compute_psnr(output, gt_batch).item()
                val_ssim += (1.0 - ssim_metric(output, gt_batch)).item()

        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        epoch_time = time.time() - epoch_start

        marker = ""
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.out)
            marker = "  <- saved best"
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f} | Time: {epoch_time:.1f}s{marker}")

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
            break

    print(f"\nTraining complete. Best model saved to: {args.out}")
    print(f"Best SSIM (on OUR val split): {best_ssim:.4f}")


if __name__ == "__main__":
    main()
