"""
compute_lpips.py

Computes LPIPS (Learned Perceptual Image Patch Similarity) for the final
model (best_model_v3.pth), on OUR exact seed=42 held-out validation split
-- the same 300 images used for every other reported PSNR/SSIM number in
this project, so all three metrics (PSNR, SSIM, LPIPS) are directly
comparable and consistent for the results slide.

LPIPS requires the `lpips` package (pip install lpips). It expects
3-channel images normalized to [-1, 1], so single-channel [0,1] arrays
are converted by repeating the channel 3x and rescaling.

Usage:
  pip install lpips
  python compute_lpips.py --weights best_model_v3.pth --gt_dir <path> --lr_dir <path>
  python compute_lpips.py --weights best_model_v3.pth --gt_dir <path> --lr_dir <path> --tta
"""

import argparse
import os

import numpy as np
import torch

from teammate_model import HighResSemiconductorNet

try:
    import lpips
except ImportError:
    raise ImportError(
        "The 'lpips' package is required for this script. Install it with:\n"
        "  pip install lpips"
    )


def get_val_filenames(gt_dir, val_size=300, seed=42):
    """EXACT same split as train.py/evaluate.py/verify_teammate_model.py --
    must match for LPIPS to be comparable to our PSNR/SSIM numbers."""
    all_filenames = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy") and not f.startswith("._"))
    rng = np.random.default_rng(seed=seed)
    shuffled = all_filenames.copy()
    rng.shuffle(shuffled)
    return shuffled[:val_size]


def to_lpips_input(arr_2d, device):
    """Converts a (H,W) array in [0,1] to a (1,3,H,W) tensor in [-1,1],
    as required by the lpips package."""
    t = torch.from_numpy(arr_2d).float()
    t = torch.clamp(t, 0, 1)
    t = t * 2.0 - 1.0  # [0,1] -> [-1,1]
    t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = t.repeat(1, 3, 1, 1)  # (1,3,H,W) -- grayscale replicated to 3 channels
    return t.to(device)


def main():
    parser = argparse.ArgumentParser(description="Compute LPIPS on OUR held-out validation set.")
    parser.add_argument("--weights", required=True, help="Path to trained model weights (.pth)")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--tta", action="store_true",
                         help="Enable Test-Time Augmentation (flip + average), matching how PSNR/SSIM were reported.")
    parser.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"],
                         help="Backbone network for LPIPS (alex is standard/fastest, matches common reporting convention).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HighResSemiconductorNet().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print(f"Loaded weights from: {args.weights}")

    print(f"Loading LPIPS ({args.net} backbone)...")
    loss_fn = lpips.LPIPS(net=args.net).to(device)
    loss_fn.eval()

    filenames = get_val_filenames(args.gt_dir)
    print(f"Evaluating LPIPS on {len(filenames)} HELD-OUT validation samples "
          f"(OUR seed=42 split -- same set used for PSNR/SSIM reporting).")
    print(f"TTA (test-time flip averaging): {'ON' if args.tta else 'OFF'}")

    total_lpips = 0.0

    with torch.no_grad():
        for fname in filenames:
            lr = np.load(os.path.join(args.lr_dir, fname)).astype(np.float32)
            gt = np.load(os.path.join(args.gt_dir, fname)).astype(np.float32)

            lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

            if args.tta:
                out1 = model(lr_tensor)
                out2 = torch.flip(model(torch.flip(lr_tensor, dims=[3])), dims=[3])
                pred = (out1 + out2) / 2.0
            else:
                pred = model(lr_tensor)

            pred_clamped = torch.clamp(pred, 0, 1).squeeze(0).squeeze(0).cpu().numpy()

            pred_lpips = to_lpips_input(pred_clamped, device)
            gt_lpips = to_lpips_input(gt, device)

            dist = loss_fn(pred_lpips, gt_lpips)
            total_lpips += dist.item()

    n = len(filenames)
    avg_lpips = total_lpips / n

    print()
    print("=" * 50)
    print(f"Samples evaluated: {n}")
    print(f"Average LPIPS ({args.net}): {avg_lpips:.4f}  (lower is better)")
    print("=" * 50)
    print()
    print("Use this alongside PSNR/SSIM from verify_teammate_model.py for your results slide:")
    print(f"  e.g. 'PSNR: XX.XX dB | SSIM: 0.XXXX | LPIPS: {avg_lpips:.4f}'")


if __name__ == "__main__":
    main()
