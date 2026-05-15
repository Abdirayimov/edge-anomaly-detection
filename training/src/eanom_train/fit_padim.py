"""Fit PaDiM Gaussian statistics over a folder of 'normal' images.

Pipeline:
  1. Walk a directory of normal-class images.
  2. Run a pretrained backbone (ResNet18 by default), capture features
     from selected stages, upsample everything to a common spatial
     size and concatenate along the channel dim.
  3. Random-subset the channel dim down to `feature_dim` (PaDiM trick
     to keep covariance tractable).
  4. For each spatial position (h, w): fit a multivariate Gaussian
     (mean + covariance) over all training images. Invert + regularise
     with `epsilon * I`.
  5. Dump everything to a single binary file consumable by the C++
     runtime's PaDiMStats::load. See padim_detector.hpp for the
     layout.

Usage:
    python -m eanom_train.fit_padim \
        --image-dir data/normal \
        --output models/stats/padim_industrial.bin \
        --onnx models/onnx/resnet18_features.onnx \
        --image-size 224 --feature-dim 100
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tqdm import tqdm

PADIM_MAGIC = 0x50414449  # 'PADI'


def _imagenet_normalise(arr: np.ndarray) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


def _build_backbone(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """Return a ResNet18 module that hooks layer1/2/3 outputs."""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval().to(device)

    activations: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def fn(_module, _inp, out):
            activations[name] = out.detach()
        return fn

    layers = ["layer1", "layer2", "layer3"]
    for name in layers:
        getattr(model, name).register_forward_hook(hook(name))
    return model, layers, activations  # type: ignore[return-value]


def _extract_features(
    model, layers, activations, image_path: Path, image_size: int, device: torch.device
) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = _imagenet_normalise(arr).to(device)

    with torch.no_grad():
        _ = model(tensor)

    feats: list[torch.Tensor] = []
    target_size = (image_size // 8, image_size // 8)
    for name in layers:
        t = activations[name]
        t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
        feats.append(t)
    cat = torch.cat(feats, dim=1).squeeze(0).cpu().numpy()
    return cat  # (C, H, W)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--feature-dim", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, layers, activations = _build_backbone(device)

    paths = sorted(p for p in args.image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise SystemExit(f"no images in {args.image_dir}")

    # Probe a single sample to learn the feature shape.
    sample = _extract_features(model, layers, activations, paths[0], args.image_size, device)
    C, H, W = sample.shape
    if args.feature_dim > C:
        raise SystemExit(f"feature_dim={args.feature_dim} exceeds backbone width {C}")

    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(C, size=args.feature_dim, replace=False)).astype(np.uint32)

    # Streaming mean / cov accumulators per position.
    D = args.feature_dim
    sum_mu = np.zeros((H, W, D), dtype=np.float64)
    sum_outer = np.zeros((H, W, D, D), dtype=np.float64)
    n = 0

    for p in tqdm(paths, desc="fitting"):
        feats = _extract_features(model, layers, activations, p, args.image_size, device)
        f = feats[selected]                      # (D, H, W)
        f = f.transpose(1, 2, 0)                 # (H, W, D)
        sum_mu += f.astype(np.float64)
        sum_outer += np.einsum("hwi,hwj->hwij", f.astype(np.float64), f.astype(np.float64))
        n += 1

    mu = sum_mu / n
    cov = sum_outer / n - np.einsum("hwi,hwj->hwij", mu, mu)
    cov += args.epsilon * np.eye(D, dtype=np.float64)[None, None, :, :]
    inv = np.linalg.inv(cov).astype(np.float32)
    mu = mu.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        f.write(struct.pack("<I", PADIM_MAGIC))
        f.write(struct.pack("<I", 1))            # version
        f.write(struct.pack("<I", H))
        f.write(struct.pack("<I", W))
        f.write(struct.pack("<I", D))
        f.write(struct.pack("<I", C))            # projection_dim_full
        f.write(selected.tobytes())
        f.write(mu.tobytes())
        f.write(inv.tobytes())

    print(f"wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"fit on {n} samples; feature dim H={H} W={W} D={D}, projection_dim_full={C}")


if __name__ == "__main__":
    main()
