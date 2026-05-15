"""Export ResNet18 features as ONNX for the C++ PaDiM runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision


class ResNet18Features(torch.nn.Module):
    """Concatenated layer1+layer2+layer3 features at the layer1 spatial
    resolution (image_size / 4 by default for 224 input -> 56x56)."""

    def __init__(self) -> None:
        super().__init__()
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self.stem = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        # Upsample b and c back to a's spatial resolution and concat
        # along channel.
        H, W = a.shape[-2:]
        b = torch.nn.functional.interpolate(b, size=(H, W), mode="bilinear",
                                            align_corners=False)
        c = torch.nn.functional.interpolate(c, size=(H, W), mode="bilinear",
                                            align_corners=False)
        return torch.cat([a, b, c], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("models/onnx/resnet18_features.onnx"))
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    model = ResNet18Features().eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(args.output),
        input_names=["images"], output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
