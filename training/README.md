# eanom_train

Python helpers to produce the artefacts the C++ runtime consumes:

- `export_backbone.py` exports ResNet18's layer1+2+3 concatenated
  features to ONNX. Build the `.engine` from this ONNX via
  `scripts/build_engines.sh`.
- `fit_padim.py` walks a folder of "normal" images, runs the
  backbone, and fits the per-position multivariate Gaussian
  statistics. Output is a single `.bin` file the C++ side reads.

## Recipe

```bash
cd training
pip install -e .[dev]

# 1. Export the backbone to ONNX.
python -m eanom_train.export_backbone --output ../models/onnx/resnet18_features.onnx

# 2. Compile a TensorRT engine.
cd ..
./scripts/build_engines.sh

# 3. Fit PaDiM stats on your "normal" data.
python -m eanom_train.fit_padim \
    --image-dir data/normal \
    --output models/stats/padim_industrial.bin \
    --image-size 224 \
    --feature-dim 100

# 4. Enable PaDiM in configs/system_config.yaml:
#       detectors.padim.enabled: true
#       detectors.padim.backbone_engine_path: models/engines/resnet18_features_fp16.engine
#       detectors.padim.stats_path: models/stats/padim_industrial.bin
```

200-500 normal images is usually enough; PaDiM is sample-efficient
because it fits one Gaussian per spatial position, not over the
whole feature space jointly.
