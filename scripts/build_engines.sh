#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="${ROOT}/models/onnx"
ENG_DIR="${ROOT}/models/engines"
mkdir -p "${ENG_DIR}"

if [[ -f "${ONNX_DIR}/resnet18_features.onnx" ]]; then
    echo "building ResNet18-features engine..."
    trtexec --onnx="${ONNX_DIR}/resnet18_features.onnx" \
        --saveEngine="${ENG_DIR}/resnet18_features_fp16.engine" \
        --fp16 --workspace=4096
fi

ls -lh "${ENG_DIR}/"
