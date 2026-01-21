#!/bin/bash
# Download V-JEPA 2 ViT-G pretrained weights
WEIGHTS_DIR="${1:-./weights}"
mkdir -p "$WEIGHTS_DIR"
wget -O "$WEIGHTS_DIR/vitg.pt" https://dl.fbaipublicfiles.com/vjepa2/vitg.pt
echo "Downloaded to $WEIGHTS_DIR/vitg.pt"
