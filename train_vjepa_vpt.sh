#!/bin/bash

# V-JEPA VPT Fine-Tuning - Quick Start Script
# This script guides you through the fine-tuning process

set -e  # Exit on error

echo "========================================="
echo "V-JEPA Fine-Tuning on VPT Dataset"
echo "========================================="
echo ""

# Configuration
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-checkpoints/vjepa_vit-l-16.pt}"
VPT_DATA_PATH="${VPT_DATA_PATH:-VPT/shard-*.tar}"
CONFIG_FILE="${CONFIG_FILE:-vjepa2/configs/train/vitl16/minecraft-256px-8f.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
SCALER_PATH="vpt_action_scaler.pkl"

echo "Configuration:"
echo "  Pre-trained checkpoint: $PRETRAIN_CHECKPOINT"
echo "  VPT data path: $VPT_DATA_PATH"
echo "  Config file: $CONFIG_FILE"
echo "  Number of GPUs: $NUM_GPUS"
echo ""

# Step 1: Check dependencies
echo "Step 1: Checking dependencies..."
if ! python -c "import torch, webdataset, decord, sklearn" 2>/dev/null; then
    echo "  ⚠ Missing dependencies. Installing..."
    pip install -r requirements.txt
else
    echo "  ✓ Dependencies installed"
fi
echo ""

# Step 2: Check pre-trained checkpoint
echo "Step 2: Checking pre-trained checkpoint..."
if [ ! -f "$PRETRAIN_CHECKPOINT" ]; then
    echo "  ⚠ Pre-trained checkpoint not found at: $PRETRAIN_CHECKPOINT"
    echo ""
    echo "  Download options:"
    echo "    1. ViT-B/16: wget https://dl.fbaipublicfiles.com/jepa/vjepa_vit-b-16.pth.tar -O checkpoints/vjepa_vit-b-16.pt"
    echo "    2. ViT-L/16: wget https://dl.fbaipublicfiles.com/jepa/vjepa_vit-l-16.pth.tar -O checkpoints/vjepa_vit-l-16.pt"
    echo "    3. ViT-H/16: wget https://dl.fbaipublicfiles.com/jepa/vjepa_vit-h-16.pth.tar -O checkpoints/vjepa_vit-h-16.pt"
    echo ""
    echo "  Set PRETRAIN_CHECKPOINT environment variable to your checkpoint path."
    exit 1
else
    echo "  ✓ Pre-trained checkpoint found"
fi
echo ""

# Step 3: Check VPT data
echo "Step 3: Checking VPT data..."
VPT_SAMPLE=$(ls $VPT_DATA_PATH 2>/dev/null | head -1)
if [ -z "$VPT_SAMPLE" ]; then
    echo "  ⚠ VPT data not found at: $VPT_DATA_PATH"
    echo ""
    echo "  Please download VPT dataset and set VPT_DATA_PATH environment variable."
    echo "  Example: export VPT_DATA_PATH=/path/to/vpt/shard-*.tar"
    exit 1
else
    echo "  ✓ VPT data found (sample: $VPT_SAMPLE)"
fi
echo ""

# Step 4: Fit action scaler
echo "Step 4: Fitting action scaler..."
if [ ! -f "$SCALER_PATH" ]; then
    echo "  Running optimized action scaler fitting..."
    echo "  (Using 100K samples - takes ~10 minutes)"

    # Use optimized version with sensible defaults
    python action_scaler_fit_optimized.py \
        --data_path "$VPT_DATA_PATH" \
        --output "$SCALER_PATH" \
        --max_samples 100000 \
        --max_shards 20 \
        --batch_size 128 \
        --num_workers 8

    if [ -f "$SCALER_PATH" ]; then
        echo "  ✓ Action scaler fitted successfully"
    else
        echo "  ⚠ Action scaler fitting failed"
        exit 1
    fi
else
    echo "  ✓ Action scaler already exists"
    echo "  (Delete $SCALER_PATH to refit)"
fi
echo ""

# Step 5: Update config file
echo "Step 5: Updating config file..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "  ⚠ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create a temporary config with updated paths
TEMP_CONFIG=$(mktemp)
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Update pretrain checkpoint path
sed -i "s|pretrain_checkpoint: .*|pretrain_checkpoint: $PRETRAIN_CHECKPOINT|" "$TEMP_CONFIG"

# Update dataset path (handle both single and multi-dataset configs)
if grep -q "datasets:" "$TEMP_CONFIG"; then
    # Multi-dataset format
    sed -i "/datasets:/,/dataset_fpcs:/{s|- .*\.tar|- $VPT_DATA_PATH|}" "$TEMP_CONFIG"
else
    # Single dataset format
    sed -i "s|data_path: .*|data_path: $VPT_DATA_PATH|" "$TEMP_CONFIG"
fi

echo "  ✓ Config updated with your paths"
echo ""

# Step 6: Launch training
echo "Step 6: Launching training..."
echo "========================================="
echo ""

cd vjepa2

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running on single GPU..."
    python app/main.py --fname "$TEMP_CONFIG" --device cuda:0
else
    echo "Running on $NUM_GPUS GPUs (Distributed Data Parallel)..."
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        app/main_distributed.py \
        --fname "$TEMP_CONFIG"
fi

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
