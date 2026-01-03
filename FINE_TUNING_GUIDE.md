# V-JEPA Fine-Tuning on VPT Dataset - Complete Guide

This guide walks you through fine-tuning V-JEPA on the VPT (Minecraft) dataset, following the same approach Meta researchers used for fine-tuning on the Droid dataset to create V-JEPA-AC.

## Overview

**V-JEPA-AC (Action-Conditioned)** extends the pre-trained V-JEPA model by:
1. **Freezing the pre-trained encoder** (from self-supervised V-JEPA)
2. **Training a new action-conditioned predictor** that:
   - Takes actions (mouse, keyboard, camera) and states (yaw, pitch) as input
   - Predicts future frame representations conditioned on these actions
   - Uses frame-causal attention (can't see the future)
3. **Learning through two objectives**:
   - **Teacher Forcing**: Predict next frame given ground-truth actions
   - **Auto-Regressive Rollout**: Multi-step predictions using own predictions

## Architecture Comparison

| Component | V-JEPA (Pre-training) | V-JEPA-AC (VPT Fine-tuning) |
|-----------|----------------------|----------------------------|
| **Encoder** | Trained | **Frozen** (from pre-training) |
| **Predictor** | Standard predictor | **Action-Conditioned predictor** |
| **Input** | Masked video frames | Video frames + Actions + States |
| **Output** | Masked frame predictions | Action-conditioned future predictions |
| **Dataset** | Any video (unlabeled) | VPT (actions + video) |
| **Loss** | MSE on masked patches | MSE on action-conditioned predictions |

---

## Prerequisites

### 1. Pre-trained V-JEPA Checkpoint

You need a pre-trained V-JEPA model. Download from Meta's official release or train your own:

```bash
# Example: Download ViT-L/16 checkpoint
wget https://dl.fbaipublicfiles.com/jepa/vjepa_vit-l-16.pth.tar -O checkpoints/vjepa_vit-l-16.pt
```

**Available Models:**
- `vit_base_16` (ViT-B/16): 768-dim, 12 layers
- `vit_large_16` (ViT-L/16): 1024-dim, 24 layers ⭐ **Recommended**
- `vit_giant_xformers` (ViT-G/16): 1408-dim, 40 layers
- `vit_huge` (ViT-H/16): 1280-dim, 32 layers

### 2. VPT Dataset (WebDataset Format)

The VPT dataset should be in WebDataset format (`.tar` shards containing `.mp4` + `.jsonl` files).

**Expected Structure:**
```
VPT/
├── shard-000000.tar  # Contains: video_000.mp4, video_000.jsonl, video_001.mp4, ...
├── shard-000001.tar
├── shard-000002.tar
...
└── shard-000123.tar
```

**JSONL Format (per video):**
```jsonl
{"timestamp": 0.0, "mouse": {"dx": 0, "dy": 0}, "keyboard": {"keys": []}, "camera": [0.0, 0.0], "hotbar": 0}
{"timestamp": 0.2, "mouse": {"dx": 5, "dy": -2}, "keyboard": {"keys": ["key.keyboard.w"]}, "camera": [0.05, 0.0], "hotbar": 0}
...
```

**Where to get VPT data:**
- Official VPT repository: https://github.com/openai/Video-Pre-Training
- Hugging Face: https://huggingface.co/datasets/minecraft-vpt

---

## Step-by-Step Fine-Tuning Process

### Step 1: Install Dependencies

```bash
cd /home/user/craft-jepa
pip install -r requirements.txt
```

### Step 2: Fit Action Scaler

The action scaler normalizes continuous actions (mouse movements, camera changes) to zero mean and unit variance.

**Edit `action_scaler_fit.py`:**
```python
DATA_PATH = "VPT/shard-0000*.tar"  # Update to your VPT data path
SCALER_PATH = "vpt_action_scaler.pkl"
```

**Run:**
```bash
python action_scaler_fit.py
```

**Output:** `vpt_action_scaler.pkl` (will be loaded during training)

**What it does:**
- Processes a subset of VPT data
- Computes mean and std for 4 continuous action dimensions:
  - `mouse_dx` (horizontal mouse movement)
  - `mouse_dy` (vertical mouse movement)
  - `yaw_diff` (camera rotation change)
  - `pitch_diff` (camera pitch change)
- Saves StandardScaler to disk

### Step 3: Configure Training

**Edit `vjepa2/configs/train/vitl16/minecraft-256px-8f.yaml`:**

```yaml
# Update these paths:
meta:
  pretrain_checkpoint: /path/to/vjepa_vit-l-16.pt  # Your pre-trained checkpoint

data:
  datasets:
    - /path/to/VPT/shard-{000000..000123}.tar  # Your VPT shards
```

**Key Configuration Parameters:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `app` | `vpt_minecraft` | Use VPT training pipeline |
| `model.model_name` | `vit_l_16` | ViT-L/16 architecture |
| `model.action_embed_dim` | `13` | VPT action dimensions (4 continuous + 9 hotbar keys) |
| `model.use_extrinsics` | `false` | VPT has no gripper/extrinsics (unlike Droid) |
| `model.pred_is_frame_causal` | `true` | Enable causal masking (can't see future) |
| `data.frames_per_clip` | `16` | Sample 16 frames per video |
| `data.fps` | `5` | Target 5 FPS |
| `data.tubelet_size` | `2` | Temporal stride (16 frames → 8 tubelets) |
| `optimization.epochs` | `315` | Total training epochs |
| `optimization.lr` | `0.000425` | Peak learning rate |
| `loss.auto_steps` | `2` | Auto-regressive rollout steps |

**Action Dimension Breakdown (13D):**
```
4D Continuous (scaled):
  - mouse_dx_sum
  - mouse_dy_sum
  - yaw_diff
  - pitch_diff

9D Discrete (one-hot):
  - hotbar keys (1-9)

Total: 13 dimensions
```

**State Dimension (2D):**
```
- yaw (camera rotation)
- pitch (camera angle)
```

### Step 4: Launch Training

#### Single GPU:
```bash
cd /home/user/craft-jepa/vjepa2
python app/main.py \
  --fname configs/train/vitl16/minecraft-256px-8f.yaml \
  --device cuda:0
```

#### Multi-GPU (Distributed Data Parallel):
```bash
cd /home/user/craft-jepa/vjepa2

# 4 GPUs on single node
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=1 \
  app/main_distributed.py \
  --fname configs/train/vitl16/minecraft-256px-8f.yaml
```

#### SLURM Cluster:
```bash
# Update config with your SLURM settings
sbatch --nodes=4 --tasks-per-node=8 --gpus-per-task=1 \
  app/main_distributed.py --fname configs/train/vitl16/minecraft-256px-8f.yaml
```

### Step 5: Monitor Training

Training logs are saved to the folder specified in config:

```bash
tail -f /your_folder/vpt_jepa/vit_l_16-256px-16f/train.csv
```

**Metrics to monitor:**
- **jloss** (teacher forcing loss): MSE between predicted and target representations
- **sloss** (auto-regressive loss): MSE for multi-step rollouts
- **total_loss** = jloss + sloss

**Expected behavior:**
- Losses should decrease steadily
- jloss typically lower than sloss (teacher forcing is easier)
- Convergence after 100-200 epochs

### Step 6: Checkpointing

Checkpoints are saved every `save_every_freq` epochs (default: 25):

```
/your_folder/vpt_jepa/vit_l_16-256px-16f/
├── checkpoint-epoch001.pth.tar
├── checkpoint-epoch025.pth.tar
├── checkpoint-epoch050.pth.tar
...
└── checkpoint-latest.pth.tar  (symlink to most recent)
```

**Checkpoint contents:**
- `encoder`: Frozen context encoder (unchanged during training)
- `predictor`: Action-conditioned predictor (trained)
- `target_encoder`: EMA target encoder
- `optimizer`: Optimizer state
- `scaler`: Mixed precision scaler
- `epoch`, `batch`, etc.

---

## Understanding the Training Process

### Data Pipeline

1. **Video Loading** (`vpt_dataset.py`):
   - Load `.mp4` video at native FPS (usually 20 FPS)
   - Load `.jsonl` action/state sequence
   - Downsample to target FPS (5 FPS)
   - Sample 16 consecutive frames

2. **Tubelet Formation**:
   ```
   16 frames → 8 tubelets (frameskip=2)
   Tubelet i: frames [0,1], [2,3], [4,5], ..., [14,15]
   ```

3. **Action Aggregation** (per tubelet):
   ```python
   # Continuous actions: SUM over frames in tubelet
   mouse_dx_sum = sum(mouse_dx for frames in [i, i+1])
   yaw_diff = circular_diff(yaw[i+1], yaw[i])

   # Discrete actions: LOGICAL OR over frames
   keyboard_keys = union(keys pressed in [i, i+1])
   hotbar = hotbar_at_frame[i]  # Slot at tubelet start
   ```

4. **Hotbar Key Injection**:
   - When hotbar slot changes (e.g., 0 → 2), automatically inject key press `key.keyboard.3`
   - Ensures predictor learns hotbar changes

5. **Action Normalization**:
   - Continuous actions scaled using pre-fitted StandardScaler
   - Discrete actions remain binary

### Forward Pass

```python
# 1. Encode video frames (frozen encoder)
with torch.no_grad():
    h = target_encoder(video_frames)  # (B, T, D) = (8, 128, 1024)
    # 128 tokens = 8 tubelets × 16 spatial patches

# 2. Action-Conditioned Prediction
z = predictor(
    context=h[:-tokens_per_frame],  # All but last frame tokens
    actions=actions,                 # (B, T-1, 13)
    states=states[:-1],              # (B, T-1, 2)
    mask=None
)

# 3. Compute Loss
loss_tf = MSE(z_teacher_forcing, h_target)
loss_ar = MSE(z_autoregressive, h_target)
loss = loss_tf + loss_ar
```

### Training Loop

Each iteration:
1. **Teacher Forcing** (1 step):
   - Predict frame `t+1` given frames `[:t]` + actions `[:t]`
   - Ground truth actions used

2. **Auto-Regressive Rollout** (2+ steps):
   - Predict frame `t+1`, then use `t+1` prediction to predict `t+2`, etc.
   - Learns longer-term dynamics

3. **Gradient Update**:
   - Only predictor parameters updated
   - Encoder frozen (preserves pre-trained representations)

---

## Troubleshooting

### Issue: `FileNotFoundError: vpt_action_scaler.pkl`
**Solution:** Run `python action_scaler_fit.py` first

### Issue: `CUDA out of memory`
**Solutions:**
- Reduce `batch_size` (e.g., 8 → 4)
- Enable `use_activation_checkpointing: true`
- Use smaller model (ViT-L → ViT-B)
- Reduce `frames_per_clip` (16 → 8)

### Issue: Loss not decreasing
**Check:**
- Action scaler fitted correctly?
- Pre-trained checkpoint loaded? (check logs for "Loaded pretrained encoder")
- Learning rate too high/low? (try 1e-4 to 5e-4)
- Data augmentation too aggressive? (disable `motion_shift`, `horizontal_flip`)

### Issue: Very slow training
**Solutions:**
- Increase `num_workers` (default: 12)
- Use faster storage (SSD vs. HDD)
- Pre-decode videos to frames
- Use `torch.compile()` (set `compile_model: true`)

---

## Next Steps After Fine-Tuning

### 1. Downstream Task Evaluation
The fine-tuned encoder can be used for:
- **Action Anticipation**: Predict future actions from video
- **Inverse Dynamics**: Given two frames, predict action
- **Video Classification**: Minecraft task/biome classification
- **Reinforcement Learning**: Use as visual encoder for policy

### 2. Inverse Model Training
Train a lightweight model on top of frozen encoder:
```python
# Encoder outputs
h_t = encoder(frame_t)
h_t1 = encoder(frame_{t+1})

# Inverse model
predicted_action = inverse_model(torch.cat([h_t, h_t1], dim=-1))
loss = MSE(predicted_action, ground_truth_action)
```

### 3. Fine-tune on Specific Tasks
Use the action-conditioned model as initialization for task-specific learning (e.g., "mine diamond", "build house").

---

## Advanced Configuration

### Larger Models (ViT-G/16)
```yaml
# vjepa2/configs/train/vitg16/minecraft-256px-8f.yaml
model:
  model_name: vit_giant_xformers
  pred_embed_dim: 1408
  pred_depth: 24

# Requires more GPU memory
data:
  batch_size: 4  # Reduce from 8
```

### Extended Action Space (Full 36D)
Modify `vpt_dataset.py` to output all action dimensions instead of 13D:
```python
# Include all keyboard keys, mouse buttons, etc.
action_embed_dim: 36  # In config
```

### Multi-Dataset Training
```yaml
data:
  datasets:
    - /path/to/vpt_dataset1/shard-*.tar
    - /path/to/vpt_dataset2/shard-*.tar
  dataset_fpcs:
    - 16
    - 16
```

---

## References

- **V-JEPA Paper**: [Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- **V-JEPA GitHub**: https://github.com/facebookresearch/jepa
- **VPT Paper**: [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795)
- **VPT GitHub**: https://github.com/openai/Video-Pre-Training

---

## Summary

Fine-tuning V-JEPA on VPT follows the V-JEPA-AC methodology:

1. **Start with pre-trained V-JEPA** (encoder frozen)
2. **Train action-conditioned predictor** (13D actions, 2D states)
3. **Use teacher forcing + auto-regressive rollouts**
4. **Result**: Encoder that understands action-video relationships

This creates a powerful visual representation model for Minecraft that understands the link between actions (keyboard, mouse) and resulting visual changes.
