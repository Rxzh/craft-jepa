# V-JEPA-AC VPT Fine-Tuning - Quick Reference Card

## TL;DR - What You Need

1. **Pre-trained V-JEPA checkpoint** → Download ViT-L/16 from Meta AI
2. **VPT dataset** → WebDataset tar shards (video.mp4 + actions.jsonl)
3. **GPU** → 24GB VRAM minimum (for ViT-L/16)

## Three Commands to Start Training

```bash
# 1. Fit action scaler (one-time setup)
python action_scaler_fit.py

# 2. Update paths in config
vim vjepa2/configs/train/vitl16/minecraft-256px-8f.yaml

# 3. Train!
cd vjepa2 && python app/main.py --fname configs/train/vitl16/minecraft-256px-8f.yaml
```

## What is V-JEPA-AC?

**Standard V-JEPA (Pre-training):**
```
Video → Encoder → Predictor → Predict masked patches
```

**V-JEPA-AC (Fine-tuning):**
```
Video + Actions → Frozen Encoder → AC Predictor → Predict future frames
                                          ↑
                                    (learns action effects)
```

## Key Differences from Standard V-JEPA

| Aspect | V-JEPA | V-JEPA-AC (VPT) |
|--------|--------|-----------------|
| Encoder | Trained | **Frozen** |
| Predictor | Standard | **Action-Conditioned** |
| Input | Video only | Video + Actions + States |
| Task | Predict masked patches | Predict future given actions |
| Dataset | Any video | VPT (with actions) |

## Architecture Flow

```
VPT Data (.tar)
    ↓
[16 frames, actions, states]
    ↓
Frozen Encoder (ViT-L/16) → representations h
    ↓
AC Predictor(h, actions, states) → predictions z
    ↓
Loss = MSE(z, h_target)
```

## Config Parameters You'll Modify

```yaml
# vjepa2/configs/train/vitl16/minecraft-256px-8f.yaml

meta:
  pretrain_checkpoint: /path/to/vjepa_vit-l-16.pt  # ← UPDATE THIS

data:
  datasets:
    - /path/to/VPT/shard-*.tar                     # ← UPDATE THIS
  batch_size: 8                                    # ← Reduce if OOM
```

## Training Progress Indicators

**Good Signs:**
- `jloss` (teacher forcing) decreases steadily
- `sloss` (auto-regressive) decreases (slower than jloss)
- GPU utilization > 90%

**Bad Signs:**
- Loss stays flat after 50 epochs → Check learning rate
- `nan` loss → Mixed precision issue, try float32
- Very slow (<1 it/s) → Reduce num_workers or batch_size

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: vpt_action_scaler.pkl` | Run `python action_scaler_fit.py` |
| `CUDA out of memory` | Reduce `batch_size` (8→4) or use ViT-B instead of ViT-L |
| `Checkpoint not found` | Update `pretrain_checkpoint` path in config |
| Loss not decreasing | Check action scaler fitted correctly, verify data paths |

## Training Time Estimates

| Model | GPUs | Batch/GPU | Time/Epoch | Total (315 epochs) |
|-------|------|-----------|------------|-------------------|
| ViT-B/16 | 1 | 16 | ~30 min | ~6.5 days |
| ViT-L/16 | 1 | 8 | ~1 hour | ~13 days |
| ViT-L/16 | 4 | 8 | ~15 min | ~3.3 days |
| ViT-G/16 | 8 | 4 | ~20 min | ~4.4 days |

## Output Files

```
/your_folder/vpt_jepa/vit_l_16-256px-16f/
├── train.csv                      # Loss metrics (jloss, sloss, epoch, ...)
├── checkpoint-epoch001.pth.tar    # Model checkpoints
├── checkpoint-epoch025.pth.tar
└── checkpoint-latest.pth.tar      # Symlink to most recent
```

## After Fine-Tuning: What Can You Do?

1. **Inverse Dynamics Model**: Given 2 frames → predict action taken
2. **Action Anticipation**: Given current frame → predict next action
3. **RL Policy**: Use encoder as visual backbone
4. **Video Classification**: Classify Minecraft tasks/biomes

## Action Space (Current 13D)

```
4D Continuous (normalized):
  - mouse_dx    : horizontal mouse movement
  - mouse_dy    : vertical mouse movement
  - yaw_diff    : camera rotation change
  - pitch_diff  : camera pitch change

9D Discrete (binary):
  - hotbar_key_1 to hotbar_key_9
```

## Hyperparameters (Default)

```yaml
frames_per_clip: 16      # Sample 16 frames per video
fps: 5                   # Target 5 FPS (downsample from 20)
tubelet_size: 2          # 16 frames → 8 tubelets
crop_size: 256           # Spatial resolution
action_embed_dim: 13     # VPT action dimensions

epochs: 315              # Total training epochs
lr: 0.000425             # Peak learning rate
warmup: 15               # Warmup epochs
auto_steps: 2            # Auto-regressive rollout steps
```

## Monitoring Training

```bash
# Watch loss curve
tail -f /your_folder/vpt_jepa/vit_l_16-256px-16f/train.csv

# Check GPU usage
nvidia-smi -l 1

# TensorBoard (if enabled)
tensorboard --logdir /your_folder/vpt_jepa/vit_l_16-256px-16f
```

## Download Pre-trained Checkpoints

```bash
# ViT-B/16 (smallest, fastest)
wget https://dl.fbaipublicfiles.com/jepa/vjepa_vit-b-16.pth.tar

# ViT-L/16 (recommended)
wget https://dl.fbaipublicfiles.com/jepa/vjepa_vit-l-16.pth.tar

# ViT-H/16 (larger)
wget https://dl.fbaipublicfiles.com/jepa/vjepa_vit-h-16.pth.tar
```

## Further Reading

- **FINE_TUNING_GUIDE.md** - Complete step-by-step tutorial
- **ARCHITECTURE.md** - Detailed architecture diagrams
- **README.md** - Project overview

## Getting Help

If something doesn't work:
1. Check `FINE_TUNING_GUIDE.md` troubleshooting section
2. Verify your VPT data format matches expected structure
3. Check GPU memory with `nvidia-smi`
4. Review training logs for errors

---

**Remember**: The encoder is **frozen** during fine-tuning. Only the action-conditioned predictor is trained. This is the key difference from standard V-JEPA pre-training!
