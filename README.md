# craft-jepa: V-JEPA Fine-Tuning on VPT Dataset

Fine-tune V-JEPA (Vision-based Joint Embedding Predictive Architecture) on the VPT (Minecraft) dataset using the action-conditioned approach, following Meta's V-JEPA-AC methodology originally developed for the DROID robotic dataset.

## What is This?

This repository implements **V-JEPA-AC** (Action-Conditioned V-JEPA) for Minecraft:
- **Pre-trained V-JEPA encoder** (frozen) provides rich visual representations
- **Action-conditioned predictor** (trained) learns to predict future frame representations given actions
- **Result**: A powerful visual encoder that understands how actions (keyboard, mouse, camera) affect the Minecraft world

## Quick Start

### Prerequisites
1. Pre-trained V-JEPA checkpoint (ViT-L/16 recommended)
2. VPT dataset in WebDataset format (.tar shards with .mp4 + .jsonl)
3. GPU with 24GB+ VRAM (for ViT-L/16, batch_size=8)

### Installation
```bash
pip install -r requirements.txt
```

### Training in 3 Steps

**1. Fit Action Scaler:**
```bash
# Edit action_scaler_fit.py to set your VPT data path
python action_scaler_fit.py
```

**2. Update Config:**
```bash
# Edit vjepa2/configs/train/vitl16/minecraft-256px-8f.yaml
# Set pretrain_checkpoint and VPT data path
```

**3. Launch Training:**
```bash
# Single GPU
cd vjepa2
python app/main.py --fname configs/train/vitl16/minecraft-256px-8f.yaml

# Multi-GPU (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 \
  app/main_distributed.py --fname configs/train/vitl16/minecraft-256px-8f.yaml
```

**Or use the automated script:**
```bash
export PRETRAIN_CHECKPOINT=/path/to/vjepa_vit-l-16.pt
export VPT_DATA_PATH=/path/to/vpt-shards-*.tar
./train_vjepa_vpt.sh
```

## Documentation

- **[FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md)** - Complete step-by-step guide with troubleshooting
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture explanation and data flow diagrams

## Repository Structure

```
craft-jepa/
├── action_scaler_fit.py          # Fit StandardScaler for action normalization
├── train_vjepa_vpt.sh            # Automated training script
├── FINE_TUNING_GUIDE.md          # Complete fine-tuning guide
├── ARCHITECTURE.md               # Architecture documentation
└── vjepa2/                       # Main codebase
    ├── app/
    │   ├── vjepa_minecraft/      # VPT training pipeline
    │   │   ├── train.py          # Action-conditioned training loop
    │   │   ├── vpt_dataset.py    # WebDataset VPT loader
    │   │   └── utils.py          # VPT-specific utilities
    │   ├── vjepa_droid/          # DROID training pipeline (reference)
    │   └── vjepa/                # Self-supervised pre-training
    ├── configs/
    │   └── train/
    │       ├── vitl16/minecraft-256px-8f.yaml  # ViT-L/16 config
    │       └── vitg16/minecraft-256px-8f.yaml  # ViT-G/16 config
    └── src/
        ├── models/
        │   ├── vision_transformer.py      # ViT encoder
        │   ├── ac_predictor.py            # Action-conditioned predictor
        │   └── predictor.py               # Standard predictor
        └── datasets/
            └── ...
```

## Key Features

✅ **Action-Conditioned Prediction**: Learns to predict future frames given actions (mouse, keyboard, camera)
✅ **WebDataset Support**: Efficient streaming from tar shards
✅ **Distributed Training**: Multi-GPU support with DDP
✅ **Mixed Precision**: bfloat16/float16 for faster training
✅ **Action Normalization**: StandardScaler for continuous actions
✅ **Hotbar Injection**: Automatically injects hotbar key presses
✅ **Frame-Causal Attention**: Prevents looking into the future
✅ **Teacher Forcing + Auto-Regressive**: Stable multi-step prediction learning

## Model Configurations

| Model | Embed Dim | Layers | Params | Recommended Batch Size | GPU Memory |
|-------|-----------|--------|--------|------------------------|------------|
| ViT-B/16 | 768 | 12 | ~86M | 16 | 16GB |
| ViT-L/16 | 1024 | 24 | ~304M | 8 | 24GB |
| ViT-H/16 | 1280 | 32 | ~632M | 4 | 40GB |
| ViT-G/16 | 1408 | 40 | ~1B | 2 | 48GB |

## Action Space

**Current (13D):**
- 4D Continuous: `mouse_dx`, `mouse_dy`, `yaw_diff`, `pitch_diff` (StandardScaler normalized)
- 9D Discrete: Hotbar keys (1-9)

**Can be extended to 36D** (see `ARCHITECTURE.md` for full breakdown)

## Citation

If you use this code, please cite the original V-JEPA and VPT papers:

```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Rabbat, Michael and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2404.08471},
  year={2024}
}

@article{baker2022video,
  title={Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos},
  author={Baker, Bowen and Akkaya, Ilge and Zhokhov, Peter and Huizinga, Joost and Tang, Jie and Ecoffet, Adrien and Houghton, Brandon and Sampedro, Raul and Clune, Jeff},
  journal={arXiv preprint arXiv:2206.11795},
  year={2022}
}
```

## License

This project is licensed under the same terms as the original V-JEPA codebase from Meta AI.

## Acknowledgments

- Meta AI for the V-JEPA pre-training code and models
- OpenAI for the VPT dataset
- DROID team for the action-conditioned fine-tuning approach
