# Action Scaler Fitting - Optimization Guide

## Why Action Normalization is Critical

The V-JEPA-AC predictor takes **continuous actions** (mouse movements, camera changes) as input. Without normalization:

1. **Scale mismatch**: Mouse movements (~1-100 pixels) vs camera changes (~0.01-0.1 radians)
2. **Training instability**: Large gradients from unnormalized inputs
3. **Poor convergence**: Model struggles to learn meaningful relationships

**Solution**: StandardScaler (zero mean, unit variance) for the 4 continuous action dimensions:
- `mouse_dx`, `mouse_dy`, `yaw_diff`, `pitch_diff`

---

## Original vs Optimized Implementation

### Original (`action_scaler_fit.py`)

**Pros:**
✅ GPU-accelerated sufficient statistics
✅ High numerical precision (float64)
✅ Correct StandardScaler setup

**Cons:**
❌ Processes ALL data in matched shards (could take hours)
❌ No early stopping
❌ Hard-coded shard pattern (`shard-0000*.tar`)
❌ No convergence detection
❌ Overkill for statistical estimation

**Time estimate** (full VPT dataset ~500 shards):
- Single GPU: **4-8 hours** (may exceed your cluster limit!)

### Optimized (`action_scaler_fit_optimized.py`)

**Improvements:**
✅ **Sample budget**: Default 100K actions (configurable)
✅ **Early stopping**: Stops when budget reached OR statistics converge
✅ **Shard sampling**: Use subset of shards uniformly sampled across dataset
✅ **Convergence detection**: Monitors mean/std stability
✅ **Progress tracking**: tqdm progress bar with ETA
✅ **Verification**: Auto-tests scaler after fitting
✅ **CLI arguments**: Flexible configuration

**Time estimate** (100K samples):
- Single GPU: **5-15 minutes** ⚡

**Time estimate** (1M samples, for paranoia):
- Single GPU: **30-60 minutes**

---

## Statistical Justification

### How Many Samples Do You Need?

**Central Limit Theorem** tells us the standard error of the mean decreases as:

```
SE = σ / sqrt(n)
```

| Samples | Standard Error (relative) | Recommendation |
|---------|---------------------------|----------------|
| 1,000 | ~3.2% | ❌ Too few |
| 10,000 | ~1.0% | ⚠️ Minimum |
| 100,000 | ~0.32% | ✅ **Recommended** |
| 1,000,000 | ~0.1% | ✅ Overkill (but fine) |
| 10,000,000 | ~0.03% | ❌ Waste of time |

**For V-JEPA-AC training:**
- 100K samples gives **sub-1% error** on mean/std estimates
- This is **more than sufficient** for stable training
- Going beyond 1M samples has **diminishing returns**

### Real-World Example

VPT dataset structure:
- Each video clip: 16 frames → 8 tubelets → **7 actions**
- Batch size: 128 clips → **896 actions/batch**
- To reach 100K samples: **~112 batches**

At 128 batch size with 8 workers:
- **~10 minutes** on modern GPU

---

## Usage Examples

### Quick Start (Recommended)

```bash
# Use default settings (100K samples)
python action_scaler_fit_optimized.py \
  --data_path "VPT/shard-*.tar"
```

**Output:** `vpt_action_scaler.pkl` (ready to use)

### Conservative (More Samples)

```bash
# Use 500K samples for extra confidence
python action_scaler_fit_optimized.py \
  --data_path "VPT/shard-*.tar" \
  --max_samples 500000
```

### Fast Prototype (Limited Shards)

```bash
# Use only 10 shards (fastest, still statistically valid)
python action_scaler_fit_optimized.py \
  --data_path "VPT/shard-*.tar" \
  --max_shards 10 \
  --max_samples 100000
```

### Full Control

```bash
python action_scaler_fit_optimized.py \
  --data_path "VPT/shard-*.tar" \
  --output my_scaler.pkl \
  --max_samples 200000 \
  --max_shards 20 \
  --batch_size 256 \
  --num_workers 12 \
  --no_convergence_check  # Disable early stopping
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `VPT/shard-*.tar` | Glob pattern for VPT shards |
| `--output` | `vpt_action_scaler.pkl` | Output path for scaler |
| `--max_samples` | `100000` | Maximum action samples to collect |
| `--max_shards` | `None` | Limit number of shards (None = all) |
| `--batch_size` | `128` | Dataloader batch size |
| `--num_workers` | `8` | Dataloader workers |
| `--no_convergence_check` | `False` | Disable convergence early stopping |

---

## Understanding the Output

### Example Output

```
============================================================
SCALER FITTING COMPLETE
============================================================
Saved to: vpt_action_scaler.pkl
Total samples: 100,352
Converged: True

Action Statistics (Continuous Actions):
  mouse_dx  - Mean:    0.125, Std:   12.453
  mouse_dy  - Mean:   -0.087, Std:    8.932
  yaw_diff  - Mean:    0.003, Std:    0.142
  pitch_diff- Mean:   -0.001, Std:    0.089
============================================================
```

### What to Look For

**✅ Good Signs:**
- `Converged: True` → Statistics are stable
- `mouse_dx/dy` std in range 5-50 → Reasonable mouse movements
- `yaw/pitch_diff` std in range 0.01-0.5 → Reasonable camera changes
- `Total samples > 10,000` → Statistically significant

**⚠️ Warning Signs:**
- `std` close to 0 → Data might be corrupted (no movement)
- `std` extremely high (>1000) → Outliers or incorrect units
- `Converged: False` → Increase `max_samples`

---

## Integration with Training

### Update Training Config

Your config already references the scaler, no changes needed!

```yaml
# vjepa2/configs/train/vitl16/minecraft-256px-8f.yaml
# The scaler is loaded automatically by init_vpt_dataloader()
```

### How It's Used During Training

```python
# In vpt_dataset.py (automatically handled)

# 1. Load scaler
scaler = joblib.load("vpt_action_scaler.pkl")

# 2. Pass to dataloader
dataloader = init_vpt_dataloader(
    data_path="VPT/shard-*.tar",
    action_scaler=scaler,  # ← Applies normalization
    ...
)

# 3. Inside ProcessVPT
continuous_actions = np.stack([mouse_dx, mouse_dy, yaw_diff, pitch_diff], axis=1)
if self.scaler is not None:
    continuous_actions = self.scaler.transform(continuous_actions)  # ← Normalize
```

**Result**: All continuous actions fed to the predictor are **zero-mean, unit-variance**.

---

## Comparison: Time & Resource Usage

### Scenario: 500-shard VPT Dataset

| Method | Shards Used | Samples | GPU Time | CPU Workers | Total Time |
|--------|-------------|---------|----------|-------------|------------|
| Original (all data) | 500 | ~50M | 6-8 hrs | 4 | **6-8 hrs** |
| Optimized (default) | 500 | 100K | 10 min | 8 | **10 min** |
| Optimized (10 shards) | 10 | 100K | 5 min | 8 | **5 min** |
| Optimized (paranoid) | 500 | 1M | 45 min | 8 | **45 min** |

**Recommendation for 8-hour cluster limit:**
- Use optimized version with **100K samples** (default)
- Fits comfortably in **10-15 minutes**
- Leaves **7h 45m** for actual training

---

## Validation: Are 100K Samples Enough?

### Experiment (You Can Try This)

Fit scaler with different sample sizes and compare:

```bash
# 10K samples
python action_scaler_fit_optimized.py --max_samples 10000 --output scaler_10k.pkl

# 100K samples
python action_scaler_fit_optimized.py --max_samples 100000 --output scaler_100k.pkl

# 1M samples
python action_scaler_fit_optimized.py --max_samples 1000000 --output scaler_1m.pkl

# Compare
python -c "
import joblib
s10k = joblib.load('scaler_10k.pkl')
s100k = joblib.load('scaler_100k.pkl')
s1m = joblib.load('scaler_1m.pkl')

print('Mean differences (10K vs 100K):', abs(s10k.mean_ - s100k.mean_))
print('Mean differences (100K vs 1M):', abs(s100k.mean_ - s1m.mean_))
print('Std differences (10K vs 100K):', abs(s10k.scale_ - s100k.scale_))
print('Std differences (100K vs 1M):', abs(s100k.scale_ - s1m.scale_))
"
```

**Expected result**: 100K→1M has **negligible difference** (<0.1%)

---

## Troubleshooting

### Issue: "No shards found"

**Cause**: Incorrect data path

**Solution**:
```bash
# Check your actual shard pattern
ls VPT/shard-*.tar | head -5

# Update data_path accordingly
python action_scaler_fit_optimized.py --data_path "VPT/shard-{000000..000499}.tar"
```

### Issue: Very slow (>30 min for 100K samples)

**Possible causes:**
1. Slow storage (network drive, HDD)
2. Too few workers
3. CPU bottleneck in video decoding

**Solutions**:
```bash
# Increase workers (if you have CPU headroom)
--num_workers 16

# Increase batch size (if you have GPU memory)
--batch_size 256

# Use fewer, larger shards
--max_shards 5
```

### Issue: CUDA out of memory

**Cause**: Large batch size

**Solution**:
```bash
# Reduce batch size
--batch_size 64

# Or use CPU (slower but works)
# Edit script: device = torch.device("cpu")
```

### Issue: Statistics look weird

**Examples:**
- `std` is 0 → No variation in data (corrupted?)
- `mean` is huge → Wrong units or data format

**Debug:**
```python
# Load a batch and inspect manually
from vjepa2.app.vjepa_minecraft.vpt_dataset import init_vpt_dataloader

loader, _ = init_vpt_dataloader(
    data_path="VPT/shard-000000.tar",
    batch_size=1,
    action_scaler=None,
    num_workers=0
)

batch = next(iter(loader))
actions = batch['actions']
print("Actions shape:", actions.shape)
print("Continuous actions (first 4 dims):")
print("  Min:", actions[..., :4].min(dim=0).values.min(dim=0).values)
print("  Max:", actions[..., :4].max(dim=0).values.max(dim=0).values)
print("  Mean:", actions[..., :4].mean(dim=(0, 1)))
print("  Std:", actions[..., :4].std(dim=(0, 1)))
```

---

## Summary & Recommendation

### For Your 8-Hour Cluster Runs:

**Use the optimized version:**
```bash
python action_scaler_fit_optimized.py \
  --data_path "VPT/shard-*.tar" \
  --max_samples 100000 \
  --max_shards 20
```

**Why:**
- ⚡ **Fast**: ~10 minutes (leaves 7h 50m for training)
- 📊 **Statistically valid**: 100K samples → <1% error
- 🎯 **Reliable**: Convergence detection + verification
- 💾 **Resource-efficient**: Uses subset of shards

**When to use more samples:**
- If you see training instability (rare)
- If you want extra confidence (use `--max_samples 500000`)
- If statistics don't converge (check data quality first)

**Bottom line:** 100K samples is the sweet spot between speed and accuracy.
