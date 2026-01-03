# V-JEPA Action-Conditioned Architecture for VPT

This document explains the architecture and data flow for fine-tuning V-JEPA on VPT.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    V-JEPA-AC Fine-Tuning Pipeline                │
└─────────────────────────────────────────────────────────────────┘

                   VPT Dataset (WebDataset)
                   ┌──────────────────────┐
                   │  .mp4 (video)        │
                   │  .jsonl (actions)    │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  ProcessVPT          │
                   │  - Sample frames     │
                   │  - Aggregate actions │
                   │  - Normalize         │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │ Batch:               │
                   │  video: (B,T,C,H,W)  │
                   │  actions: (B,T-1,13) │
                   │  states: (B,T,2)     │
                   └──────────┬───────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌────────────────┐
│ Context       │    │ Target         │    │ Action/State   │
│ Encoder       │    │ Encoder        │    │ Data           │
│ (Frozen)      │    │ (Frozen EMA)   │    │                │
│               │    │                │    │                │
│ ViT-L/16      │    │ ViT-L/16       │    │                │
│ 1024-dim      │    │ 1024-dim       │    │                │
└───────┬───────┘    └────────┬───────┘    └────────┬───────┘
        │                     │                     │
        │                     │ Ground Truth        │
        │                     │ Targets (h)         │
        │            ┌────────▼───────┐             │
        │            │  (B×T, 1024)   │             │
        │            │  Detached      │             │
        │            └────────────────┘             │
        │                                           │
        └──────────────┬────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Action-Conditioned Predictor │
        │ (Trainable)                  │
        │                              │
        │ Input Sequence per frame:    │
        │  [action_emb, state_emb,     │
        │   ...frame_tokens...]        │
        │                              │
        │ Frame-Causal Attention:      │
        │  - Can't see future frames   │
        │  - 24 transformer blocks     │
        │                              │
        │ Output: predicted tokens     │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Two Training Modes:         │
        │                              │
        │  1. Teacher Forcing (1-step) │
        │     z_tf = pred(h[:t], a[:t])│
        │     loss_tf = MSE(z_tf, h[t])│
        │                              │
        │  2. Auto-Regressive Rollout  │
        │     z_1 = pred(h[:t], a[:t]) │
        │     z_2 = pred(z_1, a[:t+1]) │
        │     loss_ar = MSE(z_2, h[t+1])│
        └──────────────┬───────────────┘
                       │
                       ▼
                ┌─────────────┐
                │ Total Loss  │
                │ jloss+sloss │
                └─────────────┘
```

## Detailed Component Breakdown

### 1. Data Processing Pipeline

#### Input: WebDataset Tar Shards
```
shard-000000.tar:
  ├── minecraft_gameplay_001.mp4    (20 FPS, 1280×720)
  ├── minecraft_gameplay_001.jsonl  (20 lines/sec action logs)
  ├── minecraft_gameplay_002.mp4
  └── minecraft_gameplay_002.jsonl
```

#### ProcessVPT: Frame Sampling & Action Aggregation

**Step 1: Frame Sampling**
```python
# Input: 1000-frame video at 20 FPS (50 seconds)
# Target: 16 frames at 5 FPS

# Sample indices: [0, 4, 8, 12, ..., 60]  (every 4th frame)
# Results in: 16 frames spanning 3.2 seconds
```

**Step 2: Tubelet Formation**
```python
# 16 frames → 8 tubelets (frameskip=2)

Tubelet 0: [frame_0, frame_1]
Tubelet 1: [frame_2, frame_3]
...
Tubelet 7: [frame_14, frame_15]
```

**Step 3: Action Aggregation (per tubelet)**

For tubelet spanning frames `[i, i+frameskip]`:

**Continuous Actions (SUM):**
```python
mouse_dx_sum = sum([action[j].mouse.dx for j in range(i, i+frameskip)])
mouse_dy_sum = sum([action[j].mouse.dy for j in range(i, i+frameskip)])
yaw_diff = circular_diff(state[i+frameskip].yaw, state[i].yaw)
pitch_diff = state[i+frameskip].pitch - state[i].pitch
```

**Discrete Actions (LOGICAL OR):**
```python
keyboard_keys = set()
for j in range(i, i+frameskip):
    keyboard_keys.update(action[j].keyboard.keys)

hotbar = action[i].hotbar  # At tubelet start
```

**Hotbar Key Injection:**
```python
if action[i].hotbar != action[i-1].hotbar:
    # Hotbar changed from slot 0 to slot 2
    keyboard_keys.add(f"key.keyboard.3")  # Inject "3" key press
```

**Output Shape:**
```python
video_frames: (16, 3, 256, 256)   # T, C, H, W
actions:      (7, 36)              # T/frameskip - 1, D_action
states:       (8, 2)               # T/frameskip, [yaw, pitch]
```

### 2. Action Space Design

#### Original VPT Action Space (36D)

```
Dimension  | Type        | Description                | Range
-----------|-------------|----------------------------|---------
0-3        | Continuous  | mouse_dx, dy, yaw, pitch   | ℝ (scaled)
4-12       | Binary      | Hotbar keys (1-9)          | {0,1}
13-17      | Binary      | Movement (w,a,s,d,space)   | {0,1}
18-19      | Binary      | Special (q=drop, f=swap)   | {0,1}
20         | Binary      | Inventory (e)              | {0,1}
21         | Binary      | ESC                        | {0,1}
22-24      | Binary      | Modifiers (shift, ctrl)    | {0,1}
25-27      | Binary      | Mouse buttons (L,R,M)      | {0,1}
28-36      | One-Hot     | Hotbar slot (9 classes)    | {0,1}
37         | Binary      | GUI open                   | {0,1}
```

#### Reduced VPT Action Space (13D) - Current Implementation

```
Dimension  | Type        | Description                | Range
-----------|-------------|----------------------------|---------
0-3        | Continuous  | mouse_dx, dy, yaw, pitch   | ℝ (scaled)
4-12       | Binary      | Hotbar keys (1-9)          | {0,1}

Total: 13 dimensions
```

**Rationale for 13D:**
- Captures core Minecraft interactions (mouse + hotbar)
- Reduces predictor complexity
- Movement keys (w,a,s,d) can be inferred from camera changes
- Can be expanded to 36D if needed

### 3. Model Architecture

#### Encoder: Vision Transformer (Frozen)

```
Input: (B, T, C, H, W) = (8, 16, 3, 256, 256)

Patch Embedding (3D):
  - Patch size: 16×16
  - Tubelet size: 2
  - Output: (B, T/2 × H/16 × W/16, D) = (8, 8×16×16, 1024)
           = (8, 2048, 1024)

Positional Embeddings:
  - 3D sinusoidal (time × height × width)
  - RoPE (Rotary Position Embedding)

24 Transformer Blocks:
  - Multi-head self-attention (16 heads)
  - FFN with GELU activation
  - LayerNorm

Output: (B×T, D) = (8×2048, 1024)
```

#### Action-Conditioned Predictor (Trainable)

```
Input Preparation:
  1. Action Embedding: (B, T-1, 13) → (B, T-1, 1024)
     Linear projection: 13 → 1024

  2. State Embedding: (B, T-1, 2) → (B, T-1, 1024)
     Linear projection: 2 → 1024

  3. Frame Tokens: (B, T, 256, 1024)
     From frozen encoder (reshaped to (B, T, 256, 1024))

Interleaving:
  For each timestep t ∈ [0, T-1]:
    sequence[t] = [action_emb[t], state_emb[t], *frame_tokens[t]]
    # Shape: (2 + 256) = 258 tokens per timestep

  Total sequence: (B, T × 258, 1024) = (8, 2064, 1024)

Frame-Causal Attention Mask:
  - Tokens at time t can attend to tokens at time ≤ t
  - Actions/states at t can attend to all tokens at t
  - Prevents "cheating" by looking at future frames

24 Transformer Blocks (AC):
  - Causal masked multi-head attention
  - FFN with GELU
  - LayerNorm

Output Extraction:
  - Extract only frame tokens (drop action/state tokens)
  - Reshape: (B, T, 256, 1024)
  - Predictions: z[t] = prediction for frame t
```

### 4. Training Dynamics

#### Teacher Forcing (1-step prediction)

```python
# Given frames 0 to t-1, predict frame t
for t in range(1, T):
    context = h[:t]              # (B, t×256, 1024)
    actions_so_far = a[:t]       # (B, t-1, 13)
    states_so_far = s[:t]        # (B, t-1, 2)

    z_t = predictor(context, actions_so_far, states_so_far)

    loss_tf += MSE(z_t, h[t])    # Compare to ground truth

# Averaged over all timesteps
jloss = loss_tf / (T-1)
```

**Intuition:**
- Learns to predict next frame given perfect context
- "If I do action A, what will I see next?"
- Easy mode: always has ground truth history

#### Auto-Regressive Rollout (multi-step prediction)

```python
# Predict multiple steps into the future
z_prev = h[0]  # Start with first frame

for t in range(1, auto_steps+1):
    z_t = predictor(
        z_prev,                  # Use previous prediction (not ground truth!)
        a[:t],
        s[:t]
    )

    loss_ar += MSE(z_t, h[t])
    z_prev = z_t                 # Use prediction for next step

# Averaged over rollout steps
sloss = loss_ar / auto_steps
```

**Intuition:**
- Learns to predict without perfect context
- Errors compound over time (harder than teacher forcing)
- Encourages robust representations
- Tests "what happens if I keep doing actions?"

#### Combined Loss

```python
total_loss = jloss + sloss
```

### 5. Key Differences from Standard V-JEPA

| Aspect | V-JEPA (Pre-training) | V-JEPA-AC (VPT Fine-tuning) |
|--------|----------------------|----------------------------|
| **Encoder Training** | Trained end-to-end | **Frozen** (from pre-training) |
| **Predictor Input** | Masked frame tokens only | Actions + States + Frame tokens |
| **Masking Strategy** | Random spatial-temporal masks | Frame-causal masking |
| **Prediction Target** | Masked patches | Future frames (conditioned on actions) |
| **Loss Function** | MSE(predicted_masked, target_masked) | MSE(predicted_future, target_future) |
| **Dataset Requirement** | Any video (unlabeled) | Action-annotated video (VPT) |
| **Training Objective** | Self-supervised representation learning | Action-conditioned dynamics learning |

### 6. Why This Works

**1. Transfer Learning from Self-Supervised Pre-training:**
   - Pre-trained V-JEPA encoder already understands:
     - Object boundaries
     - Motion patterns
     - Spatial relationships
   - Fine-tuning adds: **action causality**

**2. Action-Conditioned Prediction:**
   - Encoder learns: "This is what the world looks like"
   - Predictor learns: "This is what happens when I do X"
   - Representation becomes **action-aware**

**3. Frame Causality:**
   - Real-world constraint: can't see the future
   - Forces model to learn temporal dependencies
   - Mimics agent's decision-making process

**4. Teacher Forcing + Auto-Regressive:**
   - Teacher forcing: stable training, fast convergence
   - Auto-regressive: prevents overfitting to one-step predictions
   - Combination: robust multi-step dynamics model

### 7. Downstream Applications

After fine-tuning, the encoder can be used for:

#### A) Inverse Dynamics Model
```python
# Given two frames, predict what action was taken
h_t = encoder(frame_t)
h_t1 = encoder(frame_{t+1})
predicted_action = inverse_model([h_t, h_t1])
```

**Use case:** Imitation learning from unlabeled gameplay videos

#### B) Action Anticipation
```python
# Given current frame, predict next action
h_t = encoder(frame_t)
next_action = action_predictor(h_t)
```

**Use case:** Predicting player intent (e.g., "about to mine")

#### C) Reinforcement Learning
```python
# Use encoder as visual backbone for RL policy
h_t = encoder(observation)
action = policy_head(h_t)
```

**Use case:** Training RL agents with better visual features

#### D) Video Classification
```python
# Classify Minecraft tasks/biomes
h = encoder(video_clip)
h_pooled = temporal_pool(h)
class_logits = classifier(h_pooled)
```

**Use case:** Automated gameplay analysis

---

## Summary

V-JEPA-AC fine-tuning on VPT creates a powerful video encoder that:
1. Understands Minecraft visual dynamics
2. Knows how actions (mouse, keyboard) affect the world
3. Can predict future states given actions
4. Provides rich features for downstream tasks

The key insight: **combining self-supervised visual pre-training with action-conditioned fine-tuning yields representations that are both visually rich and action-aware.**
