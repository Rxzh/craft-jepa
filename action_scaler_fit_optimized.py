#!/usr/bin/env python3
"""
Optimized Action Scaler Fitting for Large-Scale VPT Dataset

Key Optimizations:
1. Fixed sample budget (default: 100K actions, configurable)
2. Early stopping when budget reached
3. Automatic shard sampling (use subset of shards)
4. Progress tracking with ETA
5. Statistical validation (convergence check)
6. GPU-accelerated with float64 precision

Usage:
    python action_scaler_fit_optimized.py --max_samples 100000 --shards 10
"""

import sys
sys.path.append('vjepa2')
import argparse
import logging
import joblib
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import glob
from vjepa2.app.vjepa_minecraft.vpt_dataset import init_vpt_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


def estimate_convergence(running_mean, running_std, new_mean, new_std, threshold=0.01):
    """
    Check if statistics have converged.

    Args:
        running_mean: Current running mean
        running_std: Current running std
        new_mean: Mean after adding new batch
        new_std: Std after adding new batch
        threshold: Relative change threshold (1% default)

    Returns:
        bool: True if converged
    """
    mean_change = torch.abs((new_mean - running_mean) / (running_mean + 1e-8))
    std_change = torch.abs((new_std - running_std) / (running_std + 1e-8))

    return (mean_change < threshold).all() and (std_change < threshold).all()


def fit_action_scaler_optimized(
    data_path,
    scaler_save_path,
    max_samples=100000,
    max_shards=None,
    batch_size=128,
    num_workers=8,
    check_convergence=True,
    convergence_patience=5
):
    """
    Fit action scaler with early stopping based on sample budget.

    Args:
        data_path: Glob pattern for VPT shards (e.g., "VPT/shard-*.tar")
        scaler_save_path: Where to save the fitted scaler
        max_samples: Maximum number of ACTION samples to use (not video clips)
        max_shards: Limit number of shards to process (None = all matching)
        batch_size: Batch size for dataloader
        num_workers: Number of dataloader workers
        check_convergence: Stop early if statistics converge
        convergence_patience: Number of checks before declaring convergence
    """

    logger.info("=" * 60)
    logger.info("OPTIMIZED ACTION SCALER FITTING")
    logger.info("=" * 60)

    # --- 1. Shard Selection ---
    all_shards = sorted(glob.glob(data_path))
    if not all_shards:
        raise FileNotFoundError(f"No shards found at: {data_path}")

    if max_shards is not None and max_shards < len(all_shards):
        # Sample uniformly across dataset
        indices = np.linspace(0, len(all_shards)-1, max_shards, dtype=int)
        selected_shards = [all_shards[i] for i in indices]
        logger.info(f"Using {max_shards} / {len(all_shards)} shards (sampled uniformly)")
    else:
        selected_shards = all_shards
        logger.info(f"Using all {len(all_shards)} shards")

    # --- 2. Initialize Dataloader ---
    logger.info(f"Initializing dataloader...")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Workers: {num_workers}")
    logger.info(f"  Target samples: {max_samples:,}")

    scaler_fit_loader, _ = init_vpt_dataloader(
        data_path=selected_shards,  # Pass list of shard paths directly
        batch_size=batch_size,
        action_scaler=None,  # Don't apply scaler during fitting
        frames_per_clip=64,  # Use more frames per clip for efficiency
        fps=5,
        frameskip=2,
        crop_size=256,
        rank=0,
        world_size=1,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True,
    )

    # --- 3. Initialize Accumulators ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computing on: {device}")

    total_count = 0
    total_sum = torch.zeros(4, device=device, dtype=torch.float64)
    total_sq_sum = torch.zeros(4, device=device, dtype=torch.float64)

    # For convergence tracking
    running_mean = None
    running_std = None
    convergence_counter = 0

    # --- 4. Collect Statistics ---
    logger.info("Collecting action statistics...")
    start_time = time.time()

    pbar = tqdm(total=max_samples, desc="Action Samples", unit="samples")

    batch_count = 0
    converged = False

    with torch.no_grad():
        for batch in scaler_fit_loader:
            # Extract continuous actions (first 4 dims)
            actions_batch = batch['actions'].to(device)  # (B, T_actions, D)
            cont_actions = actions_batch[..., :4].reshape(-1, 4).double()

            if cont_actions.shape[0] == 0:
                continue

            # Accumulate statistics
            batch_samples = cont_actions.shape[0]
            total_count += batch_samples
            total_sum += torch.sum(cont_actions, dim=0)
            total_sq_sum += torch.sum(cont_actions ** 2, dim=0)

            pbar.update(batch_samples)
            batch_count += 1

            # --- Check for early stopping ---

            # 1. Sample budget reached
            if total_count >= max_samples:
                logger.info(f"\n✓ Reached sample budget: {total_count:,} samples")
                break

            # 2. Convergence check (every 10 batches)
            if check_convergence and batch_count % 10 == 0:
                current_mean = total_sum / total_count
                current_var = (total_sq_sum / total_count) - (current_mean ** 2)
                current_var = torch.clamp(current_var, min=0.0)
                current_std = torch.sqrt(current_var)

                if running_mean is not None:
                    if estimate_convergence(running_mean, running_std, current_mean, current_std):
                        convergence_counter += 1
                        if convergence_counter >= convergence_patience:
                            logger.info(f"\n✓ Statistics converged at {total_count:,} samples")
                            converged = True
                            break
                    else:
                        convergence_counter = 0

                running_mean = current_mean.clone()
                running_std = current_std.clone()

    pbar.close()

    elapsed = time.time() - start_time
    logger.info(f"Collection completed in {elapsed:.1f}s ({total_count/elapsed:.0f} samples/sec)")

    # --- 5. Compute Final Statistics ---
    logger.info("Computing final statistics...")

    mean = total_sum / total_count
    variance = (total_sq_sum / total_count) - (mean ** 2)
    variance = torch.clamp(variance, min=0.0)
    std_dev = torch.sqrt(variance)

    # Move to CPU for sklearn
    mean_np = mean.cpu().numpy()
    std_np = std_dev.cpu().numpy()
    var_np = variance.cpu().numpy()

    # --- 6. Create StandardScaler ---
    scaler = StandardScaler()
    scaler.mean_ = mean_np
    scaler.scale_ = std_np
    scaler.var_ = var_np
    scaler.n_samples_seen_ = total_count
    scaler.with_mean = True
    scaler.with_std = True
    scaler.copy = True

    # --- 7. Save ---
    joblib.dump(scaler, scaler_save_path)

    # --- 8. Report ---
    logger.info("=" * 60)
    logger.info("SCALER FITTING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Saved to: {scaler_save_path}")
    logger.info(f"Total samples: {total_count:,}")
    logger.info(f"Converged: {converged}")
    logger.info("")
    logger.info("Action Statistics (Continuous Actions):")
    logger.info(f"  mouse_dx  - Mean: {mean_np[0]:8.3f}, Std: {std_np[0]:8.3f}")
    logger.info(f"  mouse_dy  - Mean: {mean_np[1]:8.3f}, Std: {std_np[1]:8.3f}")
    logger.info(f"  yaw_diff  - Mean: {mean_np[2]:8.3f}, Std: {std_np[2]:8.3f}")
    logger.info(f"  pitch_diff- Mean: {mean_np[3]:8.3f}, Std: {std_np[3]:8.3f}")
    logger.info("=" * 60)

    return scaler


def verify_scaler(scaler_path):
    """Verify the saved scaler works correctly."""
    logger.info("Verifying scaler...")

    loaded_scaler = joblib.load(scaler_path)

    # Test transform
    test_data = np.random.randn(10, 4) * 100  # Random actions
    try:
        scaled = loaded_scaler.transform(test_data)
        inverse = loaded_scaler.inverse_transform(scaled)

        # Check reconstruction
        reconstruction_error = np.abs(test_data - inverse).max()
        assert reconstruction_error < 1e-5, f"Reconstruction error too high: {reconstruction_error}"

        logger.info("✓ Scaler verification passed")
        logger.info(f"  Transform shape: {test_data.shape} → {scaled.shape}")
        logger.info(f"  Scaled data mean: {scaled.mean(axis=0)}")
        logger.info(f"  Scaled data std: {scaled.std(axis=0)}")

    except Exception as e:
        logger.error(f"✗ Scaler verification failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit action scaler for VPT dataset")

    parser.add_argument(
        "--data_path",
        type=str,
        default="VPT/shard-*.tar",
        help="Glob pattern for VPT shards"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vpt_action_scaler.pkl",
        help="Output path for scaler"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100000,
        help="Maximum number of action samples (not videos)"
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Maximum number of shards to use (None = all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for dataloader"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--no_convergence_check",
        action="store_true",
        help="Disable convergence-based early stopping"
    )

    args = parser.parse_args()

    try:
        # Fit scaler
        scaler = fit_action_scaler_optimized(
            data_path=args.data_path,
            scaler_save_path=args.output,
            max_samples=args.max_samples,
            max_shards=args.max_shards,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            check_convergence=not args.no_convergence_check,
        )

        # Verify
        verify_scaler(args.output)

        logger.info("\n✓ All done! You can now use this scaler for training.")

    except KeyboardInterrupt:
        logger.warning("\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
