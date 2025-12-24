import sys
sys.path.append('vjepa2')
import logging
import joblib
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from vjepa2.app.vjepa_minecraft.vpt_dataset import init_vpt_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

DATA_PATH = "VPT/shard-0000*.tar" 
SCALER_PATH = "vpt_action_scaler.pkl"

# OPTIMIZATION TIP: 
# Batch size 1 is very slow for statistics gathering. 
# Increase this to max out your GPU memory (e.g., 64, 128, 256).
FIT_BATCH_SIZE = 64  
# Increase workers to prefetch data while GPU computes
NUM_WORKERS = 4      

def fit_action_scaler_fast(data_path, scaler_save_path):
    logger.info(f"Initializing dataloader to fit scaler from: {data_path}")

    scaler_fit_loader, _ = init_vpt_dataloader(
        data_path=data_path,
        batch_size=FIT_BATCH_SIZE,
        action_scaler=None,
        frames_per_clip=64,
        fps=5,
        frameskip=2,
        crop_size=256,
        rank=0,
        world_size=1,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    # Initialize accumulators on the device (GPU if available)
    # Assuming 4 continuous action dimensions based on your snippet
    device = torch.device("cuda")
    
    total_count = 0
    total_sum = torch.zeros(4, device=device, dtype=torch.float64) # Use float64 for precision
    total_sq_sum = torch.zeros(4, device=device, dtype=torch.float64)

    logger.info("Starting fast scaler fitting loop (Sufficient Statistics)...")
    
    with torch.no_grad():
        for batch in tqdm(scaler_fit_loader, desc="Collecting Stats"):
            # Extract and reshape
            actions_batch = batch['actions'].to(device)
            cont_actions = actions_batch[..., :4].reshape(-1, 4).double() # Cast to double for precision

            if cont_actions.shape[0] > 0:
                # Accumulate stats purely in PyTorch
                total_count += cont_actions.shape[0]
                total_sum += torch.sum(cont_actions, dim=0)
                total_sq_sum += torch.sum(cont_actions ** 2, dim=0)

    # --- Final Calculation ---
    # Calculate Mean: E[X]
    mean = total_sum / total_count
    
    # Calculate Variance: E[X^2] - (E[X])^2
    # Note: We use the population variance formula (div by N) to match sklearn's default
    variance = (total_sq_sum / total_count) - (mean ** 2)
    
    # Handle potential negative zero due to precision issues
    variance = torch.clamp(variance, min=0.0)
    std_dev = torch.sqrt(variance)

    # Move to CPU/Numpy for Scikit-Learn
    mean_np = mean.cpu().numpy()
    std_np = std_dev.cpu().numpy()
    var_np = variance.cpu().numpy()

    # --- Manually Inject into StandardScaler ---
    # We create the object and force-feed the calculated attributes
    scaler = StandardScaler()
    scaler.mean_ = mean_np
    scaler.scale_ = std_np
    scaler.var_ = var_np
    scaler.n_samples_seen_ = total_count
    
    # Important: Set copy=True and with_mean/std=True to ensure it works correctly
    scaler.with_mean = True
    scaler.with_std = True
    scaler.copy = True

    # Save
    joblib.dump(scaler, scaler_save_path)
    
    logger.info("=" * 30)
    logger.info(f"Scaler fitting complete and saved to: {scaler_save_path}")
    logger.info(f"  Total Samples: {total_count}")
    logger.info(f"  Mean: {scaler.mean_}")
    logger.info(f"  Scale (StdDev): {scaler.scale_}")
    logger.info("=" * 30)
    
    return scaler

if __name__ == "__main__":
    try:
        fit_action_scaler_fast(DATA_PATH, SCALER_PATH)
        
        # Verify
        logger.info(f"Loading scaler from {SCALER_PATH} for verification...")
        loaded_scaler = joblib.load(SCALER_PATH)
        # Test a transform to ensure internal state is valid
        dummy_data = np.random.rand(5, 4)
        scaled_dummy = loaded_scaler.transform(dummy_data)
        logger.info("Verification transform successful.")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()