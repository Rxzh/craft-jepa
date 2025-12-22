import sys
sys.path.append('vjepa2')
import logging
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from vjepa2.app.vjepa_minecraft.vpt_dataset import init_vpt_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

DATA_PATH = "VPT/shard-0000*.tar" #first 100 shards only
SCALER_PATH = "vpt_action_scaler.pkl"
FIT_BATCH_SIZE = 1
NUM_WORKERS = 0


def fit_action_scaler(data_path, scaler_save_path):
    """
    Iterates through the dataset to fit a StandardScaler on the
    continuous action dimensions.
    """
    logger.info(f"Initializing dataloader to fit scaler from: {data_path}")

    scaler_fit_loader, _ = init_vpt_dataloader(
        data_path=data_path,
        batch_size=FIT_BATCH_SIZE,
        action_scaler=None,
        frames_per_clip=64,
        fps=5,
        frameskip=2, #tubelet size
        crop_size=256,
        rank=0,
        world_size=1,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    scaler = StandardScaler()
    
    logger.info("Starting scaler fitting loop...")
    
    with torch.no_grad():
        for batch in tqdm(scaler_fit_loader, desc="Fitting Scaler"):
            actions_batch_tensor = batch['actions']
            continuous_actions_tensor = actions_batch_tensor[:, :, :4]
            continuous_actions_flat = continuous_actions_tensor.reshape(-1, 4)
            continuous_actions_np = continuous_actions_flat.cpu().numpy()
            if len(continuous_actions_np) > 0:
                scaler.partial_fit(continuous_actions_np)

    joblib.dump(scaler, scaler_save_path)
    
    logger.info("=" * 30)
    logger.info(f"Scaler fitting complete and saved to: {scaler_save_path}")
    logger.info(f"  Mean: {scaler.mean_}")
    logger.info(f"  Scale (StdDev): {scaler.scale_}")
    logger.info("=" * 30)
    
    return scaler

if __name__ == "__main__":
    try:
        fit_action_scaler(DATA_PATH, SCALER_PATH)
        
        logger.info(f"Loading scaler from {SCALER_PATH} for verification...")
        loaded_scaler = joblib.load(SCALER_PATH)
        logger.info(f"Loaded Mean: {loaded_scaler.mean_}")
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please update the DATA_PATH variable in fit_scaler.py")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
