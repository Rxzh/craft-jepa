import io
import json
import glob
from logging import getLogger
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import webdataset as wds
from decord import VideoReader, cpu
from torchvision import transforms as T

# --- Setup ---
logger = getLogger(__name__)
torch.manual_seed(0)
np.random.seed(0)

# --------------------------------------------------------------------------
# 1. Main Data Loader Initialization Function (Mirrors your template)
# --------------------------------------------------------------------------

def init_vpt_dataloader(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    crop_size=256,
    rank=0,
    world_size=1,
    frameskip=2,  # Corresponds to `tubelet_size`
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
):
    """
    Initializes the WebDataset-based data loader for VPT.
    
    Args:
        data_path (str): A glob string for the .tar shards 
                         (e.g., "data/vpt-shards-{000000..000123}.tar")
    """
    logger.info(f"Initializing VPT WebDataset loader from: {data_path}")

    # Find all the shard files
    shard_urls = sorted(glob.glob(data_path))
    if not shard_urls:
        logger.error(f"No shards found at path: {data_path}")
        raise FileNotFoundError(f"No .tar files found at {data_path}")
        
    logger.info(f"Found {len(shard_urls)} shards.")

    # --- 1.1. Define Video Transformation ---
    # Your plan: Resize shortest side to 256, then 256x256 crop.
    # We also add normalization required by V-JEPA (standard ImageNet mean/std).
    vjepa_transform = T.Compose([
        # Input from ProcessVPT is (T, H, W, C) numpy array
        T.Lambda(lambda x: torch.from_numpy(x).permute(0, 3, 1, 2)),  # (T, C, H, W)
        T.Resize(crop_size, antialias=True),  # Resize shortest side to crop_size
        T.CenterCrop(crop_size),              # Crop 256x256 from center
        T.Lambda(lambda x: x.float() / 255.0), # Normalize to [0, 1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 1.2. Instantiate the Processor ---
    # This class contains the logic adapted from DROIDVideoDataset
    vpt_processor = ProcessVPT(
        frames_per_clip=frames_per_clip,
        fps=fps,
        frameskip=frameskip,
        transform=vjepa_transform
    )

    # --- 1.3. Build the WebDataset Pipeline ---
    dataset = wds.WebDataset(shard_urls)
    
    # For multi-epoch training, repeat the dataset
    dataset = dataset.repeat()
    
    # Shuffle the order of shards (buffer_size=100)
    dataset = dataset.shuffle(100)

    # Handle distributed training (DDP)
    if world_size > 1:
        # Split shards by node, and samples by worker
        dataset = dataset.split_by_node(rank, world_size) \
                         .split_by_worker(wds.worker_id, wds.num_workers)

    # Decode the raw bytes from the .tar file
    dataset = dataset.map(decode_sample)
    
    # Apply the DROID-based processing logic
    dataset = dataset.map(vpt_processor)
    
    # Filter out any samples that failed to load or process
    dataset = dataset.select(lambda x: x is not None)

    # --- 1.4. Create the PyTorch DataLoader ---
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=drop_last,
        # No sampler needed, WebDataset handles shuffling
        # Default collator works perfectly since our processor returns dicts
    )

    logger.info("VPT WebDataset data loader created")
    
    # Return None for sampler, as it's integrated into the WebDataset pipeline
    return data_loader, None 


# --------------------------------------------------------------------------
# 2. WebDataset Helper Functions (Decoding)
# --------------------------------------------------------------------------

def decode_sample(sample):
    """
    Decodes the raw bytes from a WebDataset sample.
    - '.mp4' bytes -> decord.VideoReader
    - '.jsonl' bytes -> pd.DataFrame
    """
    try:
        # --- 1. Decode JSONL (Actions) ---
        json_bytes = sample['.jsonl']
        json_string = json_bytes.decode('utf-8')
        # .jsonl is lines of json objects.
        action_list = [json.loads(line) for line in json_string.strip().split('\n')]
        actions_df = pd.DataFrame(action_list)
        sample['actions_df'] = actions_df

        # --- 2. Decode MP4 (Video) ---
        video_bytes = sample['.mp4']
        # Use io.BytesIO to treat the byte string as a file
        video_buffer = io.BytesIO(video_bytes)
        # Load the video from the in-memory buffer
        vr = VideoReader(video_buffer, ctx=cpu(0), num_threads=1)
        sample['video_reader'] = vr
        
    except Exception as e:
        logger.warning(f"Error decoding sample {sample['__key__']}: {e}")
        return None # Skip this sample
        
    return sample


# --------------------------------------------------------------------------
# 3. Core Logic Class (Adapted from DROIDVideoDataset)
# --------------------------------------------------------------------------

class ProcessVPT:
    """
    This class adapts the logic from `DROIDVideoDataset.__getitem__`
    and `loadvideo_decord` for the WebDataset pipeline.
    
    It takes a decoded sample (with 'video_reader' and 'actions_df')
    and performs:
    1. Frame sampling (random window, FPS-based indexing)
    2. Action processing (aligning actions to frames, calculating diffs)
    3. Video transformation
    """
    def __init__(self, frames_per_clip=16, fps=5, frameskip=2, transform=None):
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.frameskip = frameskip  # V-JEPA 2-AC's `tubelet_size`
        self.transform = transform

    def __call__(self, sample):
        """Processes a single decoded sample."""
        if sample is None:
            return None

        try:
            vr = sample['video_reader']
            actions_df = sample['actions_df']
            key = sample['__key__']

            # --- 1. Video Frame Sampling (from DROID `loadvideo_decord`) ---
            vfps = vr.get_avg_fps()
            fpc = self.frames_per_clip
            target_fps = self.fps if self.fps is not None else vfps
            
            # Calculate frame step size to achieve target FPS
            fstp = ceil(vfps / target_fps)
            
            # Total number of frames needed *before* sampling
            nframes = int(fpc * fstp)
            vlen = len(vr) # Total frames in video

            # Check if video is long enough
            if vlen < nframes:
                logger.debug(f"Skipping short video: {key}, {vlen=}, {nframes=}")
                return None
            
            # Check for action data mismatch
            if vlen != len(actions_df):
                 logger.debug(f"Skipping video: {key}, frame/action mismatch "
                              f"({vlen} frames vs {len(actions_df)} actions)")
                 return None

            # Sample a random window (end frame, start frame)
            ef = np.random.randint(nframes, vlen)
            sf = ef - nframes
            
            # Get the indices of frames to sample
            indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
            
            # Ensure indices are within bounds and we have enough
            indices = indices[indices < vlen]
            if len(indices) < fpc:
                 logger.debug(f"Skipping video (not enough frames after sampling): {key}")
                 return None
            indices = indices[:fpc]  # Ensure exactly `frames_per_clip`

            # Load the video frames (buffer)
            vr.seek(0)
            buffer = vr.get_batch(indices).asnumpy()  # (T, H, W, C)

            # --- 2. Action Processing (from DROID `loadvideo_decord`) ---
            
            # 2.1. Get states corresponding to the sampled video frames
            sampled_states_df = actions_df.iloc[indices]

            # 2.2. Sub-sample *these* states by `frameskip` (e.g., tubelet_size)
            # This matches the DROID logic: `states = states[indices, :][:: self.frameskip]`
            sub_sampled_states_df = sampled_states_df.iloc[::self.frameskip]
            
            # 2.3. Calculate action diffs (replaces `poses_to_diffs`)
            # This takes T states and returns T-1 actions
            vpt_actions_np = self.vpt_states_to_diffs(sub_sampled_states_df)

            # --- 3. Video Transform ---
            if self.transform:
                buffer = self.transform(buffer)  # (T, C, H, W) tensor
            
            # --- 4. Prepare Final Output ---
            actions_tensor = torch.from_numpy(vpt_actions_np).float()
            
            # Also return the sub-sampled states, just like the DROID loader
            # **ADJUST THESE COLUMNS** based on your needs
            state_cols = ['yaw', 'pitch'] 
            states_np = sub_sampled_states_df[state_cols].values
            states_tensor = torch.from_numpy(states_np).float()

            return {
                "video_frames": buffer,
                "actions": actions_tensor,
                "states": states_tensor
            }

        except Exception as e:
            logger.warning(f"Error processing sample {sample.get('__key__', 'N/A')}: {e}")
            # import traceback
            # traceback.print_exc()
            return None

    def vpt_states_to_diffs(self, states_df):
        """
        Calculates action "diffs" from a DataFrame of states.
        This is the critical adaptation of `poses_to_diffs` for VPT data.
        
        **!! ATTENTION !!**
        You MUST verify the column names in your .jsonl file and
        adjust the keys (e.g., 'yaw', 'mouse_dx') in this function.
        """
        
        # --- 1. Continuous states that need diffs (e.g., camera angle) ---
        # Assuming 'yaw' and 'pitch' are in RADIANS
        yaw = states_df['yaw'].values
        pitch = states_df['pitch'].values
        
        yaw_diff = yaw[1:] - yaw[:-1]
        pitch_diff = pitch[1:] - pitch[:-1]
        
        # Handle angle wrap-around (e.g., -pi to +pi)
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        pitch_diff = (pitch_diff + np.pi) % (2 * np.pi) - np.pi

        # --- 2. Actions that are already diffs (mouse) or discrete (keys) ---
        # We take the action from the *start* of the interval (T-1 actions)
        mouse_dx = states_df['mouse_dx'].values[:-1]
        mouse_dy = states_df['mouse_dy'].values[:-1]
        
        # --- 3. Discrete keyboard/mouse button presses ---
        # **ADJUST THIS LIST** to match all your action columns
        key_cols = [
            'attack', 'back', 'forward', 'jump', 'left', 
            'right', 'sneak', 'sprint', 'use'
        ]
        # Filter for keys that actually exist in the dataframe
        key_cols = [col for col in key_cols if col in states_df.columns]
        key_presses = states_df[key_cols].values[:-1]  # (T-1, num_keys)

        # --- 4. Concatenate all actions into a single vector ---
        # The final action vector shape will be (T_actions, D_actions)
        # where T_actions = (frames_per_clip / frameskip) - 1
        actions = np.concatenate([
            mouse_dx[:, np.newaxis],
            mouse_dy[:, np.newaxis],
            yaw_diff[:, np.newaxis],
            pitch_diff[:, np.newaxis],
            key_presses
        ], axis=1)
        
        return actions.astype(np.float32)


# --------------------------------------------------------------------------
# 4. Example Usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # --- Configuration ---
    # !! IMPORTANT: Update this path !!
    DATA_PATH = "path/to/your/vpt-shards-{000000..000001}.tar" # Use a small subset for testing
    
    BATCH_SIZE = 4
    FRAMES_PER_CLIP = 16
    FPS = 5
    FRAMESKIP = 2  # This is the `tubelet_size`
    CROP_SIZE = 256
    NUM_WORKERS = 0 # Set to 0 for initial debugging, then increase

    # --- Initialize Loader ---
    # We run in a single-process mode (world_size=1, rank=0) for this test
    try:
        data_loader, _ = init_vpt_dataloader(
            data_path=DATA_PATH,
            batch_size=BATCH_SIZE,
            frames_per_clip=FRAMES_PER_CLIP,
            fps=FPS,
            crop_size=CROP_SIZE,
            rank=0,
            world_size=1,
            frameskip=FRAMESKIP,
            num_workers=NUM_WORKERS,
        )

        # --- Fetch a Batch ---
        logger.info("Fetching one batch...")
        batch = next(iter(data_loader))

        # --- Inspect the Batch ---
        video_tensor = batch['video_frames']
        actions_tensor = batch['actions']
        states_tensor = batch['states']

        logger.info(f"Batch loaded successfully!")
        
        # (B, T, C, H, W)
        logger.info(f"  Video tensor shape: {video_tensor.shape}") 
        
        # (B, T_actions, D_actions)
        # T_actions = (FRAMES_PER_CLIP / FRAMESKIP) - 1 = (16/2) - 1 = 7
        logger.info(f"  Actions tensor shape: {actions_tensor.shape}")
        
        # (B, T_states, D_states)
        # T_states = FRAMES_PER_CLIP / FRAMESKIP = 16/2 = 8
        logger.info(f"  States tensor shape: {states_tensor.shape}")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please update the DATA_PATH variable in the __main__ block.")
    except KeyError as e:
         logger.error(f"Error: Missing column {e} in your .jsonl data.")
         logger.error("Please verify all column names in vpt_states_to_diffs().")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()