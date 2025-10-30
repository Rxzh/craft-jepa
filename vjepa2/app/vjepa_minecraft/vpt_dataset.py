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
    action_scaler=None,
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
        transform=vjepa_transform,
        action_scaler=action_scaler,
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
        
        # This DataFrame will have object columns for 'mouse' and 'keyboard'
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
    def __init__(self, frames_per_clip=16, fps=5, frameskip=2, transform=None, action_scaler=None):
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.frameskip = frameskip  # V-JEPA 2-AC's `tubelet_size`
        self.transform = transform
        self.action_scaler = action_scaler

        # --- Define Action Vocabulary ---
        # Based on VPT dataset exploration.
        # **EDIT THIS LIST** to match all keys you want to model.
        self.KEYBOARD_KEYS = [
            'W', 'A', 'S', 'D', 'Space', 'Shift', 'Sprint', 'E', 
            # DROID keys for reference (may or may not be in VPT):
            # 'attack', 'back', 'forward', 'jump', 'left', 
            # 'right', 'sneak', 'sprint', 'use'
        ]
        
        # **EDIT THIS LIST** for mouse buttons
        self.MOUSE_BUTTONS = ['left', 'right', 'middle']
        
        # Hotbar has 9 slots (0-8)
        self.HOTBAR_SIZE = 9


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
            
            vlen_total = len(vr)
            alen_total = len(actions_df)

            # --- 
            # MODIFICATION: Handle 6001 frames vs 6000 actions
            # We expect N actions for N+1 frames (e.g., 6000 actions, 6001 frames)
            # The N actions correspond to frames 0...N-1
            # The last frame (N) has no action data
            # ---
            if vlen_total != alen_total + 1:
                 logger.debug(f"Skipping video: {key}, frame/action mismatch "
                              f"({vlen_total} frames vs {alen_total} actions)")
                 return None

            # Usable length is the number of actions (e.g., 6000)
            # We can sample frames from index 0 to alen_total - 1
            vlen_usable = alen_total 

            if vlen_usable < nframes:
                logger.debug(f"Skipping short video: {key}, {vlen_usable=}, {nframes=}")
                return None
            
            # Sample a random window (end frame, start frame)
            # `high` is exclusive, so `vlen_usable + 1` makes the max
            # possible `ef` equal to `vlen_usable` (e.g., 6000)
            ef = np.random.randint(nframes, vlen_usable + 1) 
            sf = ef - nframes
            
            # Get the indices of frames to sample
            indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
            
            # Ensure indices are within bounds
            indices = indices[indices < vlen_usable] # Max index is vlen_usable - 1
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
            
            # 2.2. Sub-sample *both* the states AND the video buffer by `frameskip`
            sub_sampled_states_df = sampled_states_df.iloc[::self.frameskip] # 100 states
            sub_sampled_buffer = buffer[::self.frameskip] # 100 frames
            
            # 2.3. Calculate action diffs from the sub-sampled states
            # This takes 100 states and returns 99 actions
            vpt_actions_np = self.vpt_states_to_diffs(sub_sampled_states_df)

            # --- 3. Video Transform ---
            if self.transform:
                # Apply transform to the *sub-sampled* buffer
                sub_sampled_buffer = self.transform(sub_sampled_buffer)  
            
            # --- 4. Prepare Final Output ---
            actions_tensor = torch.from_numpy(vpt_actions_np).float()
            
            state_cols = ['yaw', 'pitch'] 
            states_np = sub_sampled_states_df[state_cols].values
            states_tensor = torch.from_numpy(states_np).float()

            return {
                "video_frames": sub_sampled_buffer, 
                "actions": actions_tensor,
                "states": states_tensor
            }
            # ---
            # END OF FIX
            # ---

        except Exception as e:
            logger.warning(f"Error processing sample {sample.get('__key__', 'N/A')}: {e}")
            return None

    def vpt_states_to_diffs(self, states_df):
        """
        Calculates action "diffs" from a DataFrame of states.
        This is the critical adaptation for VPT data, which uses
        object columns for 'mouse' and 'keyboard'.
        
        Takes T states and returns (T-1) actions.
        """
        
        # --- 1. Continuous states that need diffs (camera angle) ---
        # Get (T,) arrays
        yaw = states_df['yaw'].values
        pitch = states_df['pitch'].values
        
        # Get (T-1,) arrays of diffs
        yaw_diff = yaw[1:] - yaw[:-1]
        pitch_diff = pitch[1:] - pitch[:-1]
        
        # Handle angle wrap-around (e.g., -pi to +pi)
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        pitch_diff = (pitch_diff + np.pi) % (2 * np.pi) - np.pi

        # --- 2. Continuous actions that are already diffs (mouse dx, dy) ---
        # Get (T,) Series of dicts
        mouse_states = states_df['mouse']
        
        # Get (T-1,) arrays by taking diffs from the first T-1 states
        mouse_dx = mouse_states.apply(lambda m: m.get('dx', 0.0)).values[:-1]
        mouse_dy = mouse_states.apply(lambda m: m.get('dy', 0.0)).values[:-1]

        # --- A. Stack all continuous actions for normalization ---
        # Shape will be (T-1, 4)
        continuous_actions = np.stack([
            mouse_dx,
            mouse_dy,
            yaw_diff,
            pitch_diff
        ], axis=1)

        # --- B. Apply the scaler IF it was provided ---
        if self.action_scaler is not None:
            try:
                continuous_actions = self.action_scaler.transform(continuous_actions)
            except Exception as e:
                logger.error(f"Failed to transform actions with scaler: {e}")
                # Handle error, e.g., return zeros or raise
                continuous_actions = np.zeros_like(continuous_actions)

        
        # --- 3. Discrete (Keyboard) - Multi-Hot Encoding ---
        # Get (T-1,) Series of dicts
        keyboard_states = states_df['keyboard'].values[:-1]
        
        # Create a (T-1, num_keys) zero array
        key_presses_np = np.zeros((len(keyboard_states), len(self.KEYBOARD_KEYS)), dtype=np.float32)
        
        # Iterate and fill the multi-hot vector
        for i, k_state in enumerate(keyboard_states):
            pressed_keys = k_state.get('keys', []) # Get list of pressed keys
            for j, key_name in enumerate(self.KEYBOARD_KEYS):
                if key_name in pressed_keys:
                    key_presses_np[i, j] = 1.0

        # --- 4. Discrete (Mouse Buttons) - Multi-Hot Encoding ---
        # Get (T-1,) Series of dicts
        mouse_btn_states = states_df['mouse'].values[:-1]
        
        # Create a (T-1, num_buttons) zero array
        mouse_presses_np = np.zeros((len(mouse_btn_states), len(self.MOUSE_BUTTONS)), dtype=np.float32)
        
        # Iterate and fill
        for i, m_state in enumerate(mouse_btn_states):
            # Use 'buttons' for held keys. Use 'newButtons' for single-tick presses
            pressed_buttons = m_state.get('buttons', []) 
            for j, button_name in enumerate(self.MOUSE_BUTTONS):
                if button_name in pressed_buttons:
                    mouse_presses_np[i, j] = 1.0

        # --- 5. Discrete (Hotbar) - One-Hot Encoding ---
        # Get (T-1,) array of integer indices (0-8)
        hotbar_indices = states_df['hotbar'].values[:-1]
        
        # Create (T-1, 9) zero array
        hotbar_one_hot_np = np.zeros((len(hotbar_indices), self.HOTBAR_SIZE), dtype=np.float32)
        
        # Fill with 1s at the correct index
        hotbar_one_hot_np[np.arange(len(hotbar_indices)), hotbar_indices] = 1.0

        # --- 6. Discrete (GUI Open) - Binary ---
        is_gui_open_np = states_df['isGuiOpen'].values[:-1].astype(np.float32)

        # --- 7. Concatenate all actions into a single vector ---
        # Note the change: continuous_actions is now first
        actions = np.concatenate([
            continuous_actions, # This is the (T-1, 4) normalized block
            key_presses_np,
            mouse_presses_np,
            hotbar_one_hot_np,
            is_gui_open_np[:, np.newaxis]
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
    # Use a small subset of shards for testing (e.g., 2 files)
    DATA_PATH = "path/to/your/vpt-shards-{000000..000001}.tar" 
    
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
         logger.error(f"Error: Missing data key {e}.")
         logger.error("This could be a column in the DataFrame (e.g., 'hotbar')")
         logger.error("OR a key in a dict (e.g., 'dx' in the 'mouse' column).")
         logger.error("Please verify all keys in vpt_states_to_diffs().")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()