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
from app.vjepa_minecraft.utils import modify_keyboard_on_change

# --- Setup ---
logger = getLogger(__name__)
torch.manual_seed(0)
np.random.seed(0)

VPT_ACTION_DIM = 36

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
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    action_scaler=None,
):
    """
    Initializes the WebDataset-based data loader for VPT.

    Args:
        data_path (str or list): Either a glob pattern string for the .tar shards
                                 (e.g., "data/vpt-shards-*.tar") or a list of
                                 explicit shard file paths.
    """
    # Handle both glob pattern (string) and list of shard paths
    if isinstance(data_path, list):
        shard_urls = sorted(data_path)
        logger.info(f"Initializing VPT WebDataset loader with {len(shard_urls)} shards")
    else:
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
    dataset = wds.WebDataset(shard_urls, resampled=False)
    
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

        if not actions_df.empty and 'hotbar' in actions_df.columns:
            # A. Convert hotbar to numeric, filling NaNs with 0
            actions_df['hotbar'] = pd.to_numeric(actions_df['hotbar'], errors='coerce').fillna(0).astype(int)
            
            # B. Calculate shift to find changes
            # We compare row[t] vs row[t-1]
            # 'shift(1)' moves values down, so row[t] has the value of row[t-1]
            actions_df['prev_hotbar'] = actions_df['hotbar'].shift(1).fillna(actions_df['hotbar'])
            
            # C. Logic: If current != prev, we need to press the key for 'current'
            # We create a mask for rows where change happened
            change_mask = actions_df['hotbar'] != actions_df['prev_hotbar']
            
            # D. Create a column 'inject_hotbar_key'
            # Default is None
            actions_df['inject_hotbar_key'] = None
            
            # Only where mask is True, set the key string (e.g., "key.keyboard.1")
            # Note: Minecraft hotbar is 0-8 internally usually, but keys are 1-9. 
            # Check your dataset: if hotbar is 0-8, add 1. If 1-9, keep as is.
            # Assuming dataset is 0-8:
            actions_df.loc[change_mask, 'inject_hotbar_key'] = \
                actions_df.loc[change_mask, 'hotbar'].apply(lambda x: f"key.keyboard.{int(x)+1}" if 0 <= x < 9 else None)
                
            # E. Apply the modifier (only updates rows where inject_hotbar_key is not None)
            actions_df['keyboard'] = actions_df.apply(modify_keyboard_on_change, axis=1)

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
    Adapts DROID/VPT logic for WebDataset with V-JEPA tubelet aggregation.
    """
    def __init__(self, frames_per_clip=16, fps=5, frameskip=2, transform=None, action_scaler=None):
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.frameskip = frameskip 
        self.transform = transform
        self.scaler = action_scaler 

        self.KEYBOARD_KEYS =    [f'key.keyboard.{i}' for i in range(1,10)] + \
                                [f'key.keyboard.{x}' for x in ['w', 'a', 's', 'd', 'space']] + \
                                [f'key.keyboard.{x}' for x in ['q','f']] + \
                                [f'key.keyboard.{x}' for x in ['e']] + \
                                [f'key.keyboard.{x}' for x in ['escape']] + \
                                [f'key.keyboard.{x}' for x in ['left.shift', 'right.shift', 'left.control', ]] 
        
        self.MOUSE_BUTTONS = [0, 1, 2]
        self.HOTBAR_SIZE = 9
        
        self.total_dim = 4 + len(self.KEYBOARD_KEYS) + len(self.MOUSE_BUTTONS) + self.HOTBAR_SIZE + 1

    def __call__(self, sample):
        if sample is None:
            return None

        try:
            vr = sample['video_reader']
            actions_df = sample['actions_df']
            key = sample['__key__']

            # --- 1. Frame Sampling ---
            vfps = vr.get_avg_fps()
            fpc = self.frames_per_clip
            target_fps = self.fps if self.fps is not None else vfps
            fstp = ceil(vfps / target_fps)
            nframes = int(fpc * fstp)
            
            vlen_total = len(vr)
            alen_total = len(actions_df)

            # Loosened check: Actions usually match frames or are off by 1
            if abs(vlen_total - alen_total) > 2:
                 return None

            vlen_usable = min(vlen_total, alen_total)
            if vlen_usable < nframes:
                return None
            
            # Sample random window
            ef = np.random.randint(nframes, vlen_usable + 1) 
            sf = ef - nframes
            
            indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
            indices = indices[indices < vlen_usable]
            
            if len(indices) < fpc:
                 return None
            indices = indices[:fpc]

            # --- 2. Load Data ---
            vr.seek(0)
            buffer = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
            
            # Get the raw sequence of actions corresponding to frames
            # We do NOT subsample the DF yet, we need the intermediate rows for aggregation
            raw_window_df = actions_df.iloc[indices].reset_index(drop=True)

            # --- 3. Action Aggregation (The Fix) ---
            # We pass the full window DF and the frameskip.
            # returns (T_subsampled - 1, D)
            vpt_actions_np = self.vpt_states_to_diffs(raw_window_df, self.frameskip)

            # --- 4. Subsample Video and States ---
            # Video: We pick every k-th frame to match the tubelet start
            sub_sampled_buffer = buffer[::self.frameskip]
            
            # States (Yaw/Pitch): We pick every k-th state
            # Note: We need one more state than action to calculate the last diff pair
            sub_sampled_states_df = raw_window_df.iloc[::self.frameskip]

            # --- 5. Transform ---
            if self.transform:
                sub_sampled_buffer = self.transform(sub_sampled_buffer)  
            
            # --- 6. Final Tensors ---
            actions_tensor = torch.from_numpy(vpt_actions_np).float()
            
            state_cols = ['yaw', 'pitch'] 
            states_np = sub_sampled_states_df[state_cols].values
            states_tensor = torch.from_numpy(states_np).float()

            return {
                "video_frames": sub_sampled_buffer, 
                "actions": actions_tensor,
                "states": states_tensor
            }

        except Exception as e:
            logger.warning(f"Error processing {sample.get('__key__', 'N/A')}: {e}")
            return None

    def vpt_states_to_diffs(self, window_df, skip):
        """
        Aggregates actions over the 'skip' window.
        window_df: DataFrame of length T (e.g., 16)
        skip: int (e.g., 2)
        Returns: numpy array of shape (T/skip - 1, ActionDim)
        """
        
        # 1. Determine number of resulting action steps
        # If we have 16 frames and skip 2, we have 8 tubelets.
        # We need transitions between them, so 7 actions.
        n_steps = (len(window_df) // skip) - 1
        
        # Arrays to hold results
        mouse_dx_aggr = np.zeros(n_steps)
        mouse_dy_aggr = np.zeros(n_steps)
        yaw_diffs = np.zeros(n_steps)
        pitch_diffs = np.zeros(n_steps)
        
        key_presses = np.zeros((n_steps, len(self.KEYBOARD_KEYS)), dtype=np.float32)
        mouse_presses = np.zeros((n_steps, len(self.MOUSE_BUTTONS)), dtype=np.float32)
        hotbar_one_hot = np.zeros((n_steps, self.HOTBAR_SIZE), dtype=np.float32)
        gui_open = np.zeros(n_steps, dtype=np.float32)

        # Loop over the tubelet windows
        for i in range(n_steps):
            # The indices in the raw window covering this transition
            # If skip=2:
            # i=0 -> start_idx=0, end_idx=2 (covers frame 0, 1)
            # i=1 -> start_idx=2, end_idx=4 (covers frame 2, 3)
            start_idx = i * skip
            next_start_idx = (i + 1) * skip
            
            # Slice the dataframe for this specific tubelet duration
            chunk = window_df.iloc[start_idx : next_start_idx]
            
            # --- A. Continuous Aggregation ---
            
            # 1. Yaw/Pitch (Absolute coordinates)
            # We take the diff between the START of this chunk and the START of next chunk
            curr_yaw = window_df.iloc[start_idx]['yaw']
            next_yaw = window_df.iloc[next_start_idx]['yaw']
            yaw_diffs[i] = (next_yaw - curr_yaw + np.pi) % (2 * np.pi) - np.pi

            curr_pitch = window_df.iloc[start_idx]['pitch']
            next_pitch = window_df.iloc[next_start_idx]['pitch']
            pitch_diffs[i] = (next_pitch - curr_pitch + np.pi) % (2 * np.pi) - np.pi
            
            # 2. Mouse Delta (Relative velocities)
            # We must SUM the deltas occurring inside the chunk
            chunk_mouse = chunk['mouse']
            dx_sum = sum(m.get('dx', 0.0) for m in chunk_mouse)
            dy_sum = sum(m.get('dy', 0.0) for m in chunk_mouse)
            mouse_dx_aggr[i] = dx_sum
            mouse_dy_aggr[i] = dy_sum

            # --- B. Discrete Aggregation (Logical OR) ---
            
            # Collect all keys pressed in ANY frame within this chunk
            chunk_keys = set()
            for k_state in chunk['keyboard']:
                chunk_keys.update(k_state.get('keys', []))
            
            for k_idx, key_name in enumerate(self.KEYBOARD_KEYS):
                if key_name in chunk_keys:
                    key_presses[i, k_idx] = 1.0
            
            # Collect all mouse buttons pressed
            chunk_buttons = set()
            for m_state in chunk['mouse']:
                chunk_buttons.update(m_state.get('buttons', []))
                
            for b_idx, btn_name in enumerate(self.MOUSE_BUTTONS):
                if btn_name in chunk_buttons:
                    mouse_presses[i, b_idx] = 1.0

            # --- C. State-based (Hotbar/GUI) ---
            # Usually take the state at the start of the transition
            # Alternatively, you could use 'mode' (most frequent), but start is standard
            start_row = window_df.iloc[start_idx]
            
            if 'hotbar' in start_row:
                hb_idx = int(start_row['hotbar'])
                if 0 <= hb_idx < self.HOTBAR_SIZE:
                    hotbar_one_hot[i, hb_idx] = 1.0
            
            if 'isGuiOpen' in start_row:
                gui_open[i] = float(start_row['isGuiOpen'])

        # --- D. Final Assembly ---
        continuous_actions = np.stack([
            mouse_dx_aggr, mouse_dy_aggr, yaw_diffs, pitch_diffs
        ], axis=1)

        if self.scaler is not None:
            continuous_actions = self.scaler.transform(continuous_actions)

        actions = np.concatenate([
            continuous_actions,
            key_presses,
            mouse_presses,
            hotbar_one_hot,
            gui_open[:, np.newaxis]
        ], axis=1)
        
        return actions.astype(np.float32)


# --------------------------------------------------------------------------
# 4. Example Usage
# --------------------------------------------------------------------------

if __name__ == "__main__":
    
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    DATA_PATH = "path/to/your/vpt-shards-{000000..000001}.tar" 
    
    BATCH_SIZE = 4
    FRAMES_PER_CLIP = 16
    FPS = 5
    FRAMESKIP = 2  # This is the `tubelet_size`
    CROP_SIZE = 256
    NUM_WORKERS = 0 # Set to 0 for initial debugging, then increase

    # --- Initialize Loader ---
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

        logger.info("Fetching one batch...")
        batch = next(iter(data_loader))

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