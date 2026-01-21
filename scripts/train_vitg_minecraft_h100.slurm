#!/bin/bash
#SBATCH --job-name=vitg-minecraft
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=2:00:00
#SBATCH --account=dxe@h100
#SBATCH --output=logs/%j_vitg_minecraft.out
#SBATCH --error=logs/%j_vitg_minecraft.err

# Jean-Zay H100 partition configuration (gpu_p6)
# Node specs: 96 CPU, 468 GB RAM, 4x H100 80GB
# Using dev QoS for initial testing (2h max, 32 GPU max)
# cpus-per-task=24 reserves 1/4 of node memory per GPU (per IDRIS docs)

# Load H100 architecture module and conda environment
module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate /lustre/fswork/projects/rech/dxe/ueq71sr/pytorch-gpu-1.11+py3.9.12-ryzh

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths for VJEPA config (these match the ${VAR} references in the YAML config)
export VJEPA_PRETRAIN_CHECKPOINT=$WORK/petrus/craft-jepa/weights/vitg.pt
export VJEPA_DATA_PATH="$WORK/petrus/craft-jepa/VPT/shard-{000000..001648}.tar"
export VJEPA_ACTION_SCALER_PATH=$WORK/petrus/craft-jepa/my_scaler_400k.pkl
export VJEPA_OUTPUT_DIR=$SCRATCH/vpt_jepa/vitg16-256px-8f

# Navigate to vjepa2 directory
cd $WORK/petrus/craft-jepa/vjepa2

# Add vjepa2 to Python path for module imports
export PYTHONPATH=$PWD:$PYTHONPATH

# Run training (debugmode runs single GPU without multiprocessing)
srun python app/main.py \
    --fname configs/train/vitg16/minecraft-256px-8f.yaml \
    --debugmode

echo "Training completed"
