#!/bin/bash
#SBATCH --job-name=vitg-minecraft
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=2:00:00
#SBATCH --account=dxe@h100
#SBATCH --output=logs/%j_vitg_minecraft.out
#SBATCH --error=logs/%j_vitg_minecraft.err

# Jean-Zay H100 partition configuration
# Node specs: 96 CPU, 468 GB RAM, 4x H100 80GB
# Using dev QoS for initial testing (2h max)

# Load H100 architecture module
module purge
module load arch/h100
module load python/3.11.5

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths - update these for your setup
export WEIGHTS_DIR=weights
export VPT_DATA_PATH=VPT/vpt-shards-{000000..000123}.tar
export ACTION_SCALER_PATH=my_scaler_400k.pkl
export OUTPUT_DIR=$SCRATCH/vpt_jepa/vitg16-256px-8f

# Navigate to vjepa2 directory
cd $WORK/petrus/craft-jepa/vjepa2

# Run training
srun app/main.py \
    --fname configs/train/vitg16/minecraft-256px-8f.yaml \
    --debugmode

echo "Training completed"
