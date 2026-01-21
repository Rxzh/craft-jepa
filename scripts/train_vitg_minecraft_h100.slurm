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

# Load H100 architecture module and conda environment
module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate pytorch-gpu-2.4.0+py3.11.9

# Create logs directory if it doesn't exist
mkdir -p logs

# Set paths - update these for your setup
export WEIGHTS_DIR=$WORK/petrus/craft-jepa/weights
export VPT_DATA_PATH=$WORK/petrus/craft-jepa/VPT/shard-{000000..001648}.tar
export ACTION_SCALER_PATH=$WORK/petrus/craft-jepa/my_scaler_400k.pkl
export OUTPUT_DIR=$SCRATCH/vpt_jepa/vitg16-256px-8f

# Navigate to vjepa2 directory
cd $WORK/petrus/craft-jepa/vjepa2

# Add vjepa2 to Python path for module imports
export PYTHONPATH=$PWD:$PYTHONPATH

# Run training
srun python app/main.py \
    --fname configs/train/vitg16/minecraft-256px-8f.yaml \
    --debugmode

echo "Training completed"
