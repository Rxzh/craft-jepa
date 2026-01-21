#!/bin/bash
#SBATCH --job-name=vitg-minecraft-finetune
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00
#SBATCH --account=dxe@h100
#SBATCH --output=logs/%j_vitg_minecraft_finetune.out
#SBATCH --error=logs/%j_vitg_minecraft_finetune.err

# Jean-Zay H100 partition (gpu_p6): 96 CPU, 468 GB RAM, 4x H100 80GB per node
# cpus-per-task=24 reserves 1/4 of node memory per GPU (117 GB)
# ntasks-per-node=4 spawns 4 processes for distributed training

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate /lustre/fswork/projects/rech/dxe/ueq71sr/pytorch-gpu-1.11+py3.9.12-ryzh

mkdir -p logs

export VJEPA_PRETRAIN_CHECKPOINT=$WORK/petrus/craft-jepa/weights/vitg.pt
export VJEPA_DATA_PATH="$WORK/petrus/craft-jepa/VPT/shard-{000000..001648}.tar"
export VJEPA_ACTION_SCALER_PATH=$WORK/petrus/craft-jepa/my_scaler_400k.pkl
export VJEPA_OUTPUT_DIR=$SCRATCH/vpt_jepa/vitg16-256px-8f-finetune

cd $WORK/petrus/craft-jepa/vjepa2
export PYTHONPATH=$PWD:$PYTHONPATH

srun python app/main.py \
    --fname configs/train/vitg16/minecraft-256px-8f-finetune.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3

echo "Fine-tuning completed"
