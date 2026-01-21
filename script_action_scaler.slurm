#!/bin/bash
#SBATCH --job-name=RYZH_JEPA_SCALER
#SBATCH --output=RYZH_JEPA_SCALER%j.out
#SBATCH --error=RYZH_JEPA_SCALER%j.err
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --account=dxe@v100
#SBATCH --hint=nomultithread


module purge

nvidia-smi

module load pytorch-gpu/py3/2.3.0
conda deactivate

conda activate /lustre/fswork/projects/rech/dxe/ueq71sr/pytorch-gpu-1.11+py3.9.12-ryzh

set -x
srun python action_scaler_fit_optimized.py --data_path "VPT/shard-*.tar" --output my_scaler_400k.pkl --max_samples 400000 --max_shards 10 --batch_size 4 --num_workers 0
