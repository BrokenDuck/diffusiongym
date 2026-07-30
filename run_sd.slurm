#!/usr/bin/bash
#SBATCH --job-name=sd-adjoint
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=sd-adjoint-%j.out
#SBATCH --error=sd-adjoint-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export HF_HOME="$SCRATCH/huggingface"
export TORCH_HOME="$SCRATCH/torch"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working directory: $PWD"

nvidia-smi
uv run --frozen python run_sd.py
