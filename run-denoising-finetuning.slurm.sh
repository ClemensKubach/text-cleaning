#!/bin/bash
#
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job configuration
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ocr_finetune                             # Job name
#SBATCH --output=logs/ocr_finetune-%x-%j.out                # Name of stdout output file. %x=job_name, %j=job_number
#SBATCH --error=logs/ocr_finetune-%x-%j.err                 # Name of stderr output file. %x=job_name, %j=job_number
#SBATCH -A try25_navigli            # account name
#SBATCH -p boost_usr_prod                       # partition (adjust as needed)
#SBATCH --time=00:15:00              # timing: HH:MM:SS
#SBATCH -N 1                         # number of nodes
#SBATCH --ntasks=1                   # number of tasks
#SBATCH --ntasks-per-node=1          # number of tasks per node
#SBATCH --cpus-per-task=4            # number of cpu per task (adjust as needed)
#SBATCH --gres=gpu:1                 # number of GPUs per node

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────

# load required modules (adjust versions as needed)
module load profile/deeplrn cuda/12.1 cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.1

curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/text-cleaning/LLaMA-Factory

# activate env
source .venv/bin/activate

# (optional) point HF and W&B to local caches if on a no‐internet cluster
export HF_HOME=~/hf_cache
export HF_DATASETS_CACHE=~/hf_cache
export HUGGINGFACE_HUB_CACHE=~/hf_cache
export TRANSFORMERS_CACHE=~/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline            # set the wandb offline

# (optional) load your HuggingFace token from local file
# export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

# ─────────────────────────────────────────────────────────────────────────────
# Run your Python module
# ─────────────────────────────────────────────────────────────────────────────

# Run LLaMA-Factory fine-tuning with uv and llamafactory-cli
uv run --prerelease=allow llamafactory-cli train ../data/fine_tuning/train_configs/ocr-llama-the_vampyre-config.json
