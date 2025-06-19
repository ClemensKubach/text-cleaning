#!/bin/bash
#
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job configuration
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=llama_factory_finetune                             # Job name
#SBATCH --output=logs/llama_factory_finetune-%x-%j.out                # Name of stdout output file. %x=job_name, %j=job_number
#SBATCH --error=logs/llama_factory_finetune-%x-%j.err                 # Name of stderr output file. %x=job_name, %j=job_number
#SBATCH -A <Account_Name>            # account name
#SBATCH -p gpu                       # partition (adjust as needed)
#SBATCH --time=24:00:00              # timing: HH:MM:SS
#SBATCH -N 1                         # number of nodes
#SBATCH --ntasks=1                   # number of tasks
#SBATCH --ntasks-per-node=1          # number of tasks per node
#SBATCH --cpus-per-task=4            # number of cpu per task (adjust as needed)
#SBATCH --gres=gpu:1                 # number of GPUs per node

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────

# load required modules (adjust versions as needed)
module load profile/deeplrn cuda/12.1

# activate your virtualenv / conda env
source /path/to/your/env/bin/activate

# (optional) point HF and W&B to local caches if on a no‐internet cluster
export HF_DATASETS_CACHE=/path/to/hf_cache
export HUGGINGFACE_HUB_CACHE=/path/to/hf_cache
export WANDB_MODE=offline            # set the wandb offline

# (optional) load your HuggingFace token from local file
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

# ─────────────────────────────────────────────────────────────────────────────
# Run your Python module
# ─────────────────────────────────────────────────────────────────────────────

# Run LLaMA-Factory fine-tuning with uv and llamafactory-cli
uv run --prerelease=allow llamafactory-cli train ../data/fine_tuning/train_configs/ocr-llama-the_vampyre-config.json
