import os
import json
import subprocess
from pathlib import Path
import logging

from huggingface_hub import create_repo, upload_folder
from transformers import AutoModel, AutoTokenizer

from text_cleaning.utils import WANDB_DIR

logger = logging.getLogger(__name__)


def export_wandb(
    run_id: str | None = None,
    project_name: str | None = None,
    wandb_dir: str | Path = WANDB_DIR,
) -> None:
    """
    Upload wandb logs from offline mode to wandb server.

    Args:
        run_id: Specific run ID to upload. If None, uploads all offline runs.
        project_name: Project name to upload to. If None, uses default from environment.
        wandb_dir: Directory containing wandb logs. Defaults to "wandb".
    """
    wandb_dir = Path(wandb_dir)
    if not wandb_dir.exists():
        raise ValueError(f"Wandb directory not found at {wandb_dir}")

    # Set project name from environment or use default
    if project_name is None:
        project_name = os.environ.get("WANDB_PROJECT", None)
        if project_name is None:
            project_name = "mnlp-h2"
            os.environ["WANDB_PROJECT"] = project_name
    else:
        os.environ["WANDB_PROJECT"] = project_name

    # Check if WANDB_API_KEY is set
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError(
            "WANDB_API_KEY not found in environment variables. "
            "Please set it in your .env file or export it: export WANDB_API_KEY='your-api-key'"
        )

    # Find offline runs
    offline_runs = []
    for run_dir in wandb_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("offline-run-"):
            if run_id is None or run_id in run_dir.name:
                offline_runs.append(run_dir)

    if not offline_runs:
        logger.info("No offline runs found to upload.")
        return

    logger.info(f"Found {len(offline_runs)} offline run(s) to upload:")
    for run_dir in offline_runs:
        logger.info(f"  - {run_dir.name}")

    # Upload each run
    for run_dir in offline_runs:
        logger.info(f"Uploading {run_dir.name}...")
        try:
            # Use wandb sync to upload the offline run
            _ = subprocess.run(["wandb", "sync", str(run_dir)], capture_output=True, text=True, check=True)
            logger.info(f"Successfully uploaded {run_dir.name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error uploading {run_dir.name}: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("'wandb' command not found. Please install wandb CLI: pip install wandb")

    logger.info(f"Successfully uploaded {len(offline_runs)} run(s) to project '{project_name}'")


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Find the latest checkpoint in the given directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to the latest checkpoint directory
    """
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # Look for checkpoint directories (usually named like "checkpoint-1000", "checkpoint-2000", etc.)
    checkpoint_dirs = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            checkpoint_dirs.append(item)

    if not checkpoint_dirs:
        # If no checkpoint directories found, check if the directory itself contains model files
        safetensor_files = list(checkpoint_dir.glob("*.safetensors"))
        if safetensor_files:
            return checkpoint_dir
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Sort by checkpoint number and return the latest
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0)
    latest_checkpoint = checkpoint_dirs[-1]

    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def export_model(
    config_path: str | Path,
    private: bool = False,
    commit_message: str | None = None,
) -> None:
    """
    Export a LLaMA-Factory trained model to the Hugging Face Hub using a config file.

    This function works like the LLaMA-Factory export CLI. It loads the config JSON,
    finds the latest checkpoint in the adapter directory, and uploads the model,
    tokenizer, and related files to the Hugging Face Hub using Transformers' push_to_hub method.

    Args:
        config_path: Path to the LLaMA-Factory export config JSON file.
        private: Whether to create a private repo if it does not exist.
        commit_message: Git commit message for the upload. If None, uses a default message.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    # Load the config
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Extract required fields from config
    export_hub_model_id = config.get("export_hub_model_id")
    adapter_name_or_path = config.get("adapter_name_or_path")

    if not export_hub_model_id:
        raise ValueError("Config must contain 'export_hub_model_id' field")
    if not adapter_name_or_path:
        raise ValueError("Config must contain 'adapter_name_or_path' field")

    # Resolve relative paths relative to config file location
    adapter_path = Path(adapter_name_or_path)

    logger.info(f"Exporting model from {adapter_path} to {export_hub_model_id}")

    # Find the latest checkpoint
    latest_checkpoint = _find_latest_checkpoint(adapter_path)

    # Load model and tokenizer from the checkpoint
    logger.info(f"Loading model and tokenizer from {latest_checkpoint}")
    model = AutoModel.from_pretrained(latest_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)

    # Set default commit message if not provided
    if commit_message is None:
        commit_message = f"Upload fine-tuned model from LLaMA-Factory checkpoint: {latest_checkpoint.name}"

    # Push to hub using Transformers' built-in method
    logger.info(f"Pushing model to {export_hub_model_id}...")
    try:
        model.push_to_hub(export_hub_model_id, private=private, commit_message=commit_message)
        tokenizer.push_to_hub(export_hub_model_id, private=private, commit_message=commit_message)
        logger.info(f"Successfully uploaded model to {export_hub_model_id}")
    except Exception as e:
        raise RuntimeError(f"Error uploading model to {export_hub_model_id}: {e}")


def export_model_legacy(
    model_dir: str | Path,
    repo_base_id: str = "ClemensK/ocr-denoising",
    private: bool = False,
    commit_message: str = "Upload fine-tuned model from LLaMA-Factory",
) -> None:
    """
    Uploads a LLaMA-Factory trained model directory to the Hugging Face Hub.

    This is the legacy version of export_model. Use export_model() with config path instead.

    Args:
        model_dir: Path to the directory containing the fine-tuned model.
        repo_base_id: The Hugging Face repository ID (e.g., "username/model-name").
        private: Whether to create a private repo if it does not exist.
        commit_message: Git commit message for the upload.
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"Model directory does not exist: {model_dir}")

    model_name = Path(model_dir).name
    repo_id = f"{repo_base_id}-{model_name}"
    create_repo(repo_id, private=private, exist_ok=True)

    # Upload the directory
    upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        path_in_repo="",  # Upload as root
        commit_message=commit_message,
    )
