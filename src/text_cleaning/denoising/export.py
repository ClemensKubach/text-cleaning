import os
import subprocess
from pathlib import Path
import logging

from huggingface_hub import create_repo, upload_folder

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


def export_model(
    model_dir: str | Path,
    repo_base_id: str = "ClemensK/ocr-denoising",
    private: bool = False,
    commit_message: str = "Upload fine-tuned model from LLaMA-Factory",
) -> None:
    """
    Uploads a LLaMA-Factory trained model directory to the Hugging Face Hub.

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
