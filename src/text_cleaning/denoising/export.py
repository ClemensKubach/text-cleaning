import os
from pathlib import Path

from huggingface_hub import create_repo, upload_folder


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
