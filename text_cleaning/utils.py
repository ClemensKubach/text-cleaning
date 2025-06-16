import json
import os
from pathlib import Path

from huggingface_hub import login
from huggingface_hub.errors import HfHubHTTPError

from text_cleaning.constants import DATA_DIR, IN_COLAB, WANDB_DIR


def setup_wandb():
    os.environ["WANDB_PROJECT"] = "mnlp-h2"
    os.environ["WANDB_DIR"] = str(WANDB_DIR)


def _read_hf_token() -> str | None:
    """Read the HF_TOKEN from the userdata or the environment variables."""
    if IN_COLAB:
        try:
            return userdata.get("HF_TOKEN")  # type: ignore # noqa: F821
        except KeyError:
            return None
    else:
        return os.environ.get("HF_TOKEN", None)


def do_blocking_hf_login():
    """Run the login in a separate cell because login is non-blocking."""
    try:
        token = _read_hf_token()
        login(token=token)
        if token is None:
            # block until logged-in
            input("Press enter of finish login!")
    except HfHubHTTPError:
        print("Login via HF_TOKEN secret/envvar and via manual login widget failed or not authorized.")


def load_data(file_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json") -> dict[int, str]:
    """
    Loads data from a JSON file.

    Args:
        file_path: The path to the dataset.

    Returns:
        A dictionary mapping page numbers to text.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def save_data(file_path: Path, data: dict) -> None:
    """
    Saves data to a JSON file.

    Args:
        file_path: The path to the dataset.
        data: The data to save.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def model_name_to_path_compatible(model_name: str) -> str:
    """
    Converts a model name to a path compatible string.
    """
    return model_name.replace("/", "-")
