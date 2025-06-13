from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
WANDB_DIR = BASE_DIR / "wandb"

# Create directories if they don't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
WANDB_DIR.mkdir(parents=True, exist_ok=True)


try:
    from google.colab import userdata  # type: ignore  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False
