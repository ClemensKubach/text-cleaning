import os

from huggingface_hub import login
from huggingface_hub.errors import HfHubHTTPError

from text_cleaning.constants import IN_COLAB, WANDB_DIR


def setup_wandb():
    os.environ["WANDB_PROJECT"] = "mnlp-h2"
    os.environ["WANDB_DIR"] = str(WANDB_DIR)


def _read_hf_token() -> str | None:
    if IN_COLAB:
        try:
            return userdata.get("HF_TOKEN")  # type: ignore # noqa: F821
        except KeyError:
            return None
    else:
        return os.environ.get("HF_TOKEN", None)


def do_blocking_hf_login():
    # run the login in a separate cell because login is non-blocking
    try:
        token = _read_hf_token()
        login(token=token)
        if token is None:
            # block until logged-in
            input("Press enter of finish login!")
    except HfHubHTTPError:
        print("Login via HF_TOKEN secret/envvar and via manual login widget failed or not authorized.")
