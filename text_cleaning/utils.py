import json
import os
from pathlib import Path
from typing import Literal, Union, Tuple
import logging
import sys
import random

import torch
from huggingface_hub import login
from huggingface_hub.errors import HfHubHTTPError
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    Pipeline,
    BitsAndBytesConfig,
)

from text_cleaning.constants import DATA_DIR, IN_COLAB, WANDB_DIR

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


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
        logger.error("Login via HF_TOKEN secret/envvar and via manual login widget failed or not authorized.")


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


def load_model(
    model_name: str = "google/gemma-3-1b-it",
    model_type: Literal["causal", "seq2seq"] = "causal",
    use_4bit: bool = False,
) -> tuple[Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], AutoTokenizer, Literal["causal", "seq2seq"]]:
    """Load the small LLM and tokenizer for denoising.

    Args:
        model_name: The name of the model to load.
        model_type: The type of model to load (causal or seq2seq).

    Returns:
        A tuple containing:
        - The loaded model
        - The tokenizer
        - The model type
    """
    model = None
    if model_type == "causal":
        try:
            if use_4bit:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                    device_map="auto",  # automatically place on GPU
                    torch_dtype="auto",  # pick FP16 on GPU if available
                ).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",  # automatically place on GPU
                    torch_dtype="auto",  # pick FP16 on GPU if available
                ).eval()
        except ValueError:
            raise ValueError(f"Wrong model type {model_type} for the model {model_name}")
    elif model_type == "seq2seq":
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            ).eval()
        except ValueError:
            raise ValueError(f"Model {model_name} is neither a causal LM nor a seq2seq model")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if model_type == "causal":
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, model_type


def load_pipeline(
    model_name: str = "google/gemma-3-1b-it", model_type: Literal["causal", "seq2seq"] = "causal"
) -> tuple[Pipeline, AutoTokenizer, Literal["causal", "seq2seq"]]:
    """Load a text generation pipeline for denoising.

    Args:
        model_name: The name of the model to load.
        model_type: The type of model to load (causal or seq2seq).

    Returns:
        A tuple containing:
        - The text generation pipeline
        - The tokenizer
        - The model type
    """
    model, tokenizer, model_type = load_model(model_name, model_type)

    # Create the pipeline with the appropriate task
    pipeline_task = "text2text-generation" if model_type == "seq2seq" else "text-generation"
    text_pipeline = pipeline(
        pipeline_task,
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        return_full_text=False,  # <-- only return the new text
        clean_up_tokenization_spaces=True,
        # do_sample=False,
        # temperature=0.2,
        # top_p=0.95,
    )
    return text_pipeline, tokenizer, model_type


def split_dataset(
    data: dict[int, str], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[dict[int, str], dict[int, str]]:
    """
    Split the dataset into training and test sets.

    Args:
        data: Dictionary mapping page numbers to text.
        test_ratio: Ratio of testing set size to total dataset size.
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
        - Training set dictionary
        - Testing set dictionary
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Get all page numbers and shuffle them
    page_numbers = list(data.keys())
    random.shuffle(page_numbers)

    # Calculate split index
    split_idx = int(len(page_numbers) * (1 - test_ratio))

    # Split into train and testing sets
    train_pages = page_numbers[:split_idx]
    test_pages = page_numbers[split_idx:]

    # Create train and testing dictionaries
    train_data = {page: data[page] for page in train_pages}
    test_data = {page: data[page] for page in test_pages}

    return train_data, test_data
