import json
import os
from pathlib import Path
from typing import Literal, Union, Tuple
import logging
import sys
import random
from strenum import StrEnum

import torch
from huggingface_hub import login
from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoTokenizer,
)
from transformers.pipelines import pipeline, Pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig

from text_cleaning.constants import (
    DATA_DIR,
    IN_COLAB,
    SYNTHETIC_CLEAN_DATASET_PATH,
    SYNTHETIC_OCR_DATASET_PATH,
    WANDB_DIR,
)

logger = logging.getLogger(__name__)


class Model(StrEnum):
    GEMMA = "google/gemma-3-1b-it"
    LLAMA = "meta-llama/Llama-3.2-1B-Instruct"
    MINERVA = "sapienzanlp/Minerva-1B-base-v1.0"


def setup_logging():
    """Setup logging for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def setup_wandb(project_name: str = "mnlp-h2"):
    """Setup wandb configuration.

    Args:
        project_name: The name of the wandb project. Defaults to "mnlp-h2".
    """
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_DIR"] = str(WANDB_DIR)

    # Check if WANDB_API_KEY is set in environment
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not found in environment variables. Please set it to enable wandb logging.")


def _read_hf_token() -> str | None:
    """Read the HF_TOKEN from the userdata or the environment variables."""
    if IN_COLAB:
        try:
            from google.colab import userdata  # type: ignore[import-untyped]

            return userdata.get("HF_TOKEN")
        except (KeyError, ImportError):
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
    except OfflineModeIsEnabled:
        logger.warning("Offline mode is enabled. Please disable it to login.")


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
                    cache_dir=os.environ.get("HF_HOME"),
                ).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",  # automatically place on GPU
                    cache_dir=os.environ.get("HF_HOME"),
                ).eval()
        except ValueError:
            raise ValueError(f"Wrong model type {model_type} for the model {model_name}")
    elif model_type == "seq2seq":
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, device_map="auto", cache_dir=os.environ.get("HF_HOME")
            ).eval()
        except ValueError:
            raise ValueError(f"Model {model_name} is neither a causal LM nor a seq2seq model")

    model.forward = torch.compile(model.forward, fullgraph=False, dynamic=True)  # avoid recompiling
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=os.environ.get("HF_HOME"))
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


def merge_datasets(
    noisy_datasets: list[str],
    clean_datasets: list[str],
    out_noisy_path: str | Path = SYNTHETIC_OCR_DATASET_PATH,
    out_clean_path: str | Path = SYNTHETIC_CLEAN_DATASET_PATH,
) -> None:
    """Merge multiple datasets into a single dataset."""
    merged_noisy_data = {}
    merged_clean_data = {}
    new_page_number = 0
    for noisy_dataset, clean_dataset in zip(noisy_datasets, clean_datasets):
        noisy_data = load_data(Path(noisy_dataset))
        clean_data = load_data(Path(clean_dataset))
        for page, noisy_text in noisy_data.items():
            merged_noisy_data[new_page_number] = noisy_text
            merged_clean_data[new_page_number] = clean_data.pop(page)
            new_page_number += 1
        if len(clean_data) > 0:
            raise ValueError(
                f"Clean data has {len(clean_data)} pages left after merging, maybe the passed datasets are not aligned?"
            )
    save_data(Path(out_noisy_path), merged_noisy_data)
    save_data(Path(out_clean_path), merged_clean_data)
    logger.info(f"Merged datasets saved to {out_noisy_path} and {out_clean_path}")


def _get_model_id_from_string(model_str: str) -> str:
    """Convert a simple model string to the full model ID."""
    model_str_lower = model_str.lower()
    if model_str_lower == "gemma":
        return Model.GEMMA.value
    elif model_str_lower == "llama":
        return Model.LLAMA.value
    elif model_str_lower == "minerva":
        return Model.MINERVA.value
    else:
        raise ValueError(f"Unknown model string: {model_str}. Supported values: gemma, llama, minerva")


def cache_model_and_tokenizer(
    model: Model | str | None = None,
    model_id: str | None = None,
    model_type: Literal["causal", "seq2seq"] = "causal",
):
    """Caches the model and tokenizer to be used by LLaMA-Factory offline."""
    if model is not None:
        if isinstance(model, Model):
            model_id = model.value
        else:
            model_id = _get_model_id_from_string(model)
    if model_id is None:
        raise ValueError("Model ID is required to cache a model")
    logger.info(f"Caching model and tokenizer for {model_id}...")
    try:
        if model_type == "causal":
            AutoModelForCausalLM.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME"))
        elif model_type == "seq2seq":
            AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME"))
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported values: causal, seq2seq")
        AutoTokenizer.from_pretrained(model_id, cache_dir=os.environ.get("HF_HOME"))
        logger.info(f"Successfully cached {model_id}.")
        cache_location = os.environ.get("HF_HOME")
        logger.info(
            f"Model and tokenizer are saved in: {cache_location if cache_location else 'hf default cache location'}"
        )
    except Exception as e:
        logger.error(f"Failed to cache model {model_id}: {e}")
