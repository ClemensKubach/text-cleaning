from dataclasses import dataclass
from pathlib import Path
from sre_constants import LITERAL
from tkinter import N

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoModelForSeq2SeqLM, AutoTokenizer

from text_cleaning.constants import DATA_DIR
from text_cleaning.utils import do_blocking_hf_login, load_data, model_name_to_path_compatible, save_data
import argparse
from typing import Literal, Union

@dataclass
class TextChunk:
    """Represents a chunk of text with its start and end positions.

    Attributes:
        text: The text of the chunk.
        start: The start position of the chunk in the original text.
        end: The end position of the chunk in the original text.
    """

    text: str
    start: int
    end: int


# Initialize model and tokenizer as global variables to avoid reloading
DENOISING_MODEL: AutoModelForCausalLM | None = None
DENOISING_TOKENIZER: AutoTokenizer | None = None
MODEL_TYPE: str = ""


def _load_model(
    model_name: str = "google/gemma-3-1b-it",model_type: str = "causal"
) -> tuple[Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], AutoTokenizer, Literal["causal", "seq2seq"]]:
    """Load the small LLM and tokenizer for denoising if not already loaded.

    Args:
        model_name: The name of the model to load.

    Returns:
        The model and tokenizer.
    """
    global DENOISING_MODEL, DENOISING_TOKENIZER, MODEL_TYPE

    if DENOISING_MODEL is None  or model_name not in str(DENOISING_MODEL.config._name_or_path):
        DENOISING_MODEL = None
        DENOISING_TOKENIZER = None
        MODEL_TYPE = ""
        if model_type == "causal":
            try:
                DENOISING_MODEL = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, device_map="auto"
                ).eval()
                MODEL_TYPE = "causal"
            except ValueError:
                raise ValueError(f" wrong model type for the model") from e
        elif model_type == "seq2seq":
            try:
                DENOISING_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=torch.float16, device_map="auto"
                ).eval()
                MODEL_TYPE = "seq2seq"
            except ValueError as e:
                raise ValueError(f"Model {model_name} is neither a causal LM nor a seq2seq model") from e
        

        DENOISING_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        if MODEL_TYPE == "causal":
            DENOISING_TOKENIZER.pad_token = DENOISING_TOKENIZER.eos_token
    return DENOISING_MODEL, DENOISING_TOKENIZER, MODEL_TYPE


def _split_text_with_overlap(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[TextChunk]:
    """Split text into overlapping chunks using a sliding window approach.

    Each chunk includes some context from the previous and next chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of TextChunk objects.
    """
    if len(text) <= chunk_size:
        return [TextChunk(text, 0, len(text))]

    chunks = []
    start = 0

    while start < len(text):
        # Calculate end position for this chunk
        end = min(start + chunk_size, len(text))

        # If this is not the last chunk, try to find a good break point
        if end < len(text):
            # Look for sentence boundary within overlap region
            break_point = text.rfind(". ", end - overlap, end)
            if break_point != -1:
                end = break_point + 2  # Include the period and space
            else:
                # If no sentence boundary, look for space
                break_point = text.rfind(" ", end - overlap, end)
                if break_point != -1:
                    end = break_point + 1
                else:
                    # If no good break point, use the middle of overlap
                    end = (end - overlap) // 2

        chunks.append(TextChunk(text[start:end], start, end))

        # Move start position, accounting for overlap
        start = end - overlap if end < len(text) else end

    return chunks


def _merge_overlapping_chunks(chunks: list[TextChunk]) -> str:
    """Merge overlapping chunks back together, handling overlaps intelligently.

    Args:
        chunks: List of TextChunk objects.

    Returns:
        Merged text.
    """
    if not chunks:
        return ""

    # Sort chunks by start position
    chunks.sort(key=lambda x: x.start)

    result = []
    current_pos = 0

    for chunk in chunks:
        if chunk.start > current_pos:
            # There's a gap, add the missing text
            result.append(chunk.text)
        else:
            # There's an overlap, find the best merge point
            overlap_start = chunk.start
            overlap_end = min(current_pos, chunk.end)

            # Find the best merge point in the overlap region
            # Look for sentence boundaries first
            merge_point = chunk.text.rfind(". ", 0, overlap_end - overlap_start)
            if merge_point != -1:
                merge_point += 2  # Include the period and space
            else:
                # If no sentence boundary, look for space
                merge_point = chunk.text.rfind(" ", 0, overlap_end - overlap_start)
                if merge_point != -1:
                    merge_point += 1
                else:
                    # If no good break point, use the middle of overlap
                    merge_point = (overlap_end - overlap_start) // 2
            # Add the non-overlapping part
            result.append(chunk.text[merge_point:])

        current_pos = chunk.end
    return " ".join(result)


def _denoise_chunk(chunk: TextChunk, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,model_type:Literal["causal,seq2seq"]) -> TextChunk:
    """Denoise a single chunk of text.

    Args:
        chunk: The chunk of text to denoise.
        model: The model to use for denoising.
        tokenizer: The tokenizer to use for denoising.

    Returns:
        The denoised chunk.
    """
    # Construct the prompt for denoising using a chat template
    if model_type == "causal":
        messages = [
            {
                "role": "user",
                "content": f"""clean and denoise the output of the optical character recognition(OCR) system. VERY IMPORTANT: Output only denoised text
                NOT OUTPUT anything else then the denoised text!.

                output of the optical character recognition(OCR) system to denoise:
                {chunk.text}
                """,
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.2,  # Lower temperature for more focused corrections
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_tokens = outputs[0][len(inputs.input_ids[0]) : ]
        cleaned_chunk = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
    elif model_type == "seq2seq":
        prompt =  f""" denoise the Output of the OCR system:
        "{chunk.text}"
        """
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            top_p=0.95,
            do_sample=True,
        )
        output_tokens = outputs[0][len(inputs.input_ids[0]) : ]
        cleaned_chunk =  tokenizer.decode(output_tokens, skip_special_tokens=True)


        # output_tokens = outputs[0][len(inputs.input_ids[0]) : ]
        # cleaned_chunk = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    # Decode and extract the cleaned text
    
    print('cleaned text \n')
    print(cleaned_chunk)
    return TextChunk(cleaned_chunk, chunk.start, chunk.end)


def denoise(
    text: str, model_name: str = "google/gemma-3-1b-it",model_type: str="causal", chunk_size: int | None = None, overlap: int = 100
) -> str:
    """
    Denoise OCR text using the choosen model .

    Args:
        text: The OCR text to be denoised.
        model_name: The name of the model to use for denoising.
        chunk_size: The size of the chunks to split the text into.
        overlap: The overlap between the chunks.

    Returns:
        The denoised text.
    """
    model, tokenizer,model_type= _load_model(model_name,model_type)
    if chunk_size is not None:
        text_chunks = _split_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
    else:
        text_chunks = [TextChunk(text, 0, len(text))]

    denoised_chunks = []
    for chunk in text_chunks:
        denoised_chunks.append(_denoise_chunk(chunk, model, tokenizer,model_type))
        

    # Merge the denoised chunks, handling overlaps
    return _merge_overlapping_chunks(denoised_chunks)


def denoise_dataset(
    noisy_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json",
    model_name: str = "google/gemma-3-1b-it",
    model_type: str = "causal",
    chunk_size: int | None = 1000,
    overlap: int = 100,
    subset: list[int] | None = None,
) -> str:
    """
    Denoise a dataset of text.

    The dataset is expected to be in the format of a JSON file with the following structure:
    {
        "1": "...",
        "2": "...",
        ...
    }

    The denoised data will be saved to the same

    Args:
        noisy_data_path: The path to the noisy dataset.

    Returns:
        The denoised text.
    """
    print(model_name)
    noisy_data = load_data(noisy_data_path)
    print(f"Loaded noisy data from {noisy_data_path}")

    if subset is not None:
        noisy_data = {k: v for k, v in noisy_data.items() if k in subset}
        print(f"Using subset of {len(noisy_data)} pages")

    denoised_data: dict[int, str] = {}
    for i in tqdm(noisy_data):
        noisy_text = noisy_data[i]
        denoised_data[i] = denoise(noisy_text, model_name=model_name,model_type=model_type, chunk_size=chunk_size, overlap=overlap)
    denoised_file_path = noisy_data_path.with_name(
        f"{noisy_data_path.stem}_denoised_{model_name_to_path_compatible(model_name)}{noisy_data_path.suffix}"
    )
    save_data(denoised_file_path, denoised_data)
    print(f"Denoised data saved to {denoised_file_path}")
    return denoised_data, denoised_file_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-it",
        help="The model  to perform the denoising"
    )
    parser.add_argument("--model_type",
                       type=str,
        default="causal",
        choices=["causal","seq2seq"],
        help="The model type to perform the denoising" )
    args = parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type
    do_blocking_hf_login()
    denoise_dataset(model_name=model_name,model_type=model_type)

