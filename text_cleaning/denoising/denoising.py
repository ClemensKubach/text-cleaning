from dataclasses import dataclass
from pathlib import Path
import logging
import re

import fire
from tqdm import tqdm
from transformers import AutoTokenizer, Pipeline

from text_cleaning.constants import DATA_DIR
from text_cleaning.utils import (
    do_blocking_hf_login,
    load_data,
    model_name_to_path_compatible,
    save_data,
    load_pipeline,
    setup_logging,
)
from typing import Literal


MAX_CONTEXT_TOKENS = 16000
MAX_NEW_TOKENS = MAX_CONTEXT_TOKENS // 2
DEFAULT_OVERLAP = 100


def get_in_context_messages(input_text: str, in_context: Literal["simple", "complex", "None"]) -> list[dict]:
    if in_context == "simple":
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that corrects noisy OCR text. "
                    "Your task is to fix common OCR character substitution errors. "
                    "The most frequent mistakes found in the data include:\n"
                    "- 't' often misread as 'l'\n"
                    "- 'h' often misread as 'b'\n"
                    "- 'n'often misread as'r'\n"
                    "- 'e' often misread as 'c'\n"
                    "- 'e' often misread as 'o'\n"
                    "- deletion of letters and spaces\n"
                    "- addition of letters and spaces \n"
                    "- insertion of non alpha-numeric symbols such as: 'ſ' "
                    "These are common OCR issues—use this knowledge to guide corrections."
                ),
            },
            # Example 1: t → l
            {
                "role": "user",
                "content": "Example mistake: 't' was misread as 'l'\nInput: The lalbes were fulled and lisled correctly.",
            },
            {"role": "assistant", "content": "The tables were filled and titled correctly."},
            # Example 2: h → b
            {
                "role": "user",
                "content": "Example mistake: 'h' was misread as 'b'\nInput: The beavy bag bung from the book.",
            },
            {"role": "assistant", "content": "The heavy bag hung from the hook."},
            # Example 3: n → r
            {
                "role": "user",
                "content": "Example mistake: 'n' was misread as 'r'\nInput: The rurse walked dowr the rarrow hallway.",
            },
            {"role": "assistant", "content": "The nurse walked down the narrow hallway."},
            # Example 4: e → c
            {
                "role": "user",
                "content": "Example mistake: 'e' was misread as 'c'\nInput: The accnomy cxperienced a rcccssion.",
            },
            {"role": "assistant", "content": "The economy experienced a recession."},
            # Example 5: e → o
            {
                "role": "user",
                "content": "Example mistake: 'e' was misread as 'o'\nInput: Tho dog jompod ovor tho fonco and ran around.",
            },
            {"role": "assistant", "content": "The dog jumped over the fence and ran around."},
            # Actual test task
            {"role": "user", "content": f"{input_text}"},
        ]
    elif in_context == "complex":
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that corrects noisy OCR text. "
                    "Your task is to fix OCR errors. The most common issues include character substitutions such as 't' misread as 'l', "
                    "'h' as 'b', 'e' as 'o', and other similar letter substitutions."
                    " Other errors include deletion or insertion of characters, deletion or addition of spaces,"
                    "insertion of non-alphanumeric symbols such as 'ſ', and visual-level confusions such as 'rn' misread as 'm'. "
                    "Use the examples to guide your correction."
                ),
            },
            # Example 1 — t→l, h→b, o→a
            {"role": "user", "content": "Input: The bla lelped wi h loca lasks arou d the ne ghborhood."},
            {"role": "assistant", "content": "The bat helped with local tasks around the neighborhood."},
            # Example 2 — n→r, e→c, i→j
            {"role": "user", "content": "Input: The rcjghborhood cxperience was diffcrent for everyorje."},
            {"role": "assistant", "content": "The neighborhood experience was different for everyone."},
            # Example 3 — e→o, space→missing, i→l
            {"role": "user", "content": "Input: Thoquickbrownfoxjumpodovorafoncoinono go."},
            {"role": "assistant", "content": "The quick brown fox jumped over a fence in one go."},
            # Example 4 — o→0, u→v, a→c
            {"role": "user", "content": "Input: The c0mpvtcr scftw4re has 0ver 100 featvrcs."},
            {"role": "assistant", "content": "The computer software has over 100 features."},
            # Example 5 — f→s, s→f,'ſ' inserted, space→extra space
            {
                "role": "user",
                "content": "Input: The ſlow fteps of the fcouts  sfilled the foreſt with disturbirmg ſound.",
            },
            {
                "role": "assistant",
                "content": "The slow steps of the scouts filled the forest with disturbing sound.",
            },
            # Your test input goes here
            {"role": "user", "content": f"Input: {input_text}"},
        ]
    else:
        raise ValueError(f"In-context learning type {in_context} not supported")


logger = logging.getLogger(__name__)


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


def _split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
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
    else:
        logger.info(f"Splitting text of length {len(text)} into chunks of size {chunk_size} with overlap {overlap}")

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


def extract_denoised_text(generated_text: str) -> str | None:
    """Extract the denoised text from the generated text.
    If the denoised text is not found, return None.
    """
    match = re.search(r"<denoised>(.*?)</denoised>", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _denoise_chunk(
    chunk: TextChunk,
    text_pipeline: Pipeline,
    tokenizer: AutoTokenizer,
    model_type: Literal["causal", "seq2seq"],
    in_context: Literal["simple", "complex", "None"] = "None",
    max_attempts: int = 3,
) -> TextChunk:
    """Denoise a single chunk of text.

    Args:
        chunk: The chunk of text to denoise.
        text_pipeline: The text generation pipeline to use for denoising.
        tokenizer: The tokenizer to use for denoising.
        model_type: The type of model being used.

    Returns:
        The denoised chunk.
    """
    is_instruction_model = (
        model_type == "causal" and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )
    output_marker = "\n\nDenoised text:"
    if is_instruction_model:
        if in_context == "simple" or in_context == "complex":
            message = get_in_context_messages(chunk.text, in_context)
            prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,  # Get the formatted string, not tokenized input
                add_generation_prompt=True,  # Optional: Adds assistant turn prefix if needed
            )
        else:
            # For instruction-tuned models, use chat template if available
            instruction_prompt = (
                "Clean and denoise the given noisy text received from an optical character recognition (OCR) system. "
                "VERY IMPORTANT: Output ONLY the denoised version of the text in the format <denoised>{denoised_text}</denoised>. "
                "Do NOT output anything else. "
                f"Given noisy text: {chunk.text}"
            )
            prompt = [{"role": "user", "content": instruction_prompt}]
    elif model_type == "seq2seq":
        prompt = chunk.text
    else:
        prompt = f"Noisy text: {chunk.text}{output_marker}"

    for _ in range(max_attempts):
        outputs = text_pipeline(prompt, max_new_tokens=MAX_NEW_TOKENS)
        generated_text = outputs[0]["generated_text"]

        # For base models, we need to clean up the output more carefully
        if is_instruction_model:
            last_model_response = generated_text[-1]["content"]
            denoised_chunk_text = extract_denoised_text(last_model_response)
            # # Split the output by the model's turn marker
            # model_turns = generated_text.split("<start_of_turn>model")
            # # Take the last occurrence (most recent model response)
            # last_model_turn = model_turns[-1]
            # # Remove trailing special tokens if needed
            # denoised_chunk_text = last_model_turn.split("<end_of_turn>")[0].strip()
            # print(denoised_chunk_text)
        else:
            # Extract text after the last "Output:" marker
            if output_marker in generated_text:
                denoised_chunk_text = generated_text.split(output_marker)[-1].strip()
            else:
                # Fallback to whole text if marker not found
                denoised_chunk_text = generated_text.strip()
        if denoised_chunk_text is not None:
            logger.info(f"Denoised chunk: {denoised_chunk_text}")
            return TextChunk(denoised_chunk_text, chunk.start, chunk.end)
    logger.warning(f"Failed to denoise chunk {chunk.text} after {max_attempts} attempts. Returning the generated text.")
    return TextChunk(last_model_response, chunk.start, chunk.end)


def denoise(
    text: str,
    model_name: str = "google/gemma-3-1b-it",
    model_type: Literal["causal", "seq2seq"] = "causal",
    chunk_size: int | None = MAX_CONTEXT_TOKENS,
    overlap: int = DEFAULT_OVERLAP,
    in_context: Literal["simple", "complex", "None"] = "None",
) -> str:
    """
    Denoise OCR text using the chosen model.
    Args:
        text: The OCR text to be denoised.
        model_name: The name of the model to use for denoising.
        model_type: The type of model to use for denoising (causal or seq2seq).
        chunk_size: The size of the chunks to split the text into.
        overlap: The overlap between the chunks.

    Returns:
        The denoised text.
    """
    text_pipeline, tokenizer, model_type = load_pipeline(model_name, model_type)
    if chunk_size is not None:
        text_chunks = _split_text_with_overlap(text, chunk_size=chunk_size, overlap=overlap)
    else:
        text_chunks = [TextChunk(text, 0, len(text))]

    denoised_chunks = []
    for chunk in text_chunks:
        denoised_chunks.append(_denoise_chunk(chunk, text_pipeline, tokenizer, model_type, in_context))

    # Merge the denoised chunks, handling overlaps
    return _merge_overlapping_chunks(denoised_chunks)


def denoise_dataset(
    noisy_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json",
    model_name: str = "google/gemma-3-1b-it",
    model_type: Literal["causal", "seq2seq"] = "causal",
    in_context: Literal["simple", "complex", "None"] = "None",
    chunk_size: int | None = MAX_CONTEXT_TOKENS,
    overlap: int = DEFAULT_OVERLAP,
    subset: list[int] | None = None,
) -> tuple[dict[int, str], Path]:
    """
    Denoise a dataset of text.
    The dataset is expected to be in the format of a JSON file with the following structure:
    {
        "1": "...",
        "2": "...",
        ...
    }
    Args:
        noisy_data_path: The path to the noisy dataset.
        model_name: The name of the model to use for denoising.
        model_type: The type of the model to use for denoising (causal or seq2seq).
        chunk_size: The size of the chunks to split the text into.
        overlap: The overlap between the chunks.
        subset: The subset of the data to denoise.

    Returns:
        A tuple containing:
        - The denoised data dictionary
        - The path where the denoised data was saved
    """
    noisy_data = load_data(noisy_data_path)
    logger.info(f"Loaded noisy data from {noisy_data_path}")

    if subset is not None:
        noisy_data = {k: v for k, v in noisy_data.items() if k in subset}
        logger.info(f"Using subset of {len(noisy_data)} pages")

    denoised_data: dict[int, str] = {}
    for i in tqdm(noisy_data):
        noisy_text = noisy_data[i]
        denoised_data[i] = denoise(
            noisy_text,
            model_name=model_name,
            model_type=model_type,
            chunk_size=chunk_size,
            overlap=overlap,
            in_context=in_context,
        )
    denoised_file_path = noisy_data_path.with_name(
        f"{noisy_data_path.stem}_denoised_{model_name_to_path_compatible(model_name)}{noisy_data_path.suffix}"
    )
    save_data(denoised_file_path, denoised_data)
    logger.info(f"Denoised data saved to {denoised_file_path}")
    return denoised_data, denoised_file_path


if __name__ == "__main__":
    setup_logging()
    do_blocking_hf_login()
    fire.Fire(denoise_dataset, serialize=lambda _: None)
