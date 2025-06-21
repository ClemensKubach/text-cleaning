from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.pipelines.base import Pipeline
from collections import Counter

from text_cleaning.constants import DATA_DIR
from text_cleaning.utils import (
    load_data,
    model_name_to_path_compatible,
    save_data,
    load_pipeline,
)
from typing import Literal


MAX_CONTEXT_TOKENS = 16000
MAX_NEW_TOKENS = MAX_CONTEXT_TOKENS // 2
DEFAULT_OVERLAP = 100

FILE_WRITE_CHUNK_STATS = DATA_DIR / "statistical_analysis" / "chunk_stats"
FILE_WRITE_CHUNK_STATS.mkdir(parents=True, exist_ok=True)


def get_in_context_messages(
    input_text: str, in_context: Literal["simple", "complex", "None"], is_finetuned: bool = False
) -> list[dict]:
    """Get the messages for optional in-context learning.

    Args:
        input_text: The text to be denoised.
        in_context: The type of in-context learning to use.
        is_finetuned: Whether the model is finetuned or not.

    Returns:
        The messages for optional in-context learning.
    """
    if is_finetuned:
        instruction_prompt = "Clean the following OCR text:"
    else:
        instruction_prompt = (
            "Clean and denoise the noisy text you will receive. It is determined from an optical character recognition (OCR) system. "
            "Denoise the text word by word or character by character to existing words. Do not change grammar, word order or meaning of the text. "
            "VERY IMPORTANT: Output ONLY the denoised version of the text. "
            "Do NOT output anything else. "
        )
    if in_context == "None":
        return [{"role": "system", "content": instruction_prompt}, {"role": "user", "content": input_text}]
    elif in_context == "simple":
        return [
            {
                "role": "system",
                "content": (
                    instruction_prompt + "\n" + "The most frequent mistakes found in the data include:\n"
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
                    instruction_prompt + "\n" + "You are a helpful assistant that corrects noisy OCR text. "
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


def _split_text_into_sentences(text: str, use_sentence_chunks: bool = True) -> list[TextChunk]:
    """Split text into chunks by sentences or return as single chunk.

    Args:
        text: The text to split.
        use_sentence_chunks: If True, splits at sentence boundaries. If False, returns whole text as one chunk.

    Returns:
        List of TextChunk objects. Either one per sentence or a single chunk for the whole text.
    """
    if not use_sentence_chunks:
        return [TextChunk(text, 0, len(text))]

    # Split by sentences and keep track of positions
    current_pos = 0
    chunks = []

    # Handle the case where there are no sentence boundaries
    if ". " not in text:
        return [TextChunk(text, 0, len(text))]

    # Split and process each sentence
    sentences = text.split(". ")
    for i, sentence in enumerate(sentences):
        # Add the period back except for the last sentence
        sentence_full = sentence + "." if i < len(sentences) - 1 else sentence
        sentence_len = len(sentence_full)

        chunks.append(TextChunk(sentence_full, current_pos, current_pos + sentence_len))
        current_pos += sentence_len

    logger.info(f"Split text into {len(chunks)} chunks by sentences")

    # Verify no text is lost
    reconstructed = _merge_chunks(chunks)
    if len(reconstructed) != len(text):
        logger.warning(f"Text length mismatch! Original: {len(text)}, Reconstructed: {len(reconstructed)}")
        return [TextChunk(text, 0, len(text))]

    return chunks


def _merge_chunks(chunks: list[TextChunk]) -> str:
    """Merge chunks back together.

    Since chunks are individual sentences, we just need to concatenate them
    in the correct order.

    Args:
        chunks: List of TextChunk objects.

    Returns:
        Merged text.
    """
    if not chunks:
        return ""

    # Sort chunks by start position to ensure correct order
    chunks.sort(key=lambda x: x.start)

    # Simply concatenate the chunks as they're already complete sentences
    return " ".join(chunk.text for chunk in chunks)


def _get_majority_vote_text(denoised_attempts: list[str]) -> str:
    """Get the majority vote for each word position across multiple denoising attempts.

    Args:
        denoised_attempts: List of denoised text versions.

    Returns:
        Text constructed from the most common word at each position.
    """
    # Split each attempt into words
    word_sequences = [attempt.split() for attempt in denoised_attempts]

    # Find the most common length to handle different lengths
    lengths = [len(seq) for seq in word_sequences]
    most_common_length = Counter(lengths).most_common(1)[0][0]

    # Filter sequences to only those with the most common length
    valid_sequences = [seq for seq in word_sequences if len(seq) == most_common_length]
    if not valid_sequences:
        # If no valid sequences (shouldn't happen with at least one attempt), return first attempt
        return denoised_attempts[0]

    # For each word position, find the most common word
    result_words = []
    for pos in range(most_common_length):
        words_at_pos = [seq[pos] for seq in valid_sequences]
        most_common_word = Counter(words_at_pos).most_common(1)[0][0]
        result_words.append(most_common_word)

    return " ".join(result_words)


def _denoise_chunk(
    chunk: TextChunk,
    text_pipeline: Pipeline,
    tokenizer: AutoTokenizer,
    model_type: Literal["causal", "seq2seq"],
    in_context: Literal["simple", "complex", "None"] = "None",
    num_attempts: int = 1,
    is_finetuned: bool = False,
) -> TextChunk:
    """Denoise a single chunk of text, optionally with multiple attempts and majority voting.

    Args:
        chunk: The chunk of text to denoise.
        text_pipeline: The text generation pipeline to use for denoising.
        tokenizer: The tokenizer to use for denoising.
        model_type: The type of model being used.
        in_context: The type of in-context learning to use.
        num_attempts: Number of denoising attempts for majority voting (default=1).
        is_finetuned: Whether the model is finetuned or not.

    Returns:
        The denoised chunk, if num_attempts=1 returns the direct output,
        if num_attempts>1 returns the majority vote across attempts.
    """
    is_instruction_model = False
    output_marker = "\n\nDenoised text:"

    denoised_attempts = []
    for _ in range(num_attempts):
        if model_type == "causal":
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:  # type: ignore
                is_instruction_model = True
                messages = get_in_context_messages(chunk.text, in_context, is_finetuned)
                prompt = tokenizer.apply_chat_template(  # type: ignore
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # for base models, we need a prompt that suggests to continue the text
                prompt = f"Noisy text: {chunk.text}{output_marker}"
        elif model_type == "seq2seq":
            prompt = chunk.text
        else:
            raise ValueError(f"Model type {model_type} not supported")

        outputs = text_pipeline(prompt, max_new_tokens=MAX_NEW_TOKENS)
        generated_text = outputs[0]["generated_text"].strip()  # type: ignore
        if not is_instruction_model:
            generated_text = generated_text.split(output_marker)[-1].strip()
        denoised_attempts.append(generated_text)

    # If only one attempt, return it directly
    if num_attempts == 1:
        final_text = denoised_attempts[0]
    else:
        # Otherwise, use majority voting
        final_text = _get_majority_vote_text(denoised_attempts)

    return TextChunk(final_text, chunk.start, chunk.end)


def denoise(
    text: str,
    model_name: str = "google/gemma-3-1b-it",
    model_type: Literal["causal", "seq2seq"] = "causal",
    in_context: Literal["simple", "complex", "None"] = "None",
    use_sentence_chunks: bool = True,
    num_attempts: int = 1,
    is_finetuned: bool = False,
) -> str:
    """
    Denoise OCR text using the chosen model.
    Args:
        text: The OCR text to be denoised.
        model_name: The name of the model to use for denoising.
        model_type: The type of model to use for denoising (causal or seq2seq).
        in_context: The type of in-context learning to use.
        use_sentence_chunks: If True, processes text sentence by sentence. If False, processes whole text at once.
        num_attempts: Number of denoising attempts for majority voting (default=1).
        is_finetuned: Whether the model is finetuned or not.

    Returns:
        The denoised text.
    """
    text_pipeline, tokenizer, model_type = load_pipeline(model_name, model_type)
    text_chunks = _split_text_into_sentences(text, use_sentence_chunks)

    denoised_chunks = []
    for chunk in text_chunks:
        denoised_chunks.append(
            _denoise_chunk(chunk, text_pipeline, tokenizer, model_type, in_context, num_attempts, is_finetuned)
        )

    # Merge the denoised chunks
    merged_text = _merge_chunks(denoised_chunks)
    logger.info(f"Generated text: {merged_text}")
    return merged_text


def denoise_dataset(
    noisy_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json",
    model_name: str = "google/gemma-3-1b-it",
    model_type: Literal["causal", "seq2seq"] = "causal",
    in_context: Literal["simple", "complex", "None"] = "None",
    use_sentence_chunks: bool = True,
    num_attempts: int = 1,
    subset: list[int] | None = None,
    is_finetuned: bool = False,
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
        in_context: The type of in-context learning to use.
        use_sentence_chunks: If True, processes text sentence by sentence. If False, processes whole text at once.
        num_attempts: Number of denoising attempts for majority voting (default=1).
        subset: The subset of the data to denoise.
        is_finetuned: Whether the model is finetuned or not.

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
            in_context=in_context,
            use_sentence_chunks=use_sentence_chunks,
            num_attempts=num_attempts,
            is_finetuned=is_finetuned,
        )
    denoised_file_path = noisy_data_path.with_name(
        f"{noisy_data_path.stem}_denoised_{model_name_to_path_compatible(model_name)}{noisy_data_path.suffix}"
    )
    save_data(denoised_file_path, denoised_data)
    logger.info(f"Denoised data saved to {denoised_file_path}")
    return denoised_data, denoised_file_path
