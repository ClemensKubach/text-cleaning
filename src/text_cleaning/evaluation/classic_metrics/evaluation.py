from collections.abc import Callable
from pathlib import Path
import nltk
import logging
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
import Levenshtein
from src.text_cleaning.constants import DATA_DIR, DENOISED_DIR
from src.text_cleaning.utils import load_data, save_data
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch


from dotenv import load_dotenv
from src.text_cleaning.utils import setup_logging
from evaluate import load


"""'tokenizer download for splitting the text on the words properly"""

logger = logging.getLogger(__name__)

nltk.download("punkt_tab")








def evaluate_all(clean_text: str, noisy_text: str) -> dict:
    """Evaluate noisy text against clean reference using ROUGE-1, CER, and WER metrics.
    
    Args:
        clean_text: Ground truth reference text.
        noisy_text: OCR output or denoised text to evaluate.
    
    Returns:
        Dictionary with precision, recall, F1 (ROUGE-1), CER, and WER scores.
    """
    # Initialize metrics
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    cer_metric = load("cer")
    wer_metric = load("wer")

    # Compute ROUGE-1 (handles empty strings)
    scores = scorer.score(target=clean_text, prediction=noisy_text)
    rouge1 = scores["rouge1"]
    length_clean = len(clean_text)
    length_ocr = len(noisy_text)
    
    # Compute CER/WER (wrap in lists as required by HF evaluate)
    cer_score = cer_metric.compute(predictions=[noisy_text], references=[clean_text])
    wer_score = wer_metric.compute(predictions=[noisy_text], references=[clean_text])

    return {
        "unigram_precision": rouge1.precision,
        "unigram_recall": rouge1.recall,
        "unigram_f1": rouge1.fmeasure,
        "cer": cer_score,
        "wer": wer_score,
        "length_clean":length_clean,
        "length_ocr":length_ocr
    }




"""method to evaluate the CER - error rate in terms of the character operation to character number ratio"""


# def evaluate_CER(clean_text: str, noisy_text: str) -> float:
#     # divisor = len(clean_text)
#     # print(divisor)
#     # return count_all_operations(clean_text, noisy_text) / divisor
#     cer_metric = load("cer")
    

#     return cer(noisy_text,clean_text)


"""method to evaluate the WER - error rate in terms of the character operation to words number ratio"""


# def evaluate_WER(clean_text: str, noisy_text: str) -> float:
#     # tokens = word_tokenize(clean_text)
#     # divisor = len(tokens)
#     # return count_all_operations(clean_text, noisy_text) / divisor
#     wer_metric = load("wer")
#     return wer(noisy_text, clean_text)


"""perplexity will be evaluated, empty for now"""


"""the main pipeline for evaluating the cleaning chunk by chunk  """


def evaluate_dataset(
    evaluation_method: Callable[[str, str], float],
    denoised_data_name: str,
    noisy_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json",
    cleaned_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_clean.json",
    evaluation_task: str = "single",
    model_path: str = None,
) -> dict[int, float]:
    """Evaluate the denoised data.

    Args:
        noisy_data_path: The path to the given noisy dataset.
        cleaned_data_path: The path to the clean ground truth dataset.
        denoised_data_path: The path to the generated denoised dataset.
    Returns:
        The scores.
    """
    denoised_data_path = DENOISED_DIR / denoised_data_name
    noisy_data = load_data(noisy_data_path)
    logger.info(f"Loaded noisy data from {noisy_data_path}")
    clean_data = load_data(cleaned_data_path)
    logger.info(f"Loaded clean data from {cleaned_data_path}")
    denoised_data = load_data(denoised_data_path)
    logger.info(f"Loaded denoised data from {denoised_data_path}")

    scores: dict[int, float] = {}
    for i in tqdm(denoised_data):
        noisy_text = noisy_data[i]
        clean_text = clean_data[i]
        denoised_text = denoised_data[i]
        if evaluation_task == "single":
            score = evaluation_method(clean_text, denoised_text)
            scores[i] = score


    model_identifier = denoised_data_name.split('/')[-1]
    scores_file_path  = DATA_DIR / "evaluation_scores" / "classic_evaluations"/ model_identifier
    
   
    save_data(scores_file_path, scores)
    logger.info(f"Scores saved to {scores_file_path}")
    return scores, scores_file_path


if __name__ == "__main__":
    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        default="ALL",
        choices=["ALL"],
        help="Evaluation metric to use",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="single",
        choices=["single"],
        help="Whether to evaluate denoised output against the clean text, or compare denoised and noisy output ",
    )

    parser.add_argument(
        "--denoised_data_name",
        type=str,
        required=True,
        help="the path to the denoised output",
    )
    


    args = parser.parse_args()

    metric_map = {
        "ALL" : evaluate_all
    }

    evaluation_method = metric_map[args.metric]
    evaluation_task = args.task
    denoised_data_name = args.denoised_data_name
    evaluate_dataset(
        evaluation_method=evaluation_method, denoised_data_name=denoised_data_name, evaluation_task=evaluation_task
    )
