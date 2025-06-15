from pathlib import Path
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
import Levenshtein
from text_cleaning.constants import DATA_DIR
from text_cleaning.utils import load_data, save_data
from collections import Counter
import argparse

nltk.download('punkt')

def evaluate_letter_precision(clean_text: str, noisy_text: str) -> float:
    i = 0
    matched = 0
    clean_len = len(clean_text)
    noisy_len = len(noisy_text)
    print(noisy_text)
    while i < clean_len and i < noisy_len:
        if clean_text[i] == noisy_text[i]:
            matched +=1
        i +=1
    return matched/noisy_len


def evaluate_ROGUE( clean_text: str, noisy_text: str) -> float:
    """Just a placeholder for know. Will be called from main.py.

    Args:
        noisy_text: The given noisy text.
        denoised_text: The generated denoised text.
        clean_text: The clean ground truth text.

    Returns:
        The score.

    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(clean_text,noisy_text)
    prec = scores['rouge1'].precision
    rec =scores['rouge1'].recall
    f1 = scores['rouge1'].fmeasure
    return (prec,rec,f1)
def evaluate_BLUE(clean_text: str, noisy_text: str) -> float:
    noisy_text = word_tokenize(noisy_text)
    clean_text = word_tokenize(clean_text)
    blue_score = sentence_bleu(clean_text,noisy_text)
    return blue_score

def count_all_operations(clean_text: str, noisy_text: str) -> int:
    i, j = 0, 0
    ops_count = 0
    gt_words = clean_text.split()
    ocr_words = noisy_text.split()

    while i < len(gt_words) and j < len(ocr_words):
        gt_word = gt_words[i]
        ocr_word = ocr_words[j]

        direct_dist = Levenshtein.distance(gt_word, ocr_word)
        merge_dist = float('inf')
        split_dist = float('inf')

        if len(gt_word) > len(ocr_word) and j + 1 < len(ocr_words):
            merged_ocr = ocr_word + ocr_words[j + 1]
            merge_dist = Levenshtein.distance(gt_word, merged_ocr)

        elif len(ocr_word) > len(gt_word) and i + 1 < len(gt_words):
            merged_gt = gt_word + gt_words[i + 1]
            split_dist = Levenshtein.distance(merged_gt, ocr_word)

        if direct_dist <= merge_dist and direct_dist <= split_dist:
            current_ops = Levenshtein.editops(gt_word, ocr_word)
            print(gt_word,ocr_word)
            print(len(current_ops))
            print('\n')
            ops_count += len(current_ops)
            i += 1
            j += 1

        elif merge_dist < split_dist:
            current_ops = Levenshtein.editops(gt_word, merged_ocr)
            print(gt_word,ocr_word)
            print(len(current_ops))
            print('\n')
            ops_count += len(current_ops)
            i += 1
            j += 2

        else:
            current_ops = Levenshtein.editops(merged_gt, ocr_word)
            print(gt_word,ocr_word)
            print(len(current_ops))
            print('\n')
            ops_count += len(current_ops)
            i += 2
            j += 1
    
    ops_count += sum(len(w) for w in gt_words[i:])  # remaining GT tokens, subtraction
    ops_count += sum(len(w) for w in ocr_words[j:]) # remaining OCR tokens, subtraction
    return ops_count

def evaluate_CER(clean_text:str,noisy_text:str) -> float:
    divisor = len(clean_text)
    print(divisor)
    return count_all_operations(clean_text,noisy_text)/divisor
def evaluate_WER(clean_text: str, noisy_text: str) -> float:
    tokens = word_tokenize(clean_text)
    divisor = len(tokens)
    return count_all_operations(clean_text,noisy_text)/divisor
def evaluate_perplexity(noisy_text: str, denoised_text: str, model) -> float:
    pass
def evaluate_Bert(noisy_text: str, denoised_text: str, model) -> float:
    pass



def evaluate_dataset(
    evaluation_method,
    denoised_data_path: Path,
    noisy_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json",
    cleaned_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_clean.json",
) -> dict[int, float]:
    """Evaluate the denoised data.

    Args:
        noisy_data_path: The path to the given noisy dataset.
        cleaned_data_path: The path to the clean ground truth dataset.
        denoised_data_path: The path to the generated denoised dataset.

    Returns:
        The scores.
    """
    noisy_data = load_data(noisy_data_path)
    print(f"Loaded noisy data from {noisy_data_path}")
    clean_data = load_data(cleaned_data_path)
    print(f"Loaded clean data from {cleaned_data_path}")
    denoised_data = load_data(denoised_data_path)
    print(f"Loaded denoised data from {denoised_data_path}")

    scores: dict[int, float] = {}
    for i in tqdm(denoised_data):
        noisy_text = noisy_data[i]
        clean_text = clean_data[i]
        denoised_text = denoised_data[i]
        score = evaluation_method(clean_text, noisy_text)
        scores[i] = score
    scores_file_path = denoised_data_path.with_name(f"{noisy_data_path.stem}_scores{noisy_data_path.suffix}")
    save_data(scores_file_path, scores)
    print(f"Scores saved to {scores_file_path}")
    return scores, scores_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        default="CER",
        choices=["CER", "WER", "BLUE", "ROGUE","LP"],
        help="Evaluation metric to use"
    )
    args = parser.parse_args()

    metric_map = {
        "CER": evaluate_CER,
        "WER": evaluate_WER,
        "BLUE": evaluate_BLUE,
        "ROGUE": evaluate_ROGUE,
        "LP": evaluate_letter_precision
    }

    evaluation_method = metric_map[args.metric]

    denoised_data_path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json"
    evaluate_dataset(evaluation_method=evaluation_method, denoised_data_path=denoised_data_path)
