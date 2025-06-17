from collections.abc import Callable
from pathlib import Path
import tokenize
import nltk
import logging
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
import Levenshtein
from text_cleaning.constants import DATA_DIR
from text_cleaning.utils import load_data, save_data
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
import os
import torch



''''tokenizer download for splitting the text on the words properly'''

logger = logging.getLogger(__name__)

nltk.download("punkt_tab")



'''global variable MODEL_PATH, holding the path to the model that we would want to evaluate the perplexity on
for now set to the Mistral 7B model'''

MODEL_PATH: str = "mistralai/Mistral-7B-v0.1"


'''setting the model path'''
def set_global_var(model_path):
    global MODEL_PATH
    MODEL_PATH = model_path

'''loading, architecture-agnostically, the model '''
def load_model_and_tokenizer() -> [AutoModelForCausalLM, AutoTokenizer]:
    # Load the config to inspect model type
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # Load tokenizer (doesn't depend on LM head type)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Decide on model head based on architecture
    model_type = config.architectures[0] if config.architectures else ""
    
    if "CausalLM" in model_type or "GPT2" in model_type or "RWForCausalLM" in model_type:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")
    
    return model, tokenizer

"""auxillary function to compute the perplexity """
def compute_perplexity(model, tokenizer, input_text)->float:
    try:
        # Causal LM expects only input text
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(model.device)
        normalization_factor =  input_ids.shape[1]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item() / normalization_factor
    except:
        raise ValueError(f"Unsupported model type:")
    
"""perplexity evaluation function"""
def evaluate_perplexity(noisy_text: str, denoised_text) -> float:
    model, tokenizer = load_model_and_tokenizer()
    normalized_perplexity = compute_perplexity(model,tokenizer,denoised_text)
    return normalized_perplexity


    


'''evaluate direct letter matching, simple metric not really suitable due to many possible ocr system deletion or 
addition of the  letters '''
def evaluate_letter_precision(clean_text: str, noisy_text: str) -> float:
    i = 0
    matched = 0
    clean_len = len(clean_text)
    noisy_len = len(noisy_text)
    logger.debug(noisy_text)
    while i < clean_len and i < noisy_len:
        if clean_text[i] == noisy_text[i]:
            matched += 1
        i += 1
    return matched / noisy_len

'''function to evaluate the improvement between two texts in the 
terms of number of operations in the  edit distance '''
def evaluate_improvement(clean_text: str, ocr_text: str, denoised_text: str) -> float:
        return (count_all_operations(clean_text,ocr_text) - count_all_operations(clean_text,denoised_text))/len(clean_text)
        
'''the evaluation of the improvement between two terms in the of the choosen metric '''
def evaluate_metric_improvement(clean_text: str, ocr_text: str, denoised_text: str, evaluation_method: str) -> float:
    metric_map = {
        "CER": evaluate_CER,
        "WER": evaluate_WER,
        #"BLUE": evaluate_BLUE,
        "ROGUE": evaluate_ROGUE,
        "LP": evaluate_letter_precision
    }
    print("name of the evaluation function")
    print(evaluation_method.__name__)
    improvement_ratio = None
    '''the gain function , i.e the bigger score the better '''
    if evaluation_method.__name__ in ["evaluate_ROGUE", "evaluate_letter_precision"]:
        try:
            improvement_ratio =   evaluation_method(clean_text,denoised_text)/evaluation_method(clean_text, ocr_text)
        except ZeroDivisionError:
            print("no possible evaluation")

    
    
    elif evaluation_method.__name__ in ["evaluate_CER", "evaluate_WER"]:
        try:
            improvement_ratio = evaluation_method(clean_text, ocr_text) / evaluation_method(clean_text,denoised_text)
        except ZeroDivisionError:
            print("no possible evaluation")

    return improvement_ratio

''''evaluating the precision, recall and f1 score between the ground_truth and predicted_text'''
def evaluate_ROGUE( clean_text: str, noisy_text: str) -> float:
    
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(clean_text,noisy_text)
    prec = scores['rouge1'].precision
    rec =scores['rouge1'].recall
    f1 = scores['rouge1'].fmeasure
    dict_to_return = {"precision":prec,"recall":rec,"f1":f1}
    return dict_to_return



'''auxillry method to count operations needed to match two strings '''

def count_all_operations(clean_text: str, noisy_text: str) -> int:
    i, j = 0, 0
    ops_count = 0
    gt_words = clean_text.split()
    ocr_words = noisy_text.split()

    while i < len(gt_words) and j < len(ocr_words):
        gt_word = gt_words[i]
        ocr_word = ocr_words[j]

        direct_dist = Levenshtein.distance(gt_word, ocr_word)
        merge_dist = float("inf")
        split_dist = float("inf")

        if len(gt_word) > len(ocr_word) and j + 1 < len(ocr_words):
            merged_ocr = ocr_word + ocr_words[j + 1]
            merge_dist = Levenshtein.distance(gt_word, merged_ocr)

        elif len(ocr_word) > len(gt_word) and i + 1 < len(gt_words):
            merged_gt = gt_word + gt_words[i + 1]
            split_dist = Levenshtein.distance(merged_gt, ocr_word)

        if direct_dist <= merge_dist and direct_dist <= split_dist:
            current_ops = Levenshtein.editops(gt_word, ocr_word)
            logger.debug(f"{gt_word} {ocr_word}")
            logger.debug(f"Operations: {len(current_ops)}")
            logger.debug("")
            ops_count += len(current_ops)
            i += 1
            j += 1

        elif merge_dist < split_dist:
            current_ops = Levenshtein.editops(gt_word, merged_ocr)
            logger.debug(f"{gt_word} {ocr_word}")
            logger.debug(f"Operations: {len(current_ops)}")
            logger.debug("")
            ops_count += len(current_ops)
            i += 1
            j += 2

        else:
            current_ops = Levenshtein.editops(merged_gt, ocr_word)
            logger.debug(f"{gt_word} {ocr_word}")
            logger.debug(f"Operations: {len(current_ops)}")
            logger.debug("")
            ops_count += len(current_ops)
            i += 2
            j += 1

    ops_count += sum(len(w) for w in gt_words[i:])  # remaining GT tokens, subtraction
    ops_count += sum(len(w) for w in ocr_words[j:])  # remaining OCR tokens, subtraction
    return ops_count


'''method to evaluate the CER - error rate in terms of the character operation to character number ratio'''
def evaluate_CER(clean_text:str,noisy_text:str) -> float:
    divisor = len(clean_text)
    print(divisor)
    return count_all_operations(clean_text,noisy_text)/divisor
'''method to evaluate the WER - error rate in terms of the character operation to words number ratio'''
def evaluate_WER(clean_text: str, noisy_text: str) -> float:
    tokens = word_tokenize(clean_text)
    divisor = len(tokens)
    return count_all_operations(clean_text,noisy_text)/divisor
'''perplexity will be evaluated, empty for now'''


'''the main pipeline for evaluating the cleaning chunk by chunk  '''

def evaluate_dataset(
    evaluation_method: Callable[[str, str], float],
    denoised_data_path: Path,
    noisy_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json",
    cleaned_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_clean.json",
    evaluation_task: str = 'single',
    model_path: str = None
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
        if evaluation_task  == "single":
            score = evaluation_method(clean_text, denoised_text)
            scores[i] = score
        elif evaluation_task == "comparative":
            score = evaluate_metric_improvement(clean_text,noisy_text,denoised_text,evaluation_method)
            scores[i] = score
    scores_file_path = denoised_data_path.with_name(f"{noisy_data_path.stem}_scores{noisy_data_path.suffix}")
    save_data(scores_file_path, scores)
    logger.info(f"Scores saved to {scores_file_path}")
    return scores, scores_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        type=str,
        default="CER",
        choices=["CER", "WER", "BLUE", "ROGUE", "LP"],
        help="Evaluation metric to use",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="single",
        choices=["single","comparative"],
        help="Whether to evaluate denoised output against the clean text, or compare denoised and noisy output "
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="path/name of the model to evaluate the perplexity score on"
    )
    args = parser.parse_args()

    metric_map = {
        "CER": evaluate_CER,
        "WER": evaluate_WER,
        "ROGUE": evaluate_ROGUE,
        "LP": evaluate_letter_precision,
        "PERPLEXITY":evaluate_perplexity

    }

    evaluation_method = metric_map[args.metric]
    evaluation_task = args.task
    model_path = args.model_path 
   
    if model_path:
       set_global_var(model_path)

    denoised_data_path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json"
    evaluate_dataset(evaluation_method=evaluation_method, denoised_data_path=denoised_data_path,evaluation_task=evaluation_task)
