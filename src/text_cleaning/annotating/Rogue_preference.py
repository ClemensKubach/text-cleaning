import os
import json
from rouge_score import rouge_scorer
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_rouge_scores(den1, den2, ref):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores1 = scorer.score(ref, den1)
    scores2 = scorer.score(ref, den2)

    comparison = {}
    for key in ['rouge1', 'rouge2', 'rougeL']:
        score1 = scores1[key].fmeasure
        score2 = scores2[key].fmeasure
        if score1 > score2:
            comparison[key] = 1
        elif score2 > score1:
            comparison[key] = 2
        else:
            comparison[key] = 0
    return comparison

def process_pairs(llm_dir, denoised_data, ground_truth_path, output_dir):
    gt = load_json(ground_truth_path)
    os.makedirs(output_dir, exist_ok=True)

    for pair_name, (file1, file2) in denoised_data.items():
        path1 = Path(llm_dir) / file1
        path2 = Path(llm_dir) / file2

        if not path1.exists() or not path2.exists():
            print(f"Skipping pair {pair_name}: missing files")
            continue

        den1 = load_json(path1)
        den2 = load_json(path2)

        result = {}
        keys = set(den1.keys()) & set(den2.keys()) & set(gt.keys())
        for key in keys:
            ref_text = gt[key]
            denoised_text1 = den1[key]
            denoised_text2 = den2[key]

            comparison = compare_rouge_scores(denoised_text1, denoised_text2, ref_text)
            result[key] = comparison

        out_path = Path(output_dir) / f"{pair_name}_rouge_comparison.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        print(f"Saved comparison for {pair_name} at {out_path}")

# Usage example:

llm_dir = "/workspaces/mnlp_project_2/final_denoised"
ground_truth = "/workspaces/mnlp_project_2/data/ocr_datasets/eng/the_vampyre_clean.json"
output_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/pairwise_rogue"

denoised_data = {
    "ClemensK-gemma-3-1b-it-ocr-denoising-en_AGAINST_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en.json": (
        "the_vampyre_ocr_denoised_ClemensK-gemma-3-1b-it-ocr-denoising-en.json",
        "the_vampyre_ocr_denoised_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en.json"
        
    ),
    "google-gemma-3-1b-it_AGAINST_meta-llama-Llama-3.2-1B-Instruct.json": (
        "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json",
        "the_vampyre_ocr_denoised_meta-llama-Llama-3.2-1B-Instruct.json",
        
    ),
    "Llama-3.2-1B-Instruct-simple-context_AGAINST_google-gemma-3-1b-it-simple-context.json": (
        "the_vampyre_ocr_denoised_Llama-3.2-1B-Instruct-simple-context.json",
        "the_vampyre_ocr_denoised_google-gemma-3-1b-it-simple-context.json"
    )
}

process_pairs(llm_dir, denoised_data, ground_truth, output_dir)

