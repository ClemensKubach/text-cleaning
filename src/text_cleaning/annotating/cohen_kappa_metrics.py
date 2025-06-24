import os
import json
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_kappas(judge_dir, rogue_dir):
    filenames = sorted(os.listdir(judge_dir))
    
    # Store labels per (criterion, rouge_metric) to aggregate over files
    all_human_labels = defaultdict(list)  # criterion -> list of labels
    all_rogue_labels = defaultdict(list)  # rouge_metric -> list of labels
    combined_labels = defaultdict(lambda: ([], [])) # (criterion, rouge_metric) -> ([human], [rogue])
    
    for filename in filenames:
        print(filename)
        judge_path = os.path.join(judge_dir, filename)
        rogue_path = os.path.join(rogue_dir, filename)
        
        if not os.path.exists(rogue_path):
            print(f"Skipping {filename}: no corresponding rouge file.")
            continue
        
        try:
            judge_data = load_json(judge_path)
            rogue_data = load_json(rogue_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        
        # For each key (chunk id) in judge_data & rogue_data intersection
        keys = set(judge_data.keys()) & set(rogue_data.keys())
        for key in keys:
            if judge_data[key] is None:
                continue
            judge_entry = judge_data[key]  # e.g. {"faithfulness": 0, "fluency": 2, ...}
            rogue_entry = rogue_data[key]  # e.g. {"rouge1": 1, "rouge2": 2, "rougeL": 0}
            
            if judge_entry is None or rogue_entry is None:
                continue
            
            # For each judgment criterion in this key:
            for criterion, judge_label in judge_entry.items():
                if judge_label is None:
                    continue
                
                # For each rouge metric in this key:
                for rouge_metric, rogue_label in rogue_entry.items():
                    if rogue_label is None:
                        continue
                    
                    # Append for combined kappa calculation
                    combined_labels[(criterion, rouge_metric)][0].append(judge_label)
                    combined_labels[(criterion, rouge_metric)][1].append(rogue_label)
    
    # Calculate Cohen's Kappa for each (criterion, rouge_metric) pair
    results = {}
    for (criterion, rouge_metric), (judge_list, rogue_list) in combined_labels.items():
        if len(judge_list) < 2:
            results[(criterion, rouge_metric)] = None
            print(f"Not enough data for ({criterion}, {rouge_metric})")
        else:
            kappa = cohen_kappa_score(judge_list, rogue_list)
            results[(criterion, rouge_metric)] = kappa

    
    return results


# === CONFIGURATION ===
judge_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/judge_evaluations"
rogue_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/pairwise_rogue"

if __name__ == "__main__":
    kappas = compute_kappas(judge_dir, rogue_dir)
    for (criterion, rouge_metric), kappa in sorted(kappas.items()):
        if kappa is not None:
            print(f"Cohen's kappa for {criterion} vs {rouge_metric}: {kappa:.3f}")
            print('\n')
        else:
            print(f"Cohen's kappa for {criterion} vs {rouge_metric}: Not enough data")
