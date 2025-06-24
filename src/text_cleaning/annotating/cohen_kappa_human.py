import os
import json
from sklearn.metrics import cohen_kappa_score

from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_agreements(human_dir, judge_dir, keys_to_check):
    nested_keys = set()
    human_scores = defaultdict(lambda: defaultdict(list))
    judge_scores = defaultdict(lambda: defaultdict(list))

    filenames = sorted(os.listdir(human_dir))
    results = {}

    for filename in filenames:
        human_path = os.path.join(human_dir, filename)
        judge_path = os.path.join(judge_dir, filename)

        if not os.path.exists(judge_path):
            print(f"Skipping {filename}, not found in judge directory.")
            continue

        
        try:
            human_data = load_json(human_path)
            judge_data = load_json(judge_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        

        for key in keys_to_check:
            key = str(key)
            if key not in human_data or key not in judge_data:
                continue

            if human_data[key] is None or judge_data[key] is None:
                continue
            human_inner = human_data[key]
            judge_inner = judge_data[key]

            for criterion in human_inner:
                if criterion not in judge_inner:
                    continue

                human_label = human_inner[criterion]
                judge_label = judge_inner[criterion]

                # skip if label is None or not valid
                if human_label is None or judge_label is None:
                    continue

                human_scores[criterion][filename].append(human_label)
                judge_scores[criterion][filename].append(judge_label)
                nested_keys.add(criterion)

    # Calculate Cohen's Kappa for each nested key (criterion)
    for criterion in sorted(nested_keys):
        all_human = []
        all_judge = []

        for fname in human_scores[criterion]:
            all_human.extend(human_scores[criterion][fname])
            all_judge.extend(judge_scores[criterion][fname])
        

        if len(all_human) < 2:
            kappa = None
            print(f"Not enough data for {criterion}.")
        else:
            kappa = cohen_kappa_score(all_human, all_judge)

        results[criterion] = kappa

    # Overall agreement
    total_human = []
    total_judge = []
    for crit in human_scores:
        for fname in human_scores[crit]:
            total_human.extend(human_scores[crit][fname])
            total_judge.extend(judge_scores[crit][fname])

    overall_kappa = cohen_kappa_score(total_human, total_judge) if len(total_human) > 1 else None
    results["overall"] = overall_kappa

    return results


# === CONFIGURATION ===
human_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/human_evaluation"
judge_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/judge_evaluations"
keys_to_check = ["0", "5", "36", "37", "38", "43", "47"]

# === RUN SCRIPT ===
if __name__ == "__main__":
    scores = compute_agreements(human_dir, judge_dir, keys_to_check)
    for k, v in scores.items():
        print(f"{k}: {v:.3f}" if v is not None else f"{k}: Not enough data")
