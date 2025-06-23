import json
import os
from typing import List, Tuple
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_categorical_kappa(
    file_pairs: List[Tuple[str, str]],
    keys_to_check: List[str]
) -> dict:
    """
    Compute Cohen's Kappa (unweighted) for given JSON file pairs and selected keys.
    
    Each file is expected to contain a dict: {id: {key: category_value, ...}}
    
    :param file_pairs: List of (file1_path, file2_path) tuples.
    :param keys_to_check: List of keys to evaluate (e.g., ['faithfulness', 'fluency']).
    :return: Dict with average kappa per key.
    """
    key_to_pairs = defaultdict(list)

    for file1_path, file2_path in file_pairs:
        data1 = load_json(file1_path)
        data2 = load_json(file2_path)

        # Match entries by shared keys (like "27", "36" etc.)
        shared_ids = set(data1.keys()) & set(data2.keys())

        for id_key in shared_ids:
            entry1 = data1[id_key]
            entry2 = data2[id_key]

            for key in keys_to_check:
                val1 = entry1.get(key)
                val2 = entry2.get(key)
                if val1 is not None and val2 is not None:
                    key_to_pairs[key].append((val1, val2))

    # Compute Cohen's Kappa per key
    key_to_kappa = {}
    for key, value_pairs in key_to_pairs.items():
        if not value_pairs:
            continue
        labels1, labels2 = zip(*value_pairs)
        kappa = cohen_kappa_score(labels1, labels2)  # unweighted = categorical
        key_to_kappa[key] = kappa

    return key_to_kappa
