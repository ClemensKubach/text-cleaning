



import os
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# === CONFIG ===
input_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/judge_evaluations/"
output_dir = "data/plots/"
os.makedirs(output_dir, exist_ok=True)

categories = ["faithfulness", "fluency", "completeness", "no_hallucinations", "overall_winner"]
value_range = [0, 1, 2]

# === MODEL NAME MAPPING ===
model_name_map = {
    "ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en-lora": "ft_lora_llama",
    "ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en": "ft_llama",
    "ClemensK-gemma-3-1b-it-ocr-denoising-en": "ft_gemma",
    "Llama-3.2-1B-Instruct-simple-context": "in_context_llama",
    "google-gemma-3-1b-it": "simple_gemma",
    "google-gemma-3-1b-it-simple-context": "in_context_gemma",
    "meta-llama-Llama-3.2-1B-Instruct": "simple_llama"
}

# === HELPER ===
def parse_entry(entry):
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str):
        try:
            parsed = json.loads(entry)
            return parsed if isinstance(parsed, dict) else None
        except:
            return None
    return None

# === AGGREGATE WIN COUNTS ===
win_counts = defaultdict(lambda: defaultdict(int))  # model -> category -> wins
contest_counts = defaultdict(int)  # model -> number of contests

# === PROCESS FILES ===
for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    model_a_raw, _, model_b_raw = filename.replace(".json", "").partition("_AGAINST_")
    model_a = model_name_map.get(model_a_raw, model_a_raw)
    model_b = model_name_map.get(model_b_raw, model_b_raw)

    # Ensure both models are initialized in win_counts and contest_counts
    for model in [model_a, model_b]:
        for cat in categories:
            _ = win_counts[model][cat]
        contest_counts[model] += 1

    file_path = os.path.join(input_dir, filename)
    with open(file_path) as f:
        content = f.read().strip()
        if not content:
            print(f"⚠️ Skipping empty file: {filename}")
            continue
        raw_data = json.loads(content)

    counts = {cat: {v: 0 for v in value_range} for cat in categories}

    for entry in raw_data.values():
        parsed = parse_entry(entry)
        if not parsed:
            continue
        for cat in categories:
            val = parsed.get(cat)
            if val in value_range:
                counts[cat][val] += 1

    # === TALLY WINS ===
    for cat in categories:
        if counts[cat][1] > counts[cat][2]:
            win_counts[model_a][cat] += 1
        elif counts[cat][2] > counts[cat][1]:
            win_counts[model_b][cat] += 1

    # === CLUSTERED BAR PLOT ===
    labels = categories
    value_0 = [counts[cat][0] for cat in labels]
    value_1 = [counts[cat][1] for cat in labels]
    value_2 = [counts[cat][2] for cat in labels]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(labels)):
        cat = labels[i]
        if cat == "overall_winner":
            color_a = "#08519c"  # Deep blue
            color_0 = "#999999"
            color_b = "#238b45"  # Deep green
        else:
            color_a = "#a6cee3"
            color_0 = "#bdbdbd"
            color_b = "#b2df8a"

        ax.bar(x[i] - width, value_1[i], width, label=model_a if i == 0 else "", color=color_a)
        ax.bar(x[i],         value_0[i], width, label="Draw"   if i == 0 else "", color=color_0)
        ax.bar(x[i] + width, value_2[i], width, label=model_b if i == 0 else "", color=color_b)


    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title(f"{model_a} vs. {model_b}")
    ax.legend(title="Score")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{model_a_raw}_vs_{model_b_raw}.png")
    plt.savefig(plot_path)
    plt.close()

# === SUMMARY 1: Raw Win Counts ===
print("\n🏆 Model Win Summary (per Criterion):\n")
models = sorted(win_counts.keys())
header = ["Model"] + categories
print("{:<30}{}".format(header[0], "".join(f"{h:>20}" for h in header[1:])))
for model in models:
    row = [model] + [str(win_counts[model].get(cat, 0)) for cat in categories]
    print("{:<30}{}".format(row[0], "".join(f"{val:>20}" for val in row[1:])))

# === SUMMARY 2: Normalized by # of Contests ===
print("\n📊 Model Win Rates per Contest:\n")
print("{:<30}{}".format("Model", "".join(f"{cat:>20}" for cat in categories)))
for model in models:
    row = [model]
    for cat in categories:
        wins = win_counts[model][cat]
        contests = contest_counts[model]
        ratio = wins / contests if contests > 0 else 0
        row.append(f"{ratio:.2f}")
    print("{:<30}{}".format(row[0], "".join(f"{val:>20}" for val in row[1:])))

