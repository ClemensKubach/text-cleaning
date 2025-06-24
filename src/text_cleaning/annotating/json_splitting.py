import os
import json
from collections import defaultdict

# === CONFIGURATION ===
input_dir = "/path/to/your/input_dir"
output_dir = "/path/to/your/output_dir"
group_prefix = "groupname-hw2_ocr-judge"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Initialize separate dicts per category
category_dicts = defaultdict(dict)  # category -> {key: label}

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith(".json"):
        continue

    fpath = os.path.join(input_dir, fname)

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {fname}: {e}")
        continue

    for key, category_vals in data.items():
        if category_vals is None:
            continue

        for category, label in category_vals.items():
            if label is not None:
                category_dicts[category][key] = label

# Save each category to its own file
for category, label_map in category_dicts.items():
    out_path = os.path.join(output_dir, f"{group_prefix}-{category}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

print("Done! Files created per category.")
