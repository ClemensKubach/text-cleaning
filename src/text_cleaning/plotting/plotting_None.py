import os
import json

input_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/judge_evaluations"

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse {filename}: {e}")
                continue

        null_count = sum(1 for v in data.values() if v is None)
        if null_count > 0:
            print(f"{filename}: {null_count} entries are None")