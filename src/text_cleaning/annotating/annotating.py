
import os
import json

def merge_model_outputs_with_ground_truth(
    input_dir: str,
    output_dir: str,
    keys: list,
    ground_truth_path: str
):
    # Define your groups with explicit filenames
    groups = {
        "Llama_vs_Gemma_simple-context.json": [
            "the_vampyre_ocr_denoised_Llama-3.2-1B-Instruct-simple-context.json",
            "the_vampyre_ocr_denoised_google-gemma-3-1b-it-simple-context.json"
        ],
        "ClemensK_Llama_vs_Gemma.json": [
            "the_vampyre_ocr_denoised_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en.json",
            "the_vampyre_ocr_denoised_ClemensK-gemma-3-1b-it-ocr-denoising-en.json"
        ],
        "Llama_vs_Gemma.json": [
            "the_vampyre_ocr_denoised_meta-llama-Llama-3.2-1B-Instruct.json",
            "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json"
        ]
    }

    # Load ground truth JSON once
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    for out_filename, files in groups.items():
        print(f"Processing group: {out_filename}")
        # Load both files in group
        data1_path = os.path.join(input_dir, files[0])
        data2_path = os.path.join(input_dir, files[1])

        with open(data1_path, "r", encoding="utf-8") as f1, open(data2_path, "r", encoding="utf-8") as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

        combined = {}

        for key in keys:
            combined[key] = {
                "model1": data1.get(key, ""),
                "model2": data2.get(key, ""),
                "ground_truth": ground_truth_data.get(key, "")
            }

        # Write combined output
        output_path = os.path.join(output_dir, out_filename)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(combined, out_f, indent=2, ensure_ascii=False)

        print(f"Saved combined file to {output_path}")

# Example usage:
merge_model_outputs_with_ground_truth(
     input_dir="/workspaces/mnlp_project_2/final_denoised",
     output_dir="/workspaces/mnlp_project_2/data/evaluation_scores/human_evaluation/annotations",
     keys=["0", "5", "36", "37", "38", "43", "47"],
     ground_truth_path="/workspaces/mnlp_project_2/data/ocr_datasets/eng/the_vampyre_clean.json"
 )

