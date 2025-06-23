# import os
# import json
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from collections import defaultdict

# input_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/classic_evaluations"  # Replace with your directory path
# output_dir = "data/plots/" 
# excluded_keys = {"length_clean", "length_ocr"}

# label_mapping = {
#     "the_vampyre_ocr_denoised_ClemensK-gemma-3-1b-it-ocr-denoising-en.json": "ft_lora_gemma",
#     "the_vampyre_ocr_denoised_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en-lora.json": "ft_no_lora_gemma",
#     "the_vampyre_ocr_denoised_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en.json": "ft_lora_llama",
#     "the_vampyre_ocr_denoised_google-gemma-3-1b-it-simple-context.json": "in_context_gemma",
#     "the_vampyre_ocr_denoised_Llama-3.2-1B-Instruct-simple-context.json": "in_context_lamma",
#     "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json": "simple_gemma",
#     "the_vampyre_ocr_denoised_meta-llama-Llama-3.2-1B-Instruct.json": "simple_llama"
# }

# os.makedirs(output_dir, exist_ok=True)

# # Collect all metric values from all files
# all_metrics = defaultdict(lambda: defaultdict(list))  # metric -> filename -> values

# for filename in os.listdir(input_dir):
#     if filename.endswith(".json"):
#         filepath = os.path.join(input_dir, filename)
#         with open(filepath, "r") as f:
#             data = json.load(f)
#         for entry in data.values():
#             for key, value in entry.items():
#                 if key not in excluded_keys:
#                     all_metrics[key][filename].append(value)

# # Plot and save
# for metric, file_values in all_metrics.items():
#     # KDE Plot
#     plt.figure(figsize=(8, 5))
#     for filename, values in file_values.items():
#         label = label_mapping.get(filename, filename)
#         sns.kdeplot(values, label=label, fill=True, alpha=0.3)
#     plt.title(f"KDE Plot of '{metric}' Across models")
#     plt.xlabel(metric)
#     plt.ylabel("Density")
#     plt.legend()
#     plt.tight_layout()
#     kde_path = os.path.join(output_dir, f"{metric}_kde.png")
#     plt.savefig(kde_path)
#     plt.close()

#     # Boxplot
#     plt.figure(figsize=(8, 5))
#     sns.boxplot(data=[v for v in file_values.values()], 
#                 orient="v", 
#                 palette="pastel")
#     plt.xticks(ticks=np.arange(len(file_values)), labels=[label_mapping.get(fn, fn) for fn in file_values.keys()], rotation=45)
#     plt.title(f"Boxplot of '{metric}' Across Models")
#     plt.ylabel(metric)
#     plt.tight_layout()
#     boxplot_path = os.path.join(output_dir, f"{metric}_boxplot.png")
#     plt.savefig(boxplot_path)
#     plt.close()



import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

input_dir = "/workspaces/mnlp_project_2/data/evaluation_scores/classic_evaluations"
output_dir = "data/plots/"
os.makedirs(output_dir, exist_ok=True)

excluded_keys = {"length_clean", "length_ocr"}

# label_mapping = {
#     "the_vampyre_ocr_denoised_ClemensK-gemma-3-1b-it-ocr-denoising-en.json": "ft_gemma",
#     "the_vampyre_ocr_denoised_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en-lora.json": "ft_lora_gemma",
#     "the_vampyre_ocr_denoised_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en.json": "ft_llama",
#     "the_vampyre_ocr_denoised_google-gemma-3-1b-it-simple-context.json": "in_context_gemma",
#     "the_vampyre_ocr_denoised_Llama-3.2-1B-Instruct-simple-context.json": "in_context_lamma",
#     "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json": "simple_gemma",
#     "the_vampyre_ocr_denoised_meta-llama-Llama-3.2-1B-Instruct.json": "simple_llama"
# }
label_mapping = {
    "the_vampyre_ocr_denoised_ClemensK-gemma-3-1b-it-ocr-denoising-en.json": "ft_gemma",
    "the_vampyre_ocr_denoised_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en-lora.json": "ft_lora_llama",  # Changed from gemma to llama
    "the_vampyre_ocr_denoised_ClemensK-Llama-3.2-1B-Instruct-ocr-denoising-en.json": "ft_llama",
    "the_vampyre_ocr_denoised_google-gemma-3-1b-it-simple-context.json": "in_context_gemma",
    "the_vampyre_ocr_denoised_Llama-3.2-1B-Instruct-simple-context.json": "in_context_llama",  # Fixed "lamma" to "llama"
    "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json": "simple_gemma",
    "the_vampyre_ocr_denoised_meta-llama-Llama-3.2-1B-Instruct.json": "simple_llama"
}


# Collect all metrics and prepare for aggregation
all_metrics = defaultdict(lambda: defaultdict(list))
aggregated_scores = {}

for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)
        short_name = label_mapping.get(filename, filename)
        aggregated_scores[short_name] = {}
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Collect for plots and aggregation
        for entry in data.values():
            for key, value in entry.items():
                if key not in excluded_keys:
                    all_metrics[key][short_name].append(value)
        
        # Calculate aggregated stats per file
        for metric in all_metrics.keys():
            if metric in data.get(next(iter(data)), {}):  # Check if metric exists
                values = [v[metric] for v in data.values() if metric in v]
                aggregated_scores[short_name][metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                }

# Save aggregated scores to JSON
with open(os.path.join(output_dir, 'aggregated_scores.json'), 'w') as f:
    json.dump(aggregated_scores, f, indent=4)

# Plot and save
for metric, file_values in all_metrics.items():
    # KDE Plot (unchanged)
    plt.figure(figsize=(8, 5))
    for model, values in file_values.items():
        sns.kdeplot(values, label=model, fill=True, alpha=0.3)
    plt.title(f"KDE Plot of '{metric}' Across models")
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_kde.png"))
    plt.close()

    # Boxplot with log scale for CER/WER
    plt.figure(figsize=(8, 5))
    data_to_plot = [values for values in file_values.values()]
    
    # Use log scale only for error rate metrics
    if metric.lower() in ['cer', 'wer']:
        plt.yscale('log')
        plt.ylabel(f"log({metric})")
        # Handle zero values by adding small epsilon
        data_to_plot = [[max(v, 1e-5) for v in values] for values in data_to_plot]
    else:
        plt.ylabel(metric)
    
    sns.boxplot(data=data_to_plot, 
                orient="v",
                palette="pastel")
    plt.xticks(ticks=np.arange(len(file_values)), 
               labels=file_values.keys(), 
               rotation=45)
    plt.title(f"Boxplot of '{metric}' Across Models")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_boxplot.png"))
    plt.close()

