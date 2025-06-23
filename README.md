# text-cleaning
Text denoising

## Getting Started
1. Clone the repository: `git clone https://github.com/ClemensKubach/text-cleaning` and cd into it (`cd text-cleaning`)
2. Run `uv sync` to install all dependencies and tools.
3. To make the CLI commands available, install the project in editable mode:
   ```bash
   uv pip install -e .
   ```
4. Set environment variable `HF_TOKEN` for accessing Huggingface.
5. Set environment variable `GEMINI_API_TOKEN` for accessing the Gemini API for evaluation with gemini as a judge.


## Examples

### Denoising a single example page
```bash
# Run Llama 3.2-1B with simple in-context learning without sentence chunking
text-cleaning denoise --model_name="meta-llama/Llama-3.2-1B-Instruct" --subset="[3,]" --in_context "simple" --use_sentence_chunks=False
# Run Gemma 3-1B with complex in-context learning, majority voting and sentence chunking
text-cleaning denoise --model_name="google/gemma-3-1b-it" --subset="[3,]" --in_context "complex" --num_attempts=5
# Run Minerva 1B without in-context learning (too slow at the moment)
text-cleaning denoise --model_name="sapienzanlp/Minerva-1B-base-v1.0" --subset="[3,]" --in_context "None"
# Run BART-base (is not denoising)
text-cleaning denoise --model_name="facebook/bart-base" --model_type="seq2seq" --subset="[3,]" --in_context "None"
```

### Denoising all pages
```bash
# Run Llama 3.2-1B with simple in-context learning
text-cleaning denoise --model_name="meta-llama/Llama-3.2-1B-Instruct" --in_context "simple"
# Run Gemma 3-1B
text-cleaning denoise --model_name="google/gemma-3-1b-it"
# Run Minerva 1B
text-cleaning denoise --model_name="sapienzanlp/Minerva-1B-base-v1.0"
# Run Fine-tuned Gemma 3-1B
text-cleaning denoise --model_name=ClemensK/gemma-3-1b-it-ocr-denoising-en --is_finetuned True
# Run Fine-tuned Llama 3.2-1B
text-cleaning denoise --model_name="ClemensK/Llama-3.2-1B-Instruct-ocr-denoising-en" --is_finetuned True
# Run Fine-tuned Minerva 1B
text-cleaning denoise --model_name="ClemensK/Minerva-1B-base-v1.0-ocr-denoising-en" --is_finetuned True
```

You can also use the `run-denoising.slurm.sh` script to run the denoising on the HPC.
First, cache the model and tokenizer, then run the denoising script.
```bash
export HF_HOME=$SCRATCH/hf_cache
uv run text-cleaning cache_model --model "gemma"
uv run text-cleaning cache_model --model "llama"
uv run text-cleaning cache_model --model "minerva"
uv run text-cleaning cache_model --model_id "ClemensK/gemma-3-1b-it-ocr-denoising-en"
uv run text-cleaning cache_model --model_id "ClemensK/Llama-3.2-1B-Instruct-ocr-denoising-en"

# Then run the denoising script
sbatch run-denoising.slurm.sh
```

### Merging existing datasets
For the following example, the merged dataset is already integrated and named synthetic. The output is saved in the data directory: `data/ocr_datasets/eng/synthetic_ocr.json` and `data/ocr_datasets/eng/synthetic_clean.json`.

```bash
text-cleaning merge_datasets --noisy_datasets '["src/text_cleaning/ocr_text_creating/ocr_frankenstein.json", "src/text_cleaning/ocr_text_creating/ocr_otoranto.json"]' --clean_datasets '["src/text_cleaning/ocr_text_creating/clean_frankenstein.json", "src/text_cleaning/ocr_text_creating/clean_otoranto.json"]'
```

### Evaluation with the classical ocr metrics

```bash
text-cleaning eval-classic --metric "WER" --task "single" --denoised_data_path "data/ocr_datasets/eng/the_vampyre_ocr_denoised_google-gemma-3-1b-it.json"
```

### Evaluation with the Gemini as a judge 

```bash
 text-cleaning eval-gemini --evaluation_technique "pairwise" --input_names "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json" "the_vampyre_ocr_denoised_facebook-bart-base.json"
```

### Fine-tuning with LLaMA-Factory
First, download LLaMA-Factory into the text-cleaning directory (`cd text-cleaning`):
```bash
git clone --depth 1 --branch no-processor-fallback --single-branch https://github.com/ClemensKubach/LLaMA-Factory.git
```

Set important environment variables (when only local machine (not HPC), they can be set in the .env file):
```bash
export HF_HOME=$SCRATCH/hf_cache  # for the LEONARDO HPC
export WANDB_API_KEY=<your_api_key>
```

First, prepare the fine-tuning configs and dataset (`--generate_files=True` only required when configs changed):
```bash
uv run text-cleaning fine-tune
```

We use uv to install LLaMA-Factory.
```bash
cd LLaMA-Factory
uv python pin 3.10
uv sync --extra torch --extra metrics --extra badam --extra bitsandbytes --extra deepspeed --prerelease=allow
UV_TORCH_BACKEND=cu121 uv pip install wandb setuptools torch   # force torch to install for cuda 12.1 (that may not be the default on the HPC)
```

Execute the training:
On the HPC:
```bash
cd ~/text-cleaning
sbatch run-denoising-finetuning.slurm.sh gemma
sbatch run-denoising-finetuning.slurm.sh llama
sbatch run-denoising-finetuning.slurm.sh minerva
# show output via cat logs/ocr_finetune...job_id.out
# upload run to wandb via the in log displayed command 
```
If there was an error, unset the HF_HOME variable and cache again, because it searches the models in HF_HOME/hub (known hf version issue) and try again.

Or on the local machine from LLaMA-Factory directory:
```bash
uv run --prerelease=allow llamafactory-cli train ../data/fine_tuning/train_configs/ocr-gemma-synthetic-config.json
```

Export the model:
```bash
cd ~/text-cleaning/LLaMA-Factory
uv run --prerelease=allow llamafactory-cli export ../data/fine_tuning/models/merged-gemma-synthetic-config.json
uv run --prerelease=allow llamafactory-cli export ../data/fine_tuning/models/merged-llama-synthetic-config.json
# OR (for non-lora/full finetuning)
cd ~/text-cleaning
source .venv/bin/activate
cd LLaMA-Factory  # the following must be run from the LLaMA-Factory directory
text-cleaning export_model ../data/fine_tuning/models/merged-llama-synthetic-config.json
text-cleaning export_model ../data/fine_tuning/models/merged-minerva-synthetic-config.json
```


## Development
A devcontainer is set up for this project.
Open the project in your IDE using devcontainer and extensions, ruff and everything required for development is automatically installed.

Instead of `pip install`, use `uv add` (already installed) to install necessary dependencies.
