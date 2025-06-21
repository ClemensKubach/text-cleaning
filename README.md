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
5. Set environment variable `GEMINI_API_TOKEN` for accessing the Gemini API for the `eval-gemini` command.

Now you can run the program from the command line, for example:
- Denoise text: `text-cleaning denoise --model_name <model> --model_type <type>`
- Evaluate with classic metrics: `text-cleaning eval-classic --metric <metric_name>`
- Evaluate with Gemini as a judge: `text-cleaning eval-gemini --evaluation_technique <technique> --input_names <files>`
- Prepare for fine-tuning: `text-cleaning fine-tune`

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
# Run Minerva 1B (too slow at the moment)
text-cleaning denoise --model_name="sapienzanlp/Minerva-1B-base-v1.0"
# Run BART-base (is not denoising)
text-cleaning denoise --model_name="facebook/bart-base" --model_type="seq2seq"
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
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

First, prepare the fine-tuning configs and dataset (`--generate_files=True` only required when configs changed):
```bash
source .venv/bin/activate
export HF_HOME=~/hf_cache  # might not be enough to add it to the .env file
text-cleaning fine-tune
deactivate
```

We use uv to install LLaMA-Factory.
```bash
cd LLaMA-Factory
uv python pin 3.10
uv sync --extra torch --extra metrics --extra badam --extra bitsandbytes --prerelease=allow
UV_TORCH_BACKEND=cu121 uv pip install torch  # force torch to install for cuda 12.1 (that may not be the default on the HPC)
```

Execute the training:
On the HPC:
```bash
cd ~/text-cleaning
sbatch run-denoising-finetuning.slurm.sh
```

On the local machine:
```bash
uv run --prerelease=allow llamafactory-cli train ../data/fine_tuning/train_configs/ocr-llama-the_vampyre-config.json
```

Export the model:
```bash
llamafactory-cli export ../data/fine.json
```


## Development
A devcontainer is set up for this project.
Open the project in your IDE using devcontainer and extensions, ruff and everything required for development is automatically installed.

Instead of `pip install`, use `uv add` (already installed) to install necessary dependencies.
