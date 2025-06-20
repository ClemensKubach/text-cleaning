# text-cleaning
Text denoising

## Getting Started
1. Run `uv sync` to install all dependencies and tools.
2. Set environment variable HF_TOKEN for accessing Huggingface.
3. set environment variable GEMINI_API_TOKEN  for accesing the Gemini api for the gemini_as_judge
4. Run program as module:

   To denoise the text run   `python -m text_cleaning.denoising.denoising --model_name --model_type`

   To evaluate the denoising run with classic metrics  `python -m text_cleaning.evaluation.classic_metric.evaluation --metric`
   
   To evaluate the denoising run with the LLM(Gemini) as a judge `python -m text_cleaning.evaluation.gemini_as_judge.judge_run --evaluation_technique  --input_names`
   
## Examples

### Denoising a single example page
```bash
# Run Llama 3.2-1B with simple in-context learning without sentence chunking
python -m text_cleaning.denoising.denoising --model_name="meta-llama/Llama-3.2-1B-Instruct" --subset="[3,]" --in_context "simple" --use_sentence_chunks=False

# Run Gemma 3-1B with complex in-context learning, majority voting and sentence chunking
python -m text_cleaning.denoising.denoising --model_name="google/gemma-3-1b-it" --subset="[3,]" --in_context "complex" --num_attempts=5

# Run Minerva 1B without in-context learning (too slow at the moment)
python -m text_cleaning.denoising.denoising --model_name="sapienzanlp/Minerva-1B-base-v1.0" --subset="[3,]" --in_context "None"

# Run BART-base (is not denoising)
python -m text_cleaning.denoising.denoising --model_name="facebook/bart-base" --model_type="seq2seq" --subset="[3,]" --in_context "None"
```

### Denoising all pages
```bash
# Run Llama 3.2-1B with simple in-context learning 
python -m text_cleaning.denoising.denoising --model_name="meta-llama/Llama-3.2-1B-Instruct" --in_context "simple"

# Run Gemma 3-1B
python -m text_cleaning.denoising.denoising --model_name="google/gemma-3-1b-it"

# Run Minerva 1B (too slow at the moment)
python -m text_cleaning.denoising.denoising --model_name="sapienzanlp/Minerva-1B-base-v1.0"

# Run BART-base (is not denoising)
python -m text_cleaning.denoising.denoising --model_name="facebook/bart-base" --model_type="seq2seq"
```
### Evaluation with the classical ocr metrics

```bash
python -m text_cleaning.evaluation.classic_metrics.evaluation --metric "WER" --task "single"  
```

### Evaluation with the Gemini as a judge 

```bash
 python -m text_cleaning.evaluation.gemini_as_judge.judge_run --evaluation_technique "pairwise" --input_names "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json" "the_vampyre_ocr_denoised_facebook-bart-base.json"
```

## Development
A devcontainer is set up for this project.
Open the project in your IDE using devcontainer and extensions, ruff and everything required for development is automatically installed.

Instead of `pip install`, use `uv add` (already installed) to install necessary dependencies.


### Fine-tuning with LLaMA-Factory
First, download LLaMA-Factory:
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

First, prepare the fine-tuning configs and dataset (`--generate_files=True` only required when configs changed):
```bash
source .venv/bin/activate
python -m text_cleaning.denoising.fine_tuning
deactivate
```

We use uv to install LLaMA-Factory.
```bash
cd LLaMA-Factory
uv python pin 3.10
uv sync --extra torch --extra metrics --prerelease=allow
UV_TORCH_BACKEND=cu121 uv pip install torch  # force torch to install for cuda 12.1 (that may not be the default on the HPC)
```

Execute the training:
On the HPC:

Set the following env vars:
```
HF_HOME=~/hf_cache
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
````

And run the script:
```bash
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
