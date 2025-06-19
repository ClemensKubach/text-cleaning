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
# Run Llama 3.2-1B
python -m text_cleaning.denoising.denoising --model_name="meta-llama/Llama-3.2-1B-Instruct" --model_type="causal" --subset="[3,]" --in_context "simple"

# Run Gemma 3-1B
python -m text_cleaning.denoising.denoising --model_name="google/gemma-3-1b-it" --model_type="causal" --subset="[3,]" --in_context "complex"

# Run Minerva 1B (too slow at the moment)
python -m text_cleaning.denoising.denoising --model_name="sapienzanlp/Minerva-1B-base-v1.0" --model_type="causal" --subset="[3,]" --in_context "None"

# Run BART-base (is not denoising)
python -m text_cleaning.denoising.denoising --model_name="facebook/bart-base" --model_type="seq2seq" --subset="[3,]" --in_context "None"
```

### Denoising all pages
```bash
# Run Llama 3.2-1B
python -m text_cleaning.denoising.denoising --model_name="meta-llama/Llama-3.2-1B-Instruct" --model_type="causal"

# Run Gemma 3-1B
python -m text_cleaning.denoising.denoising --model_name="google/gemma-3-1b-it" --model_type="causal"

# Run Minerva 1B (too slow at the moment)
python -m text_cleaning.denoising.denoising --model_name="sapienzanlp/Minerva-1B-base-v1.0" --model_type="causal"

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
First, prepare the fine-tuning configs and dataset (`--generate_files=True` only required once):
```bash
python -m text_cleaning.denoising.fine_tuning --generate_files True
```


Now, download LLaMA-Factory:
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

We use uv to install LLaMA-Factory with python 3.9.
```bash
uv python pin 3.9
uv sync --extra torch --extra metrics --prerelease=allow
```

Execute the training:
```bash
uv run --prerelease=allow llamafactory-cli train ../data/fine_tuning/train_configs/ocr-gemma-the_vampyre-config.json
```

Export the model:
!llamafactory-cli export ../data/fine.json
