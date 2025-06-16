# text-cleaning
Text denoising

## Getting Started
1. Run `uv sync` to install all dependencies and tools.
2. Set environment variable HF_TOKEN for accessing Huggingface.
3. Run program as module:

   To denoise the text run   `python -m text_cleaning.denoising.denoising --model_name --model_type`

   To evaluate the denoising run  `python -m text_cleaning.evaluation.evaluation --metric`
   
## Examples
```bash
# Run Llama 3.2-1B for only page 3 without chunking
python -m text_cleaning.denoising.denoising --model_name="meta-llama/Llama-3.2-1B" --model_type="causal" --subset="[3,]" --chunk_size=None

# Run Gemma
python -m text_cleaning.denoising.denoising --model_name="google/gemma-3-1b-it" --model_type="causal" --subset="[3,]" --chunk_size=None

# Run Minerva
python -m text_cleaning.denoising.denoising --model_name="sapienzanlp/Minerva-1B-base-v1.0" --model_type="causal" --subset="[3,]" --chunk_size=None
```

## Development
A devcontainer is set up for this project.
Open the project in your IDE using devcontainer and extensions, ruff and everything required for development is automatically installed.

Instead of `pip install`, use `uv add` (already installed) to install necessary dependencies.
