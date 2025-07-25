import fire
from dotenv import load_dotenv


def main():
    """Main entry point for the text-cleaning CLI."""
    load_dotenv()

    from text_cleaning.denoising.denoising import denoise_dataset
    from text_cleaning.denoising.fine_tuning import prepare_fine_tuning
    from text_cleaning.evaluation.classic_metrics.evaluation import run_evaluation
    from text_cleaning.evaluation.gemini_as_judge.judge_run import evaluate_judge
    from text_cleaning.utils import (
        do_blocking_hf_login,
        setup_logging,
        setup_wandb,
        merge_datasets,
        cache_model_and_tokenizer,
    )
    from text_cleaning.denoising.export import export_model, export_wandb

    setup_logging()
    setup_wandb()
    do_blocking_hf_login()
    fire.Fire(
        {
            "denoise": denoise_dataset,
            "fine-tune": prepare_fine_tuning,
            "export_model": export_model,
            "export_wandb": export_wandb,
            "merge_datasets": merge_datasets,
            "cache_model": cache_model_and_tokenizer,
            "evaluate_metrics": run_evaluation,
            "evaluate_judge":evaluate_judge

        },
        serialize=lambda _: None,
    )


if __name__ == "__main__":
    main()
