from text_cleaning.constants import DATA_DIR, LOG_DIR, WANDB_DIR
from text_cleaning.denoising.denoising import denoise_dataset
from text_cleaning.evaluation.evaluation import evaluate_CER, evaluate_dataset
from text_cleaning.utils import do_blocking_hf_login
import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Wandb directory: {WANDB_DIR}")

    # setup_wandb()
    do_blocking_hf_login()

    logger.info("Denoising given dataset ...")
    _, denoised_data_path = denoise_dataset(subset=[3])
    logger.info("Evaluating denoised data...")
    evaluate_dataset(evaluation_method=evaluate_CER, denoised_data_path=denoised_data_path)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Denoise and evaluate a dataset using a specified model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Name of the model to use for denoising (e.g., 'bert-base-uncased')",
    )
    args = parser.parse_args()

    # Call main with the provided model_name
    # main(args.model_name)
    main()
