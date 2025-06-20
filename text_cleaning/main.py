from text_cleaning.constants import DATA_DIR, LOG_DIR, WANDB_DIR
from text_cleaning.denoising.denoising import denoise_dataset
from text_cleaning.evaluation.evaluation import evaluate_CER, evaluate_dataset
from text_cleaning.utils import do_blocking_hf_login

import logging
import fire

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info(f"Wandb directory: {WANDB_DIR}")

    do_blocking_hf_login()

    logger.info("Denoising given dataset ...")
    _, denoised_data_path = denoise_dataset(subset=[3])
    logger.info("Evaluating denoised data...")
    evaluate_dataset(evaluation_method=evaluate_CER, denoised_data_path=denoised_data_path)


if __name__ == "__main__":
    fire.Fire(main)
