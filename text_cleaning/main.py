from text_cleaning.constants import DATA_DIR, LOG_DIR, WANDB_DIR
from text_cleaning.denoising.denoising import denoise_dataset
from text_cleaning.evaluation.evaluation import evaluate_dataset
from text_cleaning.utils import do_blocking_hf_login


def main() -> None:
    print(f"Data directory: {DATA_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Wandb directory: {WANDB_DIR}")

    # setup_wandb()
    do_blocking_hf_login()

    print("Denoising given dataset ...")
    _, denoised_data_path = denoise_dataset(subset=[3])
    print("Evaluating denoised data...")
    evaluate_dataset(denoised_data_path=denoised_data_path)


if __name__ == "__main__":
    main()
