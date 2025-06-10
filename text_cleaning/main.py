from text_cleaning.constants import DATA_DIR, LOG_DIR, WANDB_DIR
from text_cleaning.denoising.denoising import denoise
from text_cleaning.evaluation.evaluation import evaluate
from text_cleaning.utils import do_blocking_hf_login, setup_wandb


def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Wandb directory: {WANDB_DIR}")

    setup_wandb()
    do_blocking_hf_login()

    noisy_text = "This is s0me no1sy text."
    cleaned_text = denoise(noisy_text)

    score = evaluate(noisy_text, cleaned_text)
    print(f"Score: {score}")


if __name__ == "__main__":
    main()
