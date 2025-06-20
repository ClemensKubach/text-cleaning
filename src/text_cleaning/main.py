import fire
from dotenv import load_dotenv

from text_cleaning.denoising.denoising import denoise_dataset
from text_cleaning.denoising.fine_tuning import prepare_fine_tuning
from text_cleaning.utils import do_blocking_hf_login, setup_logging


def main():
    """Main entry point for the text-cleaning CLI."""
    load_dotenv()
    setup_logging()
    do_blocking_hf_login()
    fire.Fire(
        {"denoise": denoise_dataset, "fine-tune": prepare_fine_tuning},
        serialize=lambda _: None,
    )


if __name__ == "__main__":
    main()
