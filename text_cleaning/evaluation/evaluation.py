from pathlib import Path

from tqdm import tqdm

from text_cleaning.constants import DATA_DIR
from text_cleaning.utils import load_data, save_data


def evaluate(noisy_text: str, denoised_text: str, clean_text: str) -> float:
    """Just a placeholder for know. Will be called from main.py.

    Args:
        noisy_text: The given noisy text.
        denoised_text: The generated denoised text.
        clean_text: The clean ground truth text.

    Returns:
        The score.
    """
    return 1.0


def evaluate_dataset(
    denoised_data_path: Path,
    noisy_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json",
    cleaned_data_path: Path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_clean.json",
) -> dict[int, float]:
    """Evaluate the denoised data.

    Args:
        noisy_data_path: The path to the given noisy dataset.
        cleaned_data_path: The path to the clean ground truth dataset.
        denoised_data_path: The path to the generated denoised dataset.

    Returns:
        The scores.
    """
    noisy_data = load_data(noisy_data_path)
    print(f"Loaded noisy data from {noisy_data_path}")
    clean_data = load_data(cleaned_data_path)
    print(f"Loaded clean data from {cleaned_data_path}")
    denoised_data = load_data(denoised_data_path)
    print(f"Loaded denoised data from {denoised_data_path}")

    scores: dict[int, float] = {}
    for i in tqdm(denoised_data):
        noisy_text = noisy_data[i]
        clean_text = clean_data[i]
        denoised_text = denoised_data[i]
        score = evaluate(noisy_text, denoised_text, clean_text)
        scores[i] = score
    scores_file_path = denoised_data_path.with_name(f"{denoised_data_path.stem}_scores{denoised_data_path.suffix}")
    save_data(scores_file_path, scores)
    print(f"Scores saved to {scores_file_path}")
    return scores, scores_file_path


if __name__ == "__main__":
    denoised_data_path = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr_denoised_google-gemma-3-1b-it.json"
    evaluate_dataset(denoised_data_path=denoised_data_path)
