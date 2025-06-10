from text_cleaning.denoising.denoising import denoise
from text_cleaning.evaluation.evaluation import evaluate


def main():
    noisy_text = "This is some noisy text."
    cleaned_text = denoise(noisy_text)
    score = evaluate(noisy_text, cleaned_text)
    print(f"Score: {score}")


if __name__ == "__main__":
    main()
