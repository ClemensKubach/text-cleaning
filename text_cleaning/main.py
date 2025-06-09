from text_cleaning.denoising.run_denoise import denoise_text
from text_cleaning.evaluation.run_eval import evaluate


def main():
    print("Hello from text-cleaning!")
    cleaned_text = denoise_text()
    score = evaluate(cleaned_text)
    print(f"Score: {score}")


if __name__ == "__main__":
    main()
