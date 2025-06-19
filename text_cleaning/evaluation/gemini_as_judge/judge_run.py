# To run this code you need to install the following dependencies:
# pip install google-genai

import os
from google import genai
from google.genai import types
import argparse
from text_cleaning.utils import load_data, save_data
import logging

TEXT_PATH = "/workspaces/mnlp_project_2/data/ocr_datasets/eng/"
CLEAN_TEXT = TEXT_PATH + "the_vampyre_clean.json"
OCR_TEXT = TEXT_PATH + "the_vampyre_ocr.json"
SCORES_PATH = "/workspaces/mnlp_project_2/data/ocr_datasets/eng/evaluation_scores/"


logger = logging.getLogger(__name__)


def evaluate_dataset(input_paths: list[str], evaluation_technique: str) -> dict[int, float]:
    """Evaluate the denoised data.

    Args:
        denoised_data_path: The path to the generated denoised dataset.
        input_files:  the list containing names of the files to be evaluated
        evaluation_technique: whether to evaluate: explicitly, pairwise comparison, ranking
    Returns:
        The scores.
    """
    clean_data = load_data(CLEAN_TEXT)
    logger.info(f"Loaded clean data from {CLEAN_TEXT}")
    noisy_data = load_data(OCR_TEXT)
    logger.info(f"Loaded denoised data from {OCR_TEXT}")
    denoised_data = []
    for i in range(len(input_paths)):
        denoised_data.append(load_data(TEXT_PATH + input_paths[i]))
        logger.info(f"Loaded denoised data {i} from {input_paths[i]}")

    scores: dict[int, str] = {}
    print(denoised_data)
    for i in denoised_data[0]:
        texts_dict = {}
        texts_dict["clean_text"] = clean_data[i]
        texts_dict["ocr_text"] = noisy_data[i]
        # denoised_texts = [denoised_data[j][i] for j in range(len(denoised_data))]
        if evaluation_technique == "explicit":
            texts_dict["denoised_text"] = denoised_data[0][i]
            evaluation = generate(evaluation_technique=evaluation_technique, input_texts=texts_dict)

            scores[i] = evaluation
        elif evaluation_technique == "pairwise":
            texts_dict["denoised_text_1"] = denoised_data[0][i]
            texts_dict["denoised_text_2"] = denoised_data[1][i]
            evaluation = generate(evaluation_technique=evaluation_technique, input_texts=texts_dict)
            scores[i] = evaluation
        elif evaluation_technique == "ranking":
            for j in range(len(denoised_data)):
                texts_dict.update({f"denoised_text_{j + 1}": denoised_data[j][i]})
            evaluation = generate(evaluation_technique=evaluation_technique, input_texts=texts_dict)
            scores[i] = evaluation

    str_input_paths = ""
    for element in input_paths:
        str_input_paths += element

    scores_file_path = SCORES_PATH + f"{evaluation_technique}" + "_" + f"{str_input_paths}"
    save_data(scores_file_path, scores)
    logger.info(f"Scores saved to {scores_file_path}")
    return scores, scores_file_path


def generate(evaluation_technique: str, input_texts: dict[str]):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    if evaluation_technique == "explicit":
        prompt = f"""Assess the following denoised OCR output based on these 5 criteria (1–10 each):
        - Fidelity: alignment with meaning in clean_text
        - Fluency: grammar and natural style
        - Completeness: no missing or truncated segments with respect to OCR_text
        - No hallucinations: no added or altered content with respect to OCR_text
        - OCR-improvement: How much better is denoised_ocr compared to ocr_text. 1 = significantly worse than ocr_text, 3 = no improvement to ocr_text, 10 = very significant improvement toward clean text

        Average the scores to compute final_score. 
        Return ONLY  STRING in JSON format :
        {{
        "fidelity": int,
        "fluency": int,
        "completeness": int,
        "no_hallucinations": int,
        "OCR-improvement" : int,
        "final_score": float
        }}
        clean_text: {input_texts["clean_text"]}
        ocr_text: {input_texts["ocr_text"]}
        denoised_ocr_text: {input_texts["denoised_text"]}
        """

    elif evaluation_technique == "ranking":
        keys_to_drop = {"clean_text", "ocr_text"}
        denoised_texts = {k: v for k, v in input_texts.items() if k not in keys_to_drop}
        prompt = f"""Rank the following denoised OCR outputs from **best** to **worst** based on these 5 criteria:
        - Preserve the meaning (Fidelity)
        - Maintain natural grammar and style (Fluency)
        - Include all content from the OCR (Completeness)
        - Avoid adding or removing content (No hallucinations)
        - Improve upon the OCR input (OCR‑improvement)
        Input:
        clean_text: {input_texts["clean_text"]}
        ocr_text: {input_texts["ocr_text"]}
        denoised_texts: {denoised_texts}
        Respond with a single line:  
        “Ranking: [5,4,2, …]”  
        where numbers correspond to output indices, sorted from best to worst. No extra text."""
    elif evaluation_technique == "pairwise":
        prompt = f"""Assess which of the two denoised OCR outputs is better based on these 5 criteria:
        - Fidelity (alignment with clean_text)
        - Fluency (grammar & style)
        - Completeness (with respect to OCR_text)
        - No hallucinations (no added or removed content)
        - OCR-improvement (1 = worse than OCR, 3 = no improvement, 10 = maximum improvement)

        Inputs:
        clean_text: {input_texts["clean_text"]}
        ocr_text: {input_texts["ocr_text"]}
        denoised_ocr_text_1: {input_texts["denoised_text_1"]}
        denoised_ocr_text_2: {input_texts["denoised_text_2"]}

        For each criterion, whichever output scores higher gets that "point". The better_output is the output with more points.  
        Respond with exactly `1` if **denoised_ocr_text_1** is better_output, or `2` if **denoised_ocr_text_2** is better_output or `0` in case of draw(the same number of points)
        Return ONLY  STRING in JSON format :
        {{
        "better_output": int,
        }}
        **  DO NOT OUTPUT ANY ADDITIONAL TEXT.**
        """
    else:
        raise KeyError

    model = "gemini-2.5-flash-preview-05-20"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="text/plain",
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")
        response_text += chunk.text
    return response_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_technique",
        type=str,
        default="explicit",
        choices=["explicit", "pairwise", "ranking"],
        help="choose  the evaluation method ",
    )
    parser.add_argument(
        "--input_names",
        nargs="+",
        type=str,
        required=True,
        help="the name of the input  files that are to be evaluated. In case of the ranking evaluation"
        "[name_1,name_2,....]  in case of pairwise evaluation [name_1,name_2] "
        "in case of explicit evaluation [name]",
    )
    args = parser.parse_args()
    evaluation_technique = args.evaluation_technique
    input_paths = [element for element in args.input_names]
    evaluate_dataset(input_paths=input_paths, evaluation_technique=evaluation_technique)
