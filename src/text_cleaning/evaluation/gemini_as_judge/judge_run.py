# To run this code you need to install the following dependencies:
# pip install google-genai

import os
from google import genai
from google.genai import types
import argparse
from text_cleaning.utils import load_data, save_data
import logging
import time
from google.genai import errors
import random 
import json


def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True

TEXT_PATH = "/workspaces/mnlp_project_2/data/ocr_datasets/eng/"
CLEAN_TEXT = TEXT_PATH + "the_vampyre_clean.json"
OCR_TEXT = TEXT_PATH + "the_vampyre_ocr.json"
SCORES_PATH = "/workspaces/mnlp_project_2/data/ocr_datasets/eng/evaluation_scores/"

MAX_RETRIES = 10

logger = logging.getLogger(__name__)


def evaluate_dataset(input_paths: list[str], evaluation_technique: str) -> dict[int, float]:
    """Evaluate the denoised data.

    Args:
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
    '''iterating over the keys of the jsons, ASSUMING ALL THE DENOISED DATAS HAVE THE SAME KEYS'''
    for i in denoised_data[0]:
        texts_dict = {}
        texts_dict["clean_text"] = clean_data[i]
        texts_dict["ocr_text"] = noisy_data[i]
        if evaluation_technique == "explicit":
            texts_dict["denoised_text"] = denoised_data[0][i]
            evaluation = generate(evaluation_technique=evaluation_technique, input_texts=texts_dict)
            scores[i] = evaluation

        elif evaluation_technique == "pairwise":
            "randomly shuffling the order in the model to hinder the position bias"
            prob = random.random()
            changed_mapping = {"1":"2","2":"1"}
            if prob <= 0.5:
                texts_dict["denoised_text_1"] = denoised_data[0][i]
                texts_dict["denoised_text_2"] = denoised_data[1][i]
                evaluation = generate(evaluation_technique=evaluation_technique, input_texts=texts_dict)
                if not is_json(evaluation):
                    return 0, scores_file_path
            else:
                texts_dict["denoised_text_1"] = denoised_data[1][i]
                texts_dict["denoised_text_2"] = denoised_data[0][i]
                evaluation = generate(evaluation_technique=evaluation_technique, input_texts=texts_dict)
                if not is_json(evaluation):
                    return 0, scores_file_path
                for key in evaluation:
                    evaluation[key] = changed_mapping[evaluation[key]]
            
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
        prompt = f"""
        Evaluate the quality of the `denoised_ocr_output` based on the four criteria below. Each should be rated on a scale from 1 to 10.

        Use the following comparisons for each criterion:
        - Fidelity: Compare `denoised_ocr_output` to `clean_text`. Does it preserve the original meaning?
        - Fluency: Judge the grammar and naturalness of `denoised_ocr_output` on its own.
        - Completeness: Compare `denoised_ocr_output` to `ocr_text`. Is any information from `ocr_text` missing or truncated?
        - No Hallucinations: Compare `denoised_ocr_output` to `ocr_text`. Are there any added or altered element in `denoised_ocr_output` that were not present in `ocr_text` nor 'clean_text`?
        

        Compute the average of these four scores and return it as `final_score`.

        ❗️Return the result as a valid JSON string and nothing else:

        {{
        "fidelity": int,
        "fluency": int,
        "completeness": int,
        "no_hallucinations": int,
        "final_score": float
        }}

        Inputs:
        clean_text: {input_texts["clean_text"]}
        ocr_text: {input_texts["ocr_text"]}
        denoised_ocr_output: {input_texts["denoised_text"]}
        """


    elif evaluation_technique == "ranking":
        keys_to_drop = {"clean_text", "ocr_text"}
        denoised_texts = {k: v for k, v in input_texts.items() if k not in keys_to_drop}
        prompt = f"""Rank the following `denoised_texts` from **best** to **worst** based on these 5 criteria:
        -Fidelity:does the denoised output preserve the meaning of the `clean_text`
        -Fluency: Is the text grammatically correct and natural sounding?
        -Completeness: Is any information missing or truncated compared to the `ocr_text`?
        -No Hallucinations: Are there any added or altered details not found in the `ocr_text`?
        - OCR Improvement: How much better is the `denoised_ocr_output` compared to the original `ocr_text`? 
        Input:
        clean_text: {input_texts["clean_text"]}
        ocr_text: {input_texts["ocr_text"]}
        denoised_texts: {denoised_texts}
        Respond with a single line:  
        “Ranking: [5,4,2, …]”  
        where numbers correspond to output indices, sorted from best to worst. No extra text."""
    elif evaluation_technique == "pairwise":
        prompt = f"""
            You are given a clean reference (`clean_text`), a noisy OCR output (`ocr_text`), and two denoised versions (`denoised_ocr_text_1` and `denoised_ocr_text_2`).

            Evaluate which of the two denoised texts is better based on the following **4 criteria**, comparing both to either `clean_text` or `ocr_text` as instructed:

            Criteria (assign a point to the better output for each):
            1. **Faithfulness** — Does the denoised text preserve the meaning of `clean_text`?
            2. **Fluency** — Is the text grammatically correct and naturally flowing?
            3. **Completeness** — Is any information missing compared to `ocr_text`?
            4. **No Hallucinations** — Are there added or altered details that were not present in `ocr_text`?

            The output with more total points is the overall winner. In case of a tie, the winner is the one better in **faithfulness**.

            ### Output format:
            Respond **only** with a valid JSON string exactly matching this format (no quotes, no explanations, no extra text):

            {{
            "faithfulness": 1,           // 1 if denoised_ocr_text_1 is better, 2 otherwise
            "fluency": 2,
            "completeness": 1,
            "no_hallucinations": 2,
            "overall_winner": 2
            }}

            Inputs:
            clean_text: {input_texts["clean_text"]}
            ocr_text: {input_texts["ocr_text"]}
            denoised_ocr_text_1: {input_texts["denoised_text_1"]}
            denoised_ocr_text_2: {input_texts["denoised_text_2"]}
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

    retries = 0
    while retries <= MAX_RETRIES:
        try:
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                print(chunk.text, end="")
                response_text += chunk.text

            return response_text
        except errors.ServerError as e:
            if e.status == "UNAVAILABLE" and '503' in str(e):
                wait_time = 2 ** retries  # exponential backoff: 1,2,4,8,16 seconds
                print(f"\nServer overloaded (503). Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise  # re-raise unexpected errors
    raise RuntimeError("Max retries exceeded due to server overload.")


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
