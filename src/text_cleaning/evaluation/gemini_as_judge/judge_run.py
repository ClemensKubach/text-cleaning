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
SCORES_PATH = "/workspaces/mnlp_project_2/data/evaluations/judge_evaluations/"

MAX_RETRIES = 16

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
        "randomly shuffling the order in the model to hinder the position bias"
        prob = random.random()
        changed_mapping = {"1":"2","2":"1","0":"0"}
        if prob <= 0.5:
            texts_dict["denoised_text_1"] = denoised_data[0][i]
            texts_dict["denoised_text_2"] = denoised_data[1][i]
            evaluation = generate(evaluation_technique=evaluation_technique, input_texts=texts_dict)
            if not is_json(evaluation):
                return 0, scores_file_path
        else:
            texts_dict["denoised_text_1"] = denoised_data[1][i]
            texts_dict["denoised_text_2"] = denoised_data[0][i]
            evaluation = generate(input_texts=texts_dict)
            if not is_json(evaluation):
                return 0, scores_file_path
            for key in evaluation:
                evaluation[key] = changed_mapping[evaluation[key]]
            
            scores[i] = evaluation

    str_input_paths = ""
    str_input_paths += input_paths[0].split("_")[0]
    str_input_paths += "_"
    str_input_paths += input_paths[1].split("_")[0]

    scores_file_path = SCORES_PATH + f"{str_input_paths}" +".json"
    save_data(scores_file_path, scores)
    logger.info(f"Scores saved to {scores_file_path}")
    return scores, scores_file_path


def generate( input_texts: dict[str]):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    prompt = f"""
        You are given a clean reference (`clean_text`), a noisy OCR output (`ocr_text`), and two denoised versions (`denoised_ocr_text_1` and `denoised_ocr_text_2`).

        Evaluate which of the two denoised texts is better based on the following **4 criteria**, comparing both to either `clean_text` or `ocr_text` as instructed:

        ### Criteria:
        1. **Faithfulness** — Does the denoised text preserve the meaning of `clean_text`?
        2. **Fluency** — Are the words spelled correctly and is the text grammatically correct?
        3. **Completeness** — Is any information missing compared to `ocr_text`?
        4. **No Hallucinations** — Are there added or altered details that were not present in `ocr_text`?

        ### Scoring:
        - For each criterion, assign:
        - `1` if `denoised_ocr_text_1` is better,
        - `2` if `denoised_ocr_text_2` is better,
        - `0` if they are equal.

        - The output with **more total points** (sum of 1s or 2s) is the `overall_winner`.

        - In case of a tie (e.g., equal total points), break the tie using this priority order:
        1. Fluency
        2. Faithfulness
        3. No Hallucinations
        4. Completeness

        - If all criteria scores are `0`, select either output `1` or `2` randomly.

        ### Example scoring output format:
        {{
        "faithfulness": 1,
        "fluency": 0,
        "completeness": 2,
        "no_hallucinations": 2,
        "overall_winner": 2
        }}

        Below are examples illustrating differences for each category. Note that for each example, only one category differs (scored 1 or 2), and the others are equal `0`.

        ---

        Example 1 (Faithfulness difference):  
        `clean_text`: 'the cat sat on an apartment roof'  
        `ocr_text`: 'the ca t sat on an apartment roof'  
        `denoised_ocr_text_1`: 'the cat sat on a house roof'  
        `denoised_ocr_text_2`: 'the cat sat on a apartment roof'  
        output{{  
        "faithfulness": 2,  
        "fluency": 0,  
        "completeness": 0,  
        "no_hallucinations": 0,  
        "overall_winner": 2  
        }}

        ---

        Example 2 (Fluency difference):  
        `clean_text`: 'there was not any light nor dark'  
        `ocr_text`: 'thele w as notany light nor dark'  
        `denoised_ocr_text_1`: 'there was not any light nor dark'  
        `denoised_ocr_text_2`: 'thele was notany light no dark'  
        output{{  
        "faithfulness": 0,  
        "fluency": 1,  
        "completeness": 0,  
        "no_hallucinations": 0,  
        "overall_winner": 1  
        }}

        ---

        Example 3 (Completeness difference):  
        `clean_text`: 'my guest stood up and took a drink'  
        `ocr_text`: 'my gue@t stod up and took a drink'  
        `denoised_ocr_text_1`: 'my guest stood up'  
        `denoised_ocr_text_2`: 'my guest stood up and took a drink'  
        output{{  
        "faithfulness": 0,  
        "fluency": 0,  
        "completeness": 2,  
        "no_hallucinations": 0,  
        "overall_winner": 2  
        }}

        ---

        Example 4 (No hallucinations difference):  
        `clean_text`: 'I was devastated by the information of'  
        `ocr_text`: 'I was dewaStated by theinformation of'  
        `denoised_ocr_text_1`: 'I was devastated by the information of'  
        `denoised_ocr_text_2`: 'I was devastated by the information of the unexpected earthquake'  
        output{{  
        "faithfulness": 0,  
        "fluency": 0,  
        "completeness": 0,  
        "no_hallucinations": 1,  
        "overall_winner": 1  
        }}

        ---

        Inputs for evaluation:  
        `clean_text`: {input_texts["clean_text"]}  
        `ocr_text`: {input_texts["ocr_text"]}  
        `denoised_ocr_text_1`: {input_texts["denoised_text_1"]}  
        `denoised_ocr_text_2`: {input_texts["denoised_text_2"]}
        """


    
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
        "--input_names",
        nargs="+",
        type=str,
        required=True,
        help="the name of the input  files that are to be evaluated.the format should be "
        "name_1,name_2 "
    )
    args = parser.parse_args()
    evaluation_technique = args.evaluation_technique
    input_paths = [element for element in args.input_names]
    evaluate_dataset(input_paths=input_paths, evaluation_technique=evaluation_technique)
