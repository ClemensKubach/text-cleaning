import json

TEXT_PATH = "/workspaces/mnlp_project_2/data/ocr_datasets/eng/"
CLEAN_TEXT = TEXT_PATH + "the_vampyre_clean.json"
OCR_TEXT = TEXT_PATH + "the_vampyre_ocr.json"
SCORES_PATH = "/workspaces/mnlp_project_2/data/evaluations/judge_evaluations/"

with open(CLEAN_TEXT, "r") as clean:
    data = json.load(clean)

"""The json keys (numbers of the fragments) extracted """
keys = [key for key in data]

data = [[key,len(data[key])] for key in keys]
data.sort(key = lambda x:x[1])
print(data)