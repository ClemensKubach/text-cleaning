from typing import Dict, List, Tuple
import random
import json
import string
from collections import defaultdict
import re

"""creating the ocr datasets"""

clean_json_path = "/workspaces/mnlp_project_2/text_cleaning/ocr_text_creating/clean_otoranto.json"
ocr_json_path = "/workspaces/mnlp_project_2/text_cleaning/ocr_text_creating/ocr_otoranto.json"
text_path = "/workspaces/mnlp_project_2/text_cleaning/ocr_text_creating/otoranto_castel.txt"

# Substitution: (gt, ocr) → prob hardcoded by the probabilities from the big dataset
substitution_probs_raw = [
    (("e", "o"), 0.02855834703760925),
    (("i", "I"), 0.023725447043249477),
    (("i", "l"), 0.018237838399768197),
    (("a", "n"), 0.013158316735215694),
    ((",", " "), 0.012422875431726163),
    ((" ", ","), 0.009645893306820582),
    (("e", "c"), 0.009580643627262865),
    (("t", " "), 0.009477792437451546),
    (("a", " "), 0.00855102580237752),
    ((" ", "t"), 0.00825463742743738),
    (("s", "a"), 0.008229201111677591),
    ((" ", "e"), 0.00804561726749825),
    (("e", " "), 0.008004697976928156),
    (("s", " "), 0.0067937081614078),
    (("o", " "), 0.006610124317228459),
    ((" ", "o"), 0.006522756102227447),
    (("i", " "), 0.006435387887226435),
    (("e", "t"), 0.006215308459565658),
    ((" ", "n"), 0.00611245726975434),
    (("e", "r"), 0.005732018460129681),
    (("n", "a"), 0.0056888873160152575),
    (("h", "t"), 0.00558603612620394),
    ((" ", "s"), 0.005561705737216316),
    (("e", "i"), 0.005359321137910175),
    ((" ", "a"), 0.005182372854363821),
    (("n", "i"), 0.0051027461267679625),
    ((",", "."), 0.005093898712590645),
    (("i", "t"), 0.005004318644045303),
]

" Insertion: char → [(inserted_char, prob)] "
insertion_probs_raw = [
    (" ", 0.2189837029352008),
    ("e", 0.06249324557990749),
    ("t", 0.05445413478580383),
    ("o", 0.049165423853369646),
    ("n", 0.04589358276055851),
    ("i", 0.04451568106168677),
    ("a", 0.04201789651147711),
    ("r", 0.038470475078891626),
    ("l", 0.0358376021268318),
    ("s", 0.03217940820472918),
    (".", 0.030107152120347556),
    ("h", 0.024110577962218476),
    ("d", 0.020645560454761597),
]

" Deletion: char → prob "
deletion_probs_raw = [
    (" ", 0.14795397005573097),
    ("e", 0.09363043929585402),
    ("t", 0.06535782943144715),
    ("a", 0.056789228043970026),
    ("n", 0.05484525019615033),
    ("o", 0.054703698411114915),
    ("i", 0.05039243547260771),
    ("s", 0.04856709150177007),
    ("r", 0.04788090237240792),
    ("h", 0.03430541213139241),
    (".", 0.03202305811172615),
    ("l", 0.02980541347950465),
    ("d", 0.02895610276929216),
]


substitution_probs = defaultdict(list)
for (src, tgt), prob in substitution_probs_raw:
    substitution_probs[src].append((tgt, prob))

insertion_probs = defaultdict(list)
for char, prob in insertion_probs_raw:
    insertion_probs[char].append((char, prob))

deletion_probs = {char: prob for char, prob in deletion_probs_raw}


def perturb_chunk(
    chunk: str,
    substitution_probs: Dict[str, List[Tuple[str, float]]],
    insertion_probs: Dict[str, List[Tuple[str, float]]],
    deletion_probs: Dict[str, float],
    visual_noise_patterns: List[Tuple[str, str, float]],
    non_alpha_chars: List[str],
    non_alpha_insert_prob: float,
    uniform_noise_prob: float,
    uniform_noise_charset: str,
) -> str:
    perturbed = []
    i = 0
    while i < len(chunk):
        c = chunk[i]

        "Uniform noise, the normalization term, uniform noise possibility "
        if random.random() < uniform_noise_prob:
            perturbed.append(random.choice(uniform_noise_charset))
            i += 1
            continue

        " Apply visual pattern noise" 
        applied_visual = False
        for pattern, replacement, prob in visual_noise_patterns:
            if chunk[i : i + len(pattern)] == pattern and random.random() < prob:
                perturbed.append(replacement)
                i += len(pattern)
                applied_visual = True
                break
        if applied_visual:
            continue

        " Deletion"
        if c in deletion_probs and random.random() < deletion_probs[c]:
            i += 1
            continue

        " Substitution" 
        substituted = False
        if c in substitution_probs:
            for wrong_char, prob in substitution_probs[c]:
                if random.random() < prob:
                    perturbed.append(wrong_char)
                    substituted = True
                    break
        if not substituted:
            perturbed.append(c)

        " Insertion"
        if c in insertion_probs:
            for ins_char, prob in insertion_probs[c]:
                if random.random() < prob:
                    rand_shift = random.randint(5, 15)
                    if rand_shift < len(perturbed):
                        perturbed = (
                            perturbed[: len(perturbed) - rand_shift]
                            + [ins_char]
                            + perturbed[len(perturbed) - rand_shift :]
                        )
                    # perturbed.append(ins_char)

        " Non-alphanumeric  insertion"
        if random.random() < non_alpha_insert_prob:
            perturbed.append(random.choice(non_alpha_chars))

        i += 1

    return "".join(perturbed)


def generate_ocr_dataset(
    text: str,
    min_len: int,
    max_len: int,
    substitution_probs,
    insertion_probs,
    deletion_probs,
    visual_noise_patterns,
    non_alpha_chars,
    non_alpha_insert_prob,
    uniform_noise_prob,
    uniform_noise_charset,
):
    clean_json = {}
    ocr_json = {}

    i = 0
    idx = 0

    while i < len(text):
        "random length of the chunk, in the range of the lengths of the chunks in the vampyre"
        chunk_len = random.randint(min_len, max_len)
        approx_end = i + chunk_len

        if approx_end >= len(text):
            approx_end = len(text)

        "trying to split on the . to do it most naturally"
        window_start = max(i, approx_end - 100)
        window_end = min(len(text), approx_end + 100)

        " Find nearest '.' in the window" 
        nearby_text = text[window_start:window_end]
        dot_positions = [m.start() for m in re.finditer(r"\.", nearby_text)]

        if dot_positions:
            " Choose closest '.' to approx_end" 
            closest_dot = min(dot_positions, key=lambda x: abs((window_start + x) - approx_end))
            end = window_start + closest_dot + 1  
        else:
            end = approx_end

        chunk = text[i:end]


        if not chunk.strip():
            i = end
            continue

        "saving the texts to the files in json format"
        clean_json[str(idx)] = chunk
        ocr_json[str(idx)] = perturb_chunk(
            chunk,
            substitution_probs,
            insertion_probs,
            deletion_probs,
            visual_noise_patterns,
            non_alpha_chars,
            non_alpha_insert_prob,
            uniform_noise_prob,
            uniform_noise_charset,
        )

        i = end
        idx += 1

    return clean_json, ocr_json


with open(text_path, "r") as f:
    input_text = f.read()
    f.close()

clean_json, ocr_json = generate_ocr_dataset(
    text=input_text,
    min_len=1000,
    max_len=2000,
    substitution_probs=substitution_probs,
    insertion_probs=insertion_probs,
    deletion_probs=deletion_probs,
    visual_noise_patterns=[("rn", "m", 0.005), ("cl", "d", 0.005), ("vv", "w", 0.005)],  # example visual confusions
    non_alpha_chars=list("~!$%&*+=?|"),
    non_alpha_insert_prob=0.005,
    uniform_noise_prob=0.005,
    uniform_noise_charset=string.ascii_letters + string.digits + string.punctuation,
)

" Saving to the files"
with open(clean_json_path, "w") as f:
    json.dump(clean_json, f, indent=2)
with open(ocr_json_path, "w") as f:
    json.dump(ocr_json, f, indent=2)
