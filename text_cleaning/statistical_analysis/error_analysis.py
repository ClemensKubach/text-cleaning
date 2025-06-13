import Levenshtein  # Library for calculating string edit distances
import string
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter 
import json


VISUAL_PATTERNS = {
    'rn': 'm',
    'cl': 'd',
    'vv': 'w',
    'm' :  'rn',
    'd' : 'cl',
    'w' : 'vv'
}
VISUAL_PATTERNS_BIGRAMS = ['rn','cl','vv']

def count_all_characters_of_interest(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Load the JSON data
    
    # Combine all text values from the JSON dictionary
    text = ' '.join(str(value) for value in data.values())
    
    # Remove newlines and other unwanted whitespace
    text = text.replace('\n', '').replace('\r', '').strip()
    
    # Initialize counts
    char_counts = dict()
    
    # Initialize bigrams with 0 counts
    for bigram in VISUAL_PATTERNS_BIGRAMS:
        char_counts[bigram] = 0
    
    # Count characters and bigrams
    for i in range(len(text)):
        # Count single characters
        char = text[i]
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] =1
        
        # Count bigrams (only if we're not at the last character)
        if i < len(text) - 1:
            bigram = text[i:i+2]
            if bigram in VISUAL_PATTERNS_BIGRAMS:
                char_counts[bigram] += 1
                
    return char_counts  # Convert back to regular dict

def normalize_counts(count_dictionaries):
    s = 0
    for item in count_dictionaries:
        s +=  sum([v for v in item.values()])
    return [{k:v/s for k,v in dictionary.items()} for dictionary in count_dictionaries]

def normalize_counts_letter(dictionary,occurrence_dictionary):
    
    dictionary = {key:value for key,value in dictionary.items() if value!=0}
    return {k:round(v/occurrence_dictionary[k[1]],3) for k,v in dictionary.items()}



def detect_visual_errors(ocr_word, gt_word,visual_mistakes_matches):
    #visual_mistakes_matches = {(k,v):0 for k,v in VISUAL_PATTERNS.items()}
    for pattern, true_char in VISUAL_PATTERNS.items():
        i = 0
        while i < len(ocr_word) and i < len(gt_word):
            if len(pattern) ==2:
                if ocr_word[i:i+2] == pattern and  gt_word[i]== true_char:
                    visual_mistakes_matches[(pattern,true_char)] +=1
            else:
                if gt_word[i:i+2] == true_char and ocr_word[i] ==pattern:
                    visual_mistakes_matches[(pattern,true_char)] +=1
            i+=1

    return visual_mistakes_matches


def split_dataset(keys):
    X_train, X_test = train_test_split(keys, test_size=0.2, random_state=42)
    return X_train, X_test


def cluster_mistakes(substitutions):
    counts = np.array(list(substitutions.values())).reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=42).fit(counts)
    labels = kmeans.labels_
    frequent_cluster = np.argmax(kmeans.cluster_centers_)
    frequent_mistakes = [(k,substitutions[k]) for k, v in zip(substitutions.keys(), counts) if labels[counts.tolist().index([v])] == frequent_cluster]

    return  frequent_mistakes


# Tokenize a sentence into a list of words
def tokenize_into_words(sentence) -> list[str]:
    words = sentence.split()  # Split by spaces
    return words

# Extract character-level substitutions from the Levenshtein edit operations
def extract_char_subs(ocr, gt) -> list[tuple[str, str]]:
    ops = Levenshtein.editops(ocr, gt)  # Get list of edit operations to convert ocr -> gt
    subs = []
    for tag, i, j in ops:
        if tag == 'replace':
            subs.append((ocr[i], gt[j]))  # Collect character substitutions
    return subs

# Align words from ground truth and OCR output and collect character-level edits and space errors
def align_words(gt_words, ocr_words, char_level_edits,visual_level_edits=None) -> tuple[dict[tuple[str, str], int], list[list[tuple[str, str]]], list[list[tuple[tuple[str, str], str]]], list[int],dict[tuple[str, str], int]]:

    i, j = 0, 0  # Indices for ground truth and OCR word lists
    aligned = []  # List to store aligned word pairs

    # Space error counters and example collectors
    space_added_count = 0
    space_subtracted_count = 0
    space_added_examples =  []
    space_subtracted_examples = []

    # Initialize character edit dict if not provided
    if not char_level_edits:
        char_level_edits = {}
    if not visual_level_edits:
        visual_level_edits = {(k,v):0 for k,v in VISUAL_PATTERNS.items()}

    # Iterate over both gt_words and ocr_words
    while i < len(gt_words) and j < len(ocr_words):
        gt_word = gt_words[i]
        ocr_word = ocr_words[j]

        # Compute edit distance between current gt and ocr words
        direct_dist = Levenshtein.distance(gt_word, ocr_word)
        merge_dist = float('inf')  # Distance if two OCR words are merged
        split_dist = float('inf')  # Distance if two GT words are merged

        # Try merging two OCR words (case: OCR added space)
        if len(gt_word) > len(ocr_word):
            if j + 1 < len(ocr_words):
                merged_ocr = ocr_word + ocr_words[j + 1]
                merge_dist = Levenshtein.distance(gt_word, merged_ocr)

        # Try merging two GT words (case: OCR missed space)
        elif len(ocr_word) > len(gt_word):
            if i + 1 < len(gt_words):
                merged_gt = gt_word + gt_words[i + 1]
                split_dist = Levenshtein.distance(merged_gt, ocr_word)

        # Choose the operation with the lowest edit distance cost
        if direct_dist <= merge_dist and direct_dist <= split_dist:
            aligned.append((gt_word, ocr_word))
            visual_level_edits = detect_visual_errors(ocr_word,gt_word,visual_level_edits)
            subs = extract_char_subs(ocr_word,gt_word)
            for sub in subs:
                if sub in char_level_edits:
                    char_level_edits[sub] += 1
                else:
                    char_level_edits[sub] = 1
            i += 1
            j += 1

        # If merging OCR words gives better match (OCR added space)
        elif merge_dist < split_dist:
            aligned.append((gt_word, (ocr_word, ocr_words[j+1])))
            visual_level_edits = detect_visual_errors(merged_ocr,gt_word,visual_level_edits)
            subs = extract_char_subs( merged_ocr, gt_word)
            for sub in subs:
                if sub in char_level_edits:
                    char_level_edits[sub] += 1
                else:
                    char_level_edits[sub] = 1
            space_added_count += 1
            space_added_examples.append([(gt_word), (ocr_word, ocr_words[j+1])])
            i += 1
            j += 2

        # If merging GT words gives better match (OCR missed space)
        else:
            aligned.append(((gt_word, gt_words[i+1]), ocr_word))
            visual_level_edits = detect_visual_errors(ocr_word,merged_gt,visual_level_edits)
            subs = extract_char_subs(ocr_word,merged_gt)
            for sub in subs:
                if sub in char_level_edits:
                    char_level_edits[sub] += 1
                else:
                    char_level_edits[sub] = 1
            space_subtracted_count += 1
            space_subtracted_examples.append([((gt_word, gt_words[i+1]), ocr_word)])
            i += 2
            j += 1

    # Print aligned word pairs for debugging
    print(aligned)

    # Return:
    # - Dictionary of character substitutions with counts
    # - Examples of OCR words with added spaces
    # - Examples of OCR words with missing spaces
    # - Count of space addition and subtraction issues
    return char_level_edits, space_added_examples, space_subtracted_examples, [space_added_count, space_subtracted_count], visual_level_edits
