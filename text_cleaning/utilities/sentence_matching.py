import difflib
import Levenshtein
import string
alphabet = list(string.ascii_lowercase)

def extract_char_subs(ocr, gt):
    ops = Levenshtein.editops(ocr, gt)
    subs = []
    for tag, i, j in ops:
        if tag == 'replace':
            subs.append((ocr[i], gt[j]))
    return subs

def align_words(gt_words, ocr_words):
    i, j = 0, 0 
    aligned = []
    space_added = 0
    space_subtracted = 0
    char_level_edits = {[x,y]:0 for x in alphabet for y in alphabet}

    while i < len(gt_words) and j < len(ocr_words):
        gt_word = gt_words[i]
        ocr_word = ocr_words[j]

        # Direct match or small edit distance
        direct_dist = Levenshtein.distance(gt_word, ocr_word)
        merge_dist = float('inf')
        split_dist = float('inf')

        # Try merging two OCR words (maybe bad split)
        if len(gt_word) <  len(ocr_word):
            if j + 1 < len(ocr_words):
                merged_ocr = ocr_word + ocr_words[j + 1]
                merge_dist = Levenshtein.distance(gt_word, merged_ocr)

        # Try merging two GT words (maybe missing space)
        elif len(ocr_word) < len(gt_word):
            if i + 1 < len(gt_words):
                merged_gt = gt_word + gt_words[i + 1]
                split_dist = Levenshtein.distance(merged_gt, ocr_word)

        # Choose the lowest cost operation
        if direct_dist <= merge_dist and direct_dist <= split_dist:
            aligned.append((gt_word, ocr_word, "direct"))
            subs = extract_char_subs(gt_word, ocr_word)
            for sub in subs:
                char_level_edits[sub] +=1
            i += 1
            j += 1
        elif merge_dist < split_dist:
            aligned.append((gt_word, merged_ocr, "merge_ocr"))
            subs = extract_char_subs(gt_word, merged_ocr)
            for sub in subs:
                char_level_edits[sub] +=1
            space_subtracted +=1
            i += 1
            j += 2
        else:
            aligned.append((merged_gt, ocr_word, "merge_gt"))
            subs = extract_char_subs(merged_gt, ocr_word)
            for sub in subs:
                char_level_edits[sub] +=1
            space_added +=1
            i += 2
            j += 1

    return char_level_edits
