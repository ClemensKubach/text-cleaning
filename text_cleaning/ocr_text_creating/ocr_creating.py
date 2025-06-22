

from text_cleaning.constants import DATA_DIR
from text_cleaning.utils import load_data, save_data
import math 
import difflib
import random
from nltk.tokenize import sent_tokenize
from collections import Counter
from datasets import load_dataset
from text_cleaning.statistical_analysis.error_analysis import cluster_mistakes
from text_cleaning.statistical_analysis.error_analysis import count_all_characters_of_interest
from collections import Counter
import difflib

'''statistical analysisg on the big ocr corpus'''

file_write = '/workspaces/mnlp_project_2/text_cleaning/ocr_text_creating/statistical_mistakes_analysis.txt'




TEXT_PATH = "/workspaces/mnlp_project_2/data/ocr_datasets/eng/"
CLEAN_TEXT = TEXT_PATH + "the_vampyre_clean.json"
OCR_TEXT = TEXT_PATH + "the_vampyre_ocr.json"





"the function responsible for the dividing the clean text into chunks, so that to allign with the ocr text "
"respective chunk "
def chunk_clean_text(text, target_len=200):
    chunks = []
    start = 0
    toleration_dot = target_len //3
    toleration_space = target_len //10
    while start < len(text):
        end = start + target_len
        j = 0
        k = 0
        while end < len(text) and text[end] != '.' and j <= toleration_dot:
            end += 1
            j += 1
        if end < len(text) and text[end] != '.':
            while  end < len(text) and (text[end] != ' ' and text[end] != '.') and k <= toleration_space:
                end += 1 
                k += 1
        j = 0 
        k = 0
        chunk = text[start:end+1].strip()
        chunks.append(chunk)
        start = end + 1
    return chunks





def find_best_match_chunk_by_ratio(clean_chunk, ocr_text, start_hint=0, search_window=1000, stride=5):
    """
    Search for the region in OCR text that is most similar to clean_chunk using similarity ratio.
    The method is used to allign the chuks, we strive to find the most similar chunks (of the same length)
    in the predefined size window, 
    we try to match eaxact chunks with a small stride of 5 change - allowing for some inconsistencies
    but faciliating the computational speed
    we do not allign anything if the quality of the text is exceptionally poor < below 0.6 
    that allowed us to pick approximately 40% of the iterated data
    """
    search_start = max(0, start_hint - search_window // 2)
    search_end = min(len(ocr_text), search_start + search_window)

    best_score = 0
    best_region = ""
    best_offset = start_hint

    for i in range(search_start, search_end, stride):
        window = ocr_text[i:i+len(clean_chunk)]
        score = difflib.SequenceMatcher(None, clean_chunk, window).ratio()
        if score > best_score:
            best_score = score
            best_region = window
            best_offset = i
        
    if best_score > 0.6:
        return best_region, best_offset + len(best_region)
    else:
        return  None, best_offset + len(best_region)






def get_ocr_errors(clean, ocr,all_characters_count):
    """Compares two strings and returns character-level substitution, insertion, and deletion counts."""
    sm = difflib.SequenceMatcher(None, clean, ocr)
    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()
    all_characters_count.update(clean)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':
            src = clean[i1:i2]
            tgt = ocr[j1:j2]
        
            for c, o in zip(src, tgt):
                if c != o:
                    substitutions[(c, o)] += 1
            
            if len(src) > len(tgt):
                for c in src[len(tgt):]:
                    deletions[c] += 1
            elif len(tgt) > len(src):
                for c in tgt[len(src):]:
                    insertions[c] += 1

        elif tag == 'insert':
            for c in ocr[j1:j2]:
                insertions[c] += 1


        elif tag == 'delete':
            for c in clean[i1:i2]:
                deletions[c] += 1

    return substitutions, insertions, deletions,all_characters_count





def analyze_ocr_errors_from_chunks(clean_chunks, ocr_chunks,all_characters_count):
    """
    Compares aligned text chunks (may include multiple sentences).
    Returns aggregated counts of substitutions, insertions, deletions.
    """
    assert len(clean_chunks) == len(ocr_chunks), "Chunks must be aligned and equal in number."

    total_subs = Counter()
    total_ins = Counter()
    total_dels = Counter()

    for clean_chunk, ocr_chunk in zip(clean_chunks, ocr_chunks):
        if ocr_chunk is None:
            continue 
        subs, ins, dels, all_characters_count = get_ocr_errors(clean_chunk, ocr_chunk,all_characters_count)
        total_subs.update(subs)
        total_ins.update(ins)
        total_dels.update(dels)

    return total_subs, total_ins, total_dels, all_characters_count





dataset_ocr = []
dataset_clean = []
#dataset = load_dataset("PleIAs/Post-OCR-Correction", name="english", split="train[:50]")  # Load a small subset

'''using the biggest aggregated dataset with the ocr content and corrected by the state of the art
correction algorithms content'''
"we iterated over the dataset to get ranodm example because the dataset is imbalanced and the data is"
"in chronological and newspaper type order, and we didnt want to skew the error distribution just by"
" introducing the  bias, by the certain  font of newspaper, certain epoch etc"
dataset_iterator = load_dataset("PleIAs/Post-OCR-Correction", name="english", split="train", streaming=True)


'''used to randomly pick some fraction of the dataset to use here'''
"since whole dataser has 30000 training examples it gives around 600 "
DATASET_FRACTION = 0.02
# dataset  = []
for i, row in enumerate(dataset_iterator):
      if random.random() < DATASET_FRACTION:  
          dataset_ocr.append(row["text"])
          dataset_clean.append(row["corrected_text"])




clean_chunks = []
ocr_chunks = []
"""here we perfrom the allignment of the ground truth with the ocr text
It is not a trivial task unfortunately, due to often times deletion of the input but the ocr
or introducing many additional characters due to the the blurred words etc"""
for i in range(len(dataset_clean)):
    clean_chunks_current = chunk_clean_text(dataset_clean[i])
    ocr_pos_current = 0
    ocr_chunks_current = []
    for clean_chunk in clean_chunks_current:
        ocr_chunk,ocr_pos_current = find_best_match_chunk_by_ratio(clean_chunk, dataset_ocr[i], start_hint=ocr_pos_current)
        ocr_chunks_current.append(ocr_chunk)
    clean_chunks.append(clean_chunks_current)
    ocr_chunks.append(ocr_chunks_current)



good_alligned_chunks = 0
for i in range(len(clean_chunks)):
    for index, element in enumerate(clean_chunks[i]):
        if ocr_chunks[i][index] is None:
            continue
        good_alligned_chunks +=1 
        if index < len(ocr_chunks[i]):
            pass
        else:
            pass

print(f"ratio of used chunks {good_alligned_chunks}")


subs = Counter()
ins = Counter()
dels = Counter()
all_characters_count =  Counter()

"""we iterate over all training examples and aggregate the misteks with the help of the counter """
for i in range(len(clean_chunks)):
    curr_subs, curr_ins, curr_dels, all_characters_count = analyze_ocr_errors_from_chunks(clean_chunks[i], ocr_chunks[i],all_characters_count)
    subs += curr_subs
    ins += curr_ins
    dels += curr_dels
   
    # print('\n')
    # print('insertions')
    # print(ins)
    # print('\n')
    # print('deletions')
    # print(dels)
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')

# dels = {key:val/all_characters_count[key] for key,val in dels.items()}
# ins = {
#     key: val / all_characters_count[key] if all_characters_count[key] != 0 else 0
#     for key, val in ins.items()
# }

# subs = {key:val/all_characters_count[key[0]] for key,val in subs.items()}

"""we normalize the mistakes by all mistakes, to see which ones had benn most prevalent ones"""
sum_dels = sum(dels.values())
sum_ins = sum(ins.values())
sum_subs = sum(subs.values())

dels = {key:val/sum_dels for key,val in dels.items()}
ins = {
     key: val/sum_ins for key,val in  ins.items()
 }
subs = {key:val/sum_subs for key,val in subs.items()}

del subs[(' ', '\n')]





print('\n')
print('insertions')
print(ins)
print('\n')
print('deletions')
print(dels)
print('\n')
print('\n')
print('\n')
print('\n')
print(subs)

print('good alligned chunks used')
print(good_alligned_chunks)


"""we cluster the mistakes to retain the one which are happening with some resonable frequency
we do not want to replicate the exact error distirbution but rather approximate it, and the errors with
frequency below a certain threshold could be just considered noise. and we do introduce the noise to the
distirbution later uniformly """
common_dels = cluster_mistakes(dels,num_clusters=5,clusters_to_return=3,verbose=False)
common_dels.sort(key=lambda x: -x[1])

common_ins = cluster_mistakes(ins,num_clusters=5,clusters_to_return=3,verbose=False)
common_ins.sort(key=lambda x: -x[1])

common_subs = cluster_mistakes(subs,num_clusters=5,clusters_to_return=3,verbose=False)
common_subs.sort(key=lambda x: -x[1])



print('\n')
print('common_ insertions')
print(common_ins)
print('\n')
print('common_ deletions')
print(common_dels)
print('\n')
print('\n')
print('common substitutions')
print(common_subs)
print('\n')
print('\n')


"""saving the output to the files"""
with open(file_write, "w", encoding="utf-8") as write_file:
    write_file.write("insertion mistakes")
    write_file.write('\n')
    write_file.write(str(common_ins))
    write_file.write("substitution mistakes")
    write_file.write("\n")
    write_file.write(str(common_subs))
    write_file.write("deletion mistakes")
    write_file.write("\n")
    write_file.write(str(common_dels))
   






