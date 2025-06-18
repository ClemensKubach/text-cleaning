import json
from text_cleaning.statistical_analysis import error_analysis



path = './data/'
file_gt = 'ocr_datasets/eng/the_vampyre_clean.json'
file_ocr = 'ocr_datasets/eng/the_vampyre_ocr.json'
file_write = 'statistical_analysis/statistical_mistake_analysis_normalized_all.txt'


with open(path + file_gt, "r") as gt:
    data_gt = json.load(gt)

with open(path + file_ocr, "r") as ocr:
    data_ocr = json.load(ocr)

"""The json keys (numbers of the fragments) extracted """
keys = [key for key in data_ocr]


mistakes = {}
all_space_added = 0
all_space_subtracted = 0
visual_level_mistakes = None

for key in keys:
    excerpt_gt = data_gt[keys[int(key)]]
    excerpt_ocr = data_ocr[keys[int(key)]]

    excerpt_gt_words = error_analysis.tokenize_into_words(excerpt_gt)
    excerpt_ocr_words = error_analysis.tokenize_into_words(excerpt_ocr)

    mistakes, space_added, space_subtracted, spaces_count, visual_level_mistakes = error_analysis.align_words(
        excerpt_gt_words, excerpt_ocr_words, mistakes, visual_level_mistakes
    )

    all_space_added += spaces_count[0]
    all_space_subtracted += spaces_count[1]



'''changing the space mistakes into proper format for the normalization algorithm'''
space_added_mistakes = {(' ',''):all_space_added}
space_subtracted_mistakes = {('',' '): all_space_subtracted}
'''most commonly happening mistakes normalized by the whole mistakes coount  '''
counted_characters = error_analysis.count_all_characters_of_interest(path+file_gt)
mistakes.update(space_subtracted_mistakes)
mistakes.update(space_added_mistakes)
mistakes.update(visual_level_mistakes)
print(mistakes)
mistakes_normalized_letter = error_analysis.normalize_counts(mistakes)
'''the mistakes clustered to determine which are common which are not common - 2 clusters '''
mistakes_normalized_letter = error_analysis.cluster_mistakes(mistakes_normalized_letter,num_clusters=2,clusters_to_return=1,verbose=False)
mistakes_normalized_letter.sort(key=lambda x: -x[1])





with open(path + file_write, "w", encoding="utf-8") as write_file:
    write_file.write(" the most common character substitution mistakes ocr/gt to gt ratio")
    write_file.write(str(mistakes_normalized_letter))
    write_file.write("\n")
   
