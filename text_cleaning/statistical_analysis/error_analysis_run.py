

import json
from text_cleaning.statistical_analysis import error_analysis


path = './data/'
file_gt = 'ocr_datasets/eng/the_vampyre_clean.json'
file_ocr = 'ocr_datasets/eng/the_vampyre_ocr.json'
file_write = 'statistical_analysis/statistical_mistake_analysis.txt'

with open(path+file_gt, 'r') as gt:
     data_gt = json.load(gt)

with open(path+file_ocr, 'r') as ocr:
     data_ocr = json.load(ocr)

'''The json keys (numbers of the fragments) extracted '''
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


    mistakes, space_added,space_subtracted, spaces_count,visual_level_mistakes = error_analysis.align_words(excerpt_gt_words,excerpt_ocr_words,mistakes,visual_level_mistakes)


    all_space_added += spaces_count[0]
    all_space_subtracted += spaces_count[1]
    
# print(mistakes.keys())

'''changing the space mistakes into proper format for the normalization algorithm'''
space_added_mistakes = {(' ',' '):all_space_added}
space_subtracted_mistakes = {(' ',' '): all_space_subtracted}
'''most commonly happening
character level mistakes normalized by the count of the occurrence of the ground truth letter '''
counted_characters = error_analysis.count_all_characters_of_interest(path+file_gt)
mistakes_normalized_letter = error_analysis.normalize_counts_letter(mistakes,counted_characters)
mistakes_normalized_letter = error_analysis.cluster_mistakes(mistakes_normalized_letter,num_clusters=3,clusters_to_return=2)
mistakes_normalized_letter.sort(key=lambda x: -x[1])


mistakes_normalized_visual = error_analysis.normalize_counts_letter(visual_level_mistakes,counted_characters)
mistakes_normalized_visual = [[k,v] for k,v in mistakes_normalized_visual.items()]
mistakes_normalized_visual.sort(key=lambda x: -x[1])

mistakes_normalized_space_added = error_analysis.normalize_counts_letter(space_added_mistakes,counted_characters)
mistakes_normalized_space_subtracted = error_analysis.normalize_counts_letter(space_subtracted_mistakes,counted_characters)

# print(mistakes_normalized_letter)
# print(mistakes_normalized_visual)
# print(mistakes_normalized_space_added)
# print(mistakes_normalized_space_subtracted)




with open(path+file_write, 'w',encoding='utf-8') as write_file:
     

     write_file.write(' the most common character substitution mistakes ocr/gt to gt ratio')
     write_file.write(str(mistakes_normalized_letter))
     write_file.write('\n')
     write_file.write(' the most common visual substitution mistakes ocr/gt to gt ratio ')
     write_file.write(str(mistakes_normalized_visual))
     write_file.write('\n')
     write_file.write(' the number of mistakenly added space with respect to all spaces in the ground truth')
     write_file.write(str(mistakes_normalized_space_added))
     write_file.write('\n')
     write_file.write(' the number of mistakenly subtracted spaces with respect to all spaces in the ground truth')
     write_file.write(str(mistakes_normalized_space_subtracted))
     write_file.write('\n')
     write_file.close()

