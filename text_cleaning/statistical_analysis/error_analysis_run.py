

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

# train_keys, test_keys = error_analysis.train_test_split(keys)
#train_keys = keys

mistakes = {}
all_space_added = 0
all_space_subtracted = 0
visual_level_mistakes = None

for key in keys:
    excerpt_gt = data_gt[keys[int(key)]]
    excerpt_ocr = data_ocr[keys[int(key)]]

    excerpt_gt_words = error_analysis.tokenize_into_words(excerpt_gt)
    excerpt_ocr_words = error_analysis.tokenize_into_words(excerpt_ocr)

    # print(f'tokenized ocr {excerpt_ocr_words}')
    # print(f'tokenized gt {excerpt_gt_words}')


    mistakes, space_added,space_subtracted, spaces_count,visual_level_mistakes = error_analysis.align_words(excerpt_gt_words,excerpt_ocr_words,mistakes,visual_level_mistakes)


    all_space_added += spaces_count[0]
    all_space_subtracted += spaces_count[1]
    # mistakes.sort(key = lambda x: -x[1])
    # print(mistakes)

    # print(f'count of added spaces: {spaces_count[0]}')
    # print(f'count of deleted spaces: {spaces_count[1]}')

    # print('space subtracted examples')
    # for element in space_subtracted:
    #     print(element)
    # print('space added examples')
    # for element in space_added:
    #     print(element)

'''sorting the mistake type by the occurence frequency'''
space_mistakes = {"space_added":all_space_added,"space_subtracted":all_space_subtracted}
mistakes,visual_level_mistakes,space_mistakes = error_analysis.normalize_counts([mistakes,visual_level_mistakes,space_mistakes])
common_mistakes = error_analysis.cluster_mistakes(mistakes)
mistakes = [(key,value) for key,value in mistakes.items()]
mistakes.sort(key = lambda x: -x[1])
print(mistakes)
print(all_space_added)
print(all_space_subtracted)





#print(visual_level_mistakes)
common_visual_mistakes = error_analysis.cluster_mistakes(visual_level_mistakes)


common_mistakes.sort(key = lambda x: -x[1])
#print('the most common character level mistakes found in the text')
#print(common_mistakes)
for element in common_mistakes:
     print(str(element))
#print('the most common visual level - ie encompasing more than one character mistakes'\
#'found in the text')
#for element in common_visual_mistakes:
     #print(str(element))

with open(path+file_write, 'w',encoding='utf-8') as write_file:
     write_file.write('the most common mistakes on the one character level')
     write_file.write(str(common_mistakes))
     write_file.write('\n')
     write_file.write('the most common visual mistakes')
     #for element in common_visual_mistakes:
     write_file.write(str(common_visual_mistakes))
     write_file.write('\n')
     write_file.write('the mistakes regarding the spaces')
     write_file.write(str(space_mistakes))
     write_file.close()

