

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
print(mistakes.keys())

'''changing the space mistakes into proper format for the normalization algorithm'''
space_added_mistakes = {(' ',' '):all_space_added}
space_subtracted_mistakes = {(' ',' '): all_space_subtracted}
'''most commonly happening
character level mistakes normalized by the count of the occurence of the ground truth letter '''
counted_characters = error_analysis.count_all_characters_of_interest(path+file_gt)
print(counted_characters.keys())
mistakes_normalized_letter = error_analysis.normalize_counts_letter(mistakes,counted_characters)
#mistakes_normalized_letter_common = error_analysis.cluster_mistakes(mistakes_normalized_letter)
mistakes_normalized_letter = [[k,v] for k,v in mistakes_normalized_letter.items()]
mistakes_normalized_letter.sort(key=lambda x: -x[1])


mistakes_normalized_visual = error_analysis.normalize_counts_letter(visual_level_mistakes,counted_characters)
#mistakes_normalized_visual_common= error_analysis.cluster_mistakes(mistakes_normalized_visual)
mistakes_normalized_visual = [[k,v] for k,v in mistakes_normalized_visual.items()]
mistakes_normalized_visual.sort(key=lambda x: -x[1])

mistakes_normalized_space_added = error_analysis.normalize_counts_letter(space_added_mistakes,counted_characters)
mistakes_normalized_space_subtracted = error_analysis.normalize_counts_letter(space_subtracted_mistakes,counted_characters)

print(mistakes_normalized_letter)
print(mistakes_normalized_visual)
print(mistakes_normalized_space_added)
print(mistakes_normalized_space_subtracted)


'''most commonly happening mistakes by the unelative count of the them, not normalized by the gt letter frequency '''
# space_mistakes = {"space_added":all_space_added,"space_subtracted":all_space_subtracted}

# mistakes,visual_level_mistakes,space_mistakes = error_analysis.normalize_counts([mistakes,visual_level_mistakes,space_added_mistakes,space_subtracted_mistakes])
# common_mistakes = error_analysis.cluster_mistakes(mistakes)
# mistakes = [(key,value) for key,value in mistakes.items()]
# mistakes.sort(key = lambda x: -x[1])
# print(mistakes)
# print(all_space_added)
# print(all_space_subtracted)
# common_visual_mistakes = error_analysis.cluster_mistakes(visual_level_mistakes)
# common_mistakes.sort(key = lambda x: -x[1])
# #print('the most common character level mistakes found in the text')
# #print(common_mistakes)
# for element in common_mistakes:
#      print(str(element))
# #print('the most common visual level - ie encompasing more than one character mistakes'\
# #'found in the text')
# #for element in common_visual_mistakes:
#      #print(str(element))

with open(path+file_write, 'w',encoding='utf-8') as write_file:
     # write_file.write('the most common mistakes on the one character level')
     # write_file.write(str(common_mistakes))
     # write_file.write('\n')
     # write_file.write('the most common visual mistakes')
     # #for element in common_visual_mistakes:
     # write_file.write(str(common_visual_mistakes))
     # write_file.write('\n')
     # write_file.write('the mistakes regarding the  added spaces')
     # write_file.write(str(space_added_mistakes))
     # write_file.write('\n')
     # write_file.write('the mistakes regarding the  subtracted spaces')
     # write_file.write(str(space_subtracted_mistakes))
     # write_file.write('\n')
     # write_file.write('mistakes normalized by the letter i.e mistake/whole_gt_letter_occurrences')
     # write_file.write(str(mistakes_normalized_letter_common))
     # write_file.close()

     write_file.write(' the most common substitution mistakes with respect to the occurrence of ground truth letter' \
     'example of ratio: (mistakenly b instead of h)/occurences of h in ground truth ')
     write_file.write(str(mistakes_normalized_letter))
     write_file.write('\n')
     write_file.write(' the most common substitution mistakes on the visual level with respect to the occurrence ' \
     'of the ground truth letter/bigram example of ratio: (mistakenly vv instead of w)/occurences of w')
     write_file.write(str(mistakes_normalized_visual))
     write_file.write('\n')
     write_file.write(' the number of mistakenly added space with respect to all spaces in the ground truth')
     write_file.write(str(mistakes_normalized_space_added))
     write_file.write('\n')
     write_file.write(' the number of mistakenly subtracted spaces with respec to all spaces in the ground truth')
     write_file.write(str(mistakes_normalized_space_subtracted))
     write_file.write('\n')
     write_file.close()

