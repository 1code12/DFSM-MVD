
import json

file_path = '/media/thinkstation02/data/wwkfile/wwk_pycharm_project/CPG_process/Final_dataset/Devign/data_split_jsonl/devign_train_ast.jsonl'
output_file = '/media/thinkstation02/data/wwkfile/wwk_pycharm_project/CPG_process/Final_dataset/Devign/data_split_jsonl/devign_train_ast1.jsonl'

with open(file_path) as read_f, open(output_file, 'w') as write_f:
    i = 0
    for line in read_f:
        json_obj = json.loads(line)

        filename = 'train/%d.c' % i
        i += 1

        json_obj['file_path'] = filename

        json_str = json.dumps(json_obj)
        write_f.write(json_str + '\n')