import os
import json

def read_c_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_index = -1
        end_index = -1
        label = None
        for i, line in enumerate(lines):
            if line.strip() == '//start':
                start_index = i + 1
            elif line.strip() == '//end':
                end_index = i
            elif line.strip().startswith('// Target:'):
                label = int(line.strip().split(':')[-1])
        if label is None:
            raise ValueError(f"No label found in {file_path}")
        func_code = ''.join(lines[start_index:end_index])
        return label, func_code.strip()

def create_jsonl(input_dir, output_file):
    id_counter = 0
    with open(output_file, 'w') as jsonl_file:
        for i in range(len(os.listdir(input_dir))):
            c_file_path = os.path.join(input_dir, f"{i}.c")
            if os.path.isfile(c_file_path):
                label, func_code = read_c_file(c_file_path)
                json_line = json.dumps({"id": id_counter, "label": label, "func": func_code})
                jsonl_file.write(json_line + '\n')
                id_counter += 1


input_dir = '/media/thinkstation02/data/wwkfile/wwk_pycharm_project/CPG_process/other_dataset/devign/raw_code/train'  
output_file = 'devign_train_xiugai.jsonl'  
create_jsonl(input_dir, output_file)
