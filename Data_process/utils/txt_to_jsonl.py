import json

def convert_to_jsonl(input_file, output_file):
    with open(input_file, 'r',encoding='latin1') as f:
        lines = f.readlines()

    jsonl_data = []
    label_id = 0
    for line in lines:
        line = line.strip()
        label,idx, code = line.split('<CODESPLIT>')
        code = code.strip()
        json_obj = {
            "func": code,
            "label": int(label),
            "id": label_id
        }
        jsonl_data.append(json_obj)
        label_id += 1

    with open(output_file, 'w') as f:
        for json_obj in jsonl_data:
            json_line = json.dumps(json_obj)
            f.write(json_line + '\n')

if __name__ == "__main__":

    convert_to_jsonl("other_dataset/devign/FFMPeg+Qemu/train.txt", "other_dataset/devign/data_split_jsonl/train.jsonl")
