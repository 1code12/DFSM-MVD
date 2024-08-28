import json
empty_paths_txt='path_extract/empty_paths.txt'

with open(empty_paths_txt, "r") as f:
    empty_paths = set(int(line.strip()) for line in f)

old_dataset='dataset/data_split_json/devign_reveal_train.json'
filter_dataset='dataset/data_split_json/train_test.json'
with open(old_dataset, "r") as f_in:
    with open(filter_dataset, "w") as f_out:
        for line in f_in:
            obj = json.loads(line.strip())
            if int(obj["id"])  in empty_paths:
                f_out.write(json.dumps(obj) + "\n")
