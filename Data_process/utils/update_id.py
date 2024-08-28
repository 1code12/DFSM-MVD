import json

input_file = 
output_file = 


with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()


idx = 0


output_lines = []
for line in lines:
    data = json.loads(line)

    if 'idx' in data:
        del data['idx']
    data['id'] = idx
    idx += 1
    output_lines.append(json.dumps(data))


with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))