# First

import json
import os
import  re
from utils.remove_comments import remove_comments
jsonl_file_path = ""
# def remove_nested_newlines(text):
#     stack = []
#     result = ''
#
#     for char in text:
#         if char == '(':
#             stack.append(len(result))
#         elif char == ')':
#             start = stack.pop()
#             nested_text = result[start:]
#             nested_cleaned = nested_text.replace('\n\n', ' ')
#             result = result[:start] + nested_cleaned
#         else:
#             result += char
#
#     return result
def remove_nested_newlines(text,idx):
    stack = []
    result = ''

    for char in text:
        if char == '(':
            stack.append(len(result))
            result += char
        elif char == ')':
            if stack:
                start = stack.pop()
                nested_text = result[start:]
                nested_cleaned = nested_text.replace('\n\n', ' ')
                result = result[:start] + nested_cleaned + char
            else:
                print(idx)
                return text

        else:
            result += char

    return result
with open(jsonl_file_path) as f:
    for idx, line in enumerate(f):

        entry = json.loads(line)
        target = entry["label"]
        cpp_code = entry["func"]

        func_code_cleaned = remove_nested_newlines(cpp_code,idx)
        func_code_cleaned = remove_comments(func_code_cleaned, 'c')
    
        cleaned_text = re.sub(r'(\w+)\n+\s*(\()', r'\1 \2', func_code_cleaned)
      

        cleaned_text = re.sub(r'([+=/*>=])\s*\n+\s*', r'\1 ', cleaned_text)
        file_name = f"{idx}.c"
        output_directory = "other_dataset/devign/raw_code/train"
        # #
        combined_code = f"// Target: {target}\n//start\n{cleaned_text}\n//end"
        #combined_code = f"// Target: {target}\n//start\n{cpp_code}\n//end"

        with open(os.path.join(output_directory, file_name), "w") as f:
            f.write(combined_code)








print("C++ code extracted from JSONL file and saved.")


