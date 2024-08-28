import os
import json
from tree_sitter import Language, Parser


LANGUAGE_LIBRARY_PATH = 'build/my-languages.so'
CPP_LANGUAGE = Language(LANGUAGE_LIBRARY_PATH, 'cpp')

cpp_parser = Parser()
cpp_parser.set_language(CPP_LANGUAGE)

INPUT_FILE =
OUTPUT_FILE =
i=0

with open(INPUT_FILE, 'r', encoding='utf-8') as input_file, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
    for line in input_file:
    
        json_obj = json.loads(line)

   
        code = json_obj.get('func', '')

    
        tree = cpp_parser.parse(bytes(code, "utf8"))
        root = tree.root_node 
        sexp = root.sexp()
        print(i,sexp)
        i=i+1
        json_obj['ast'] = sexp
        output_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
print("total",i)
