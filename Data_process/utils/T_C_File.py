 #First

import json
import os

jsonl_file_path = ""

with open(jsonl_file_path) as f:
    for idx, line in enumerate(f):
        entry = json.loads(line)
        target = entry["label"]
        cpp_code = entry["func"]

        file_name = f"{idx}.c"
        output_directory = ""

        combined_code = f"// Target: {target}\n//start\n{cpp_code}\n//end"

        with open(os.path.join(output_directory, file_name), "w") as f:
            f.write(combined_code)

print("C++ code extracted from JSONL file and saved.")


