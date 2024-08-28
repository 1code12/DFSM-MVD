import random
import numpy as np
import torch
import logging
import torch.backends.cudnn as cudnn
import os
def set_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed) 

    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
 
    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic=True

    torch.cuda.manual_seed_all(args.seed)

logger = logging.getLogger(__name__)

from io import StringIO
import tokenize
import re
def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        prev_tokstr = ""
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            prev_tokstr = token_string
            last_col = end_col
            last_lineno = end_line
        temp = []
        temp_dict = {}
        lineno = 1
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
                temp_dict[lineno] = x
                lineno += 1
        return '\n'.join(temp), temp_dict
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        # source = source.replace(";", ";\n")
        # source = source.replace("{", "\n{")
        temp = []
        temp_dict = {}
        lineno = 1
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
                temp_dict[lineno] = x
                lineno += 1
        return '\n'.join(temp)