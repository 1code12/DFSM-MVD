import numpy as np

# from data_loader import n_identifier, g_identifier, l_identifier
import inspect
from datetime import datetime
from torch.autograd import Function
import random
import numpy as np
import torch
import logging
from io import StringIO
import tokenize
import re
import dgl
import os
import torch.backends.cudnn as cudnn
# def load_default_identifiers(n, g, l):
#     if n is None:
#         n = n_identifier
#     if g is None:
#         g = g_identifier
#     if l is None:
#         l = l_identifier
#     return n, g, l


def initialize_batch(entries, batch_size, shuffle=False):
    total = len(entries)
    print(str(total)+'k'*35)
    indices = np.arange(0, total , 1)
    if shuffle:
        np.random.shuffle(indices)
    batch_indices = []
    start = 0
    end = len(indices)
    curr = start
    while curr < end:
        c_end = curr + batch_size
        if c_end > end:
            c_end = end
        batch_indices.append(indices[curr:c_end])
        curr = c_end
    return batch_indices[::-1]


def tally_param(model):
    total = 0
    for param in model.parameters():
        total += param.data.nelement()
    return total


def debug(*msg, sep='\t'):
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    print('[' + str(time) + '] File \"' + file_name + '\", line ' + str(ln) + '  ', end='\t')
    for m in msg:
        print(m, end=sep)
    print('')

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="w", encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        #logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


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


def set_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed) 
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic=True
    torch.cuda.manual_seed_all(args.seed)
    dgl.seed(args.seed)
    dgl.random.seed(args.seed)



