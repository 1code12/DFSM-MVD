
from torch.utils.data import Dataset, SequentialSampler, RandomSampler,TensorDataset
from utils import  logger
import  json
import numpy as np
import torch
#from linevul.utils import remove_comments_and_docstrings

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 ast_ids,

                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.ast_ids = ast_ids
        self.label = label

def convert_examples_to_features(js, tokenizer, args):
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    #astsequence
    astseq = ' '.join(js['ast'].split())
    astseq_tokens = tokenizer.tokenize(astseq)[:args.block_size - 2]
    astseq_tokens = [tokenizer.cls_token] + astseq_tokens + [tokenizer.sep_token]
    astseq_ids = tokenizer.convert_tokens_to_ids(astseq_tokens)
    ast_padding_length = args.block_size - len(astseq_ids)
    astseq_ids += [tokenizer.pad_token_id] * ast_padding_length

    return InputFeatures(source_tokens, source_ids, astseq_ids,js['target'])
# TextDataset
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)

     
        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

        if 'train' in file_path:
            logger.info("*** Total Sample ***")
            logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Sample ***")
                logger.info("Total sample".format(idx))
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label),torch.tensor(self.examples[i].ast_ids)