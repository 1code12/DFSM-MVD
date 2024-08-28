import argparse

from torch.nn import CrossEntropyLoss


import torch
import numpy as np
import logging
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,RobertaModel)

from models.CodeGraphModel.CPG_Graph.CPG_Graph_model import CodeGraphEncoder

from models.FuseModel.FuseMode import FusionModel


from models.CodeTextModel.CodeFront.CodeTextModel import CodeTextEncoder

from dataset.code_dataset import CodeDataset
from tokenizers import Tokenizer
from train import Solver

import os
from train import evaluate,test
from utils import set_logger,set_seed
import random
import dgl
import torch.optim
import torch.backends.cudnn as cudnn





def warn(*args, **kwargs):
    pass



import warnings

warnings.warn = warn



torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file .")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--model_type", default="bert", type=str,
    #                     help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each  step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    parser.add_argument('--cnn_size', type=int, default=1, help="For cnn size.")
    parser.add_argument('--filter_size', type=int, default=2, help="For cnn filter size.")

    parser.add_argument('--d_size', type=int, default=128, help="For cnn filter size.")
    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")

    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.', default='FFmpeg')

    parser.add_argument('--log_dir', default='late_fuse.log', type=str)

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=2)
    parser.add_argument('--diff_sim_weight', type=float, default=0.7)
    #parser.add_argument('--sim_weight', type=float, default=0.6)
    parser.add_argument('--contrastive_weight', type=float, default=0.7)
    parser.add_argument('--recon_weight', type=float, default=0.7)

    args = parser.parse_args()

    args.evaluate_during_training=True

    args.output_dir='save_model/FuseModelSave'
    args.tokenizer_name='codebert/model'
    args.model_name_or_path='codebert/model'
    args.do_train =True
    args.do_test =True


    args.train_data_file ='Dataset/Devign/train.jsonl'
    args.eval_data_file ='Dataset/Devign/valid.jsonl'
    args.test_data_file ='Dataset/Devign/test.jsonl'
    
    args.epochs= 20
    args.block_size =512
    args.train_batch_size=64
    args.eval_batch_size =64
    args.learning_rate =2e-5
 
    args.max_grad_norm =1.0
  


    set_seed(args)
    # Setup CUDA, GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device
    # # Setup logging
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    # logger.warning("device: %s, n_gpu: %s",device, args.n_ gpu,)


    model_dir = os.path.join('log', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(model_dir, args.log_dir)
    set_logger(log_dir)




  
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    #args.use_non_pretrained_tokenizer=True
    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('../word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        print('using BPE tokenizer!')
        tokenizer = RobertaTokenizer(vocab_file="models/CodeTextModel/CodeFront/bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="models/CodeTextModel/CodeFront/bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        print('using moren tokenizer!')
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)

    codebert = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)

    text_model = CodeTextEncoder(codebert, config, tokenizer, args)
    #text_output_dir = 'save_model/CodeTextSave/CodeXGlue/model.bin'

    text_model.load_state_dict(torch.load(text_output_dir))

    for param in text_model.parameters():
        param.requires_grad = False
    graph_model = CodeGraphEncoder(input_dim=args.feature_size, output_dim=100,
                                   num_steps=args.num_steps, max_edge_types=5)
    model = FusionModel(graph_model, text_model)
    model.cuda()

    loss_fct = CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1])).float(), reduction='sum')

    loss_fct.cuda()

    logging.info("Training/evaluation parameters %s", args)
    torch.cuda.empty_cache()
    # Training
    if args.do_train:
        train_dataset = CodeDataset(src_file=args.train_data_file, n_ident='node_features', g_ident='graph',
                                    l_ident='targets')
        eval_dataset = CodeDataset(src_file=args.eval_data_file, n_ident='node_features', g_ident='graph',
                                   l_ident='targets')
        solver = Solver
        solver=solver(args, train_dataset, model, eval_dataset,loss_fct,device)
        solver.train()
    # Evaluation
    results = {}
    torch.cuda.empty_cache()
    if args.do_test:
        set_seed(args)
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))

        model.load_state_dict(torch.load(output_dir))
        print(output_dir)
        model.to(args.device)

        test_dataset = CodeDataset(src_file=args.test_data_file, n_ident='node_features', g_ident='graph', l_ident='targets')

        test(args, model, loss_fct,device, test_dataset, best_threshold=0.5)
    return results

if __name__ == "__main__":
    main()
