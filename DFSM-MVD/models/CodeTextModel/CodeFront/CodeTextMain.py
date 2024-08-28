import argparse
from utils import  set_seed,logger
import torch
import random
import numpy as np
import logging
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,RobertaModel)

from CodeTextModel import Model

from singleDataset import  TextDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from tqdm import tqdm
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
import os
import json










def warn(*args, **kwargs):
    pass



import warnings

warnings.warn = warn



torch.cuda.empty_cache()

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)  
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)  # 创建数据加载器


    args.max_steps = args.epochs * len(train_dataloader)  
    args.save_steps = len(train_dataloader)  
    args.warmup_steps = args.max_steps // 5 
    model.to(args.device)  

  
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)


    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (inputs_ids, labels,ast_ids) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(input_ids=inputs_ids,ast_ids=ast_ids, labels=labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, labels,ast_ids) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids,ast_ids=ast_ids,  labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in test_dataloader:
        (inputs_ids, labels,ast_ids) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids,ast_ids=ast_ids,  labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    torch.cuda.empty_cache()
    return result

def main():
    parser = argparse.ArgumentParser()
    ## parameters

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
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
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
    args = parser.parse_args()
    #args = parser.parse_args(args=[])#colab use
    args.evaluate_during_training=True

    args.output_dir='models/CodeTextModel/CodeFront/saved_models/checkpoint-best-f1'

    args.tokenizer_name='codebert/model'
    args.model_name_or_path='codebert/model'
    args.do_train =False

    args.do_test =True
    args.train_data_file ='Dataset/Devign/train.jsonl'
    args.eval_data_file ='Dataset/Devign/valid.jsonl'
    args.test_data_file ='Dataset/Devign/test.jsonl'

    args.epochs= 10
    args.block_size =512
    args.train_batch_size=16
    args.eval_batch_size =16
    args.learning_rate =2e-5
    args.max_grad_norm =1.0
    



    # Setup CUDA, GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)


    set_seed(args)


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
    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
        #model = RobertaModel.from_pretrained(args.model_name_or_path, config=config, ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer, args)
    # model = CodeTextEncoder(model, config, tokenizer, args)


    logger.info("Training/evaluation parameters %s", args)
    torch.cuda.empty_cache()

    if args.do_train:
        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
        train(args, train_dataset, model, tokenizer, eval_dataset)
  
    results = {}
    torch.cuda.empty_cache()
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
     
        #model.load_state_dict(torch.load(output_dir),strict=False)
        model.load_state_dict(torch.load(output_dir))

        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, args.test_data_file)
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    return results

if __name__ == "__main__":
    main()