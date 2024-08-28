# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from tqdm import tqdm
from utils import set_seed
import numpy as np
import os
from dataset.code_dataset import collate_fn
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,RobertaModel)
from all_loss_functions import  DiffLoss,MSE,CMD,SIMSE,CodeTGContrastiveLoss
import logging
#import matplotlib.pyplot as plt

class Solver(object):
    def __init__(self, args, train_dataset, model,eval_dataset,loss_fct,device):
        self.args=args
        self.train_dataset=train_dataset
        self.model=model
        self.eval_dataset=eval_dataset
        self.loss_fct=loss_fct
        self.device=device

    def train(self):


        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.loss_contrastive = CodeTGContrastiveLoss()
        """ Train the model """
        # build dataloader
        train_sampler = RandomSampler(self.train_dataset) 
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size,
                                      collate_fn=collate_fn) 

     
        self.args.max_steps = self.args.epochs * len(train_dataloader) 
        self.args.save_steps = len(train_dataloader)  
        self.args.warmup_steps = self.args.max_steps // 5  
        self.model.to(self.args.device) 


        optimizer_grouped_parameters = [
            {'params': self.model.parameters(),
             'weight_decay': 1e-6}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=self.args.max_steps)

        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(self.model)

        # Train!
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(self.train_dataset))
        logging.info("  Num Epochs = %d", self.args.epochs)
        logging.info("  Instantaneous batch size per GPU = %d", self.args.train_batch_size // max(self.args.n_gpu, 1))
        logging.info("  Total train batch size = %d", self.args.train_batch_size * self.args.gradient_accumulation_steps)
        logging.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", self.args.max_steps)

        global_step = 0
        tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
        best_f1 = 0

        self.model.zero_grad()
        # Lists to store the losses for each epoch
        overall_losses = []
        intra_modal_losses = []
        inter_modal_losses = []
        for idx in range(self.args.epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_num = 0
            train_loss = 0
            epoch_diff_loss = 0
            epoch_recon_loss = 0
            epoch_cmd_loss = 0
            epoch_contrastive_loss = 0
            for step, batch in enumerate(bar):
                self.model.train()
                (graph, label, text_inputs_ids, ast_inputs_ids) = [x for x in batch]
                text_inputs_ids = torch.tensor(text_inputs_ids).to(self.device)
                ast_inputs_ids = torch.tensor(ast_inputs_ids).to(self.device)

                targets = label.to(self.device)

                logit,logits = self.model(graph, text_inputs_ids, ast_inputs_ids)
                cls_loss = self.loss_fct(logit, targets.long())
                diff_loss = self.get_diff_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                contrastive_loss = self.get_contrastive_loss()
                intra_modal_loss = self.args.diff_sim_weight * (diff_loss + cmd_loss + recon_loss)
                inter_modal_loss = self.args.contrastive_weight * contrastive_loss
                loss = cls_loss + intra_modal_loss + inter_modal_loss
              
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                tr_loss += loss.item()
                epoch_diff_loss += diff_loss.item()*self.args.diff_sim_weight
                epoch_recon_loss += recon_loss.item()*self.args.diff_sim_weight
                epoch_cmd_loss += cmd_loss.item()*self.args.diff_sim_weight
                epoch_contrastive_loss += contrastive_loss.item()*self.args.contrastive_weight
                tr_num += 1
                train_loss += loss.item()
                if avg_loss == 0:
                    avg_loss = tr_loss

                avg_loss = round(train_loss / tr_num, 5)
                bar.set_description("epoch {} loss {}".format(idx, avg_loss))

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    output_flag = True
                    avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                    if global_step % self.args.save_steps == 0:
                        results = evaluate(self.args, self.model, self.loss_fct, self.device, self.eval_dataset, eval_when_training=True)

                        # Save model checkpoint
                        if results['eval_f1'] > best_f1:                 
                            best_f1 = results['eval_f1']
                        
                            logging.info("  " + "*" * 20)
                            logging.info("  Best f1:%s", round(best_f1, 4))
                            logging.info("  " + "*" * 20)

                            checkpoint_prefix = 'checkpoint-best-f1'
                            output_dir = os.path.join(self.args.output_dir, '{}'.format(checkpoint_prefix))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                            output_dir = os.path.join(output_dir, '{}'.format(self.args.model_name))
                            torch.save(model_to_save.state_dict(), output_dir)
                            logging.info("Saving model checkpoint to %s", output_dir)
            # Calculate average losses for this epoch
            avg_epoch_loss = train_loss / tr_num  
            avg_diff_loss = epoch_diff_loss / tr_num
            avg_recon_loss = epoch_recon_loss / tr_num
            avg_cmd_loss = epoch_cmd_loss / tr_num
            avg_contrastive_loss = epoch_contrastive_loss / tr_num

       
            #overall_losses.append(avg_epoch_loss)
            #intra_modal_losses.append(avg_diff_loss + avg_cmd_loss + avg_recon_loss)
            #inter_modal_losses.append(avg_contrastive_loss)


        # Save the losses to text files
        #np.savetxt('overall_losses.txt', overall_losses)
        #np.savetxt('intra_modal_losses.txt', intra_modal_losses)
        #np.savetxt('inter_modal_losses.txt', inter_modal_losses)
        # # Plot the losses
        # epochs = range(1, self.args.epochs + 1)
        # plt.figure()
        # plt.plot(epochs, overall_losses, label='Overall Loss')
        # plt.plot(epochs, intra_modal_losses, label='Intra-modal Loss')
        # plt.plot(epochs, inter_modal_losses, label='Inter-modal Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training Loss Over Epochs')
        # plt.legend()
        # plt.show()

    def get_cmd_loss(self):

        loss = self.loss_cmd(self.model.shared_t, self.model.shared_g, 5)
        return loss

    def get_diff_loss(self):

        shared_t = self.model.shared_t
        shared_g = self.model.shared_g
     
        private_t = self.model.private_text
        private_g = self.model.private_graph
      

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_g, shared_g)

        # Across privates
        loss += self.loss_diff(private_t, private_g)

        loss = loss / 3.0

        return loss

    def get_recon_loss(self):
        loss = self.loss_recon(self.model.code_t_recon, self.model.orig_t)
        loss += self.loss_recon(self.model.code_g_recon, self.model.orig_g)
        loss = loss / 2.0
        return loss

    def get_contrastive_loss(self):
        loss = self.loss_contrastive(self.model.orig_t, self.model.orig_g)
        return loss
def evaluate(args, model, loss_fct,device, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,collate_fn=collate_fn, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        #(inputs_ids, labels,ast_ids) = [x.to(args.device) for x in batch]
        (graph, label, text_inputs_ids, ast_inputs_ids) = [x for x in batch]
        text_inputs_ids = torch.tensor(text_inputs_ids).to(device)
        ast_inputs_ids = torch.tensor(ast_inputs_ids).to(device)

        targets = label.to(device)
        with torch.no_grad():
            #lm_loss, logit = model(input_ids=inputs_ids,ast_ids=ast_ids,  labels=labels)
            prop,logit = model(graph, text_inputs_ids, ast_inputs_ids)
            lm_loss = loss_fct(prop, targets.long())
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(targets.cpu().numpy())
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
    acc = accuracy_score(y_trues, y_preds)
    result = {
         "eval_recall": float(recall),
         "eval_precision": float(precision),
         "eval_f1": float(f1),
         "eval_threshold": best_threshold,
     }
 
    logging.info("***** eval results *****")
  
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model,loss_fct,device, test_dataset, best_threshold=0.5):

    set_seed(args)  
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size,collate_fn=collate_fn, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logging.info("***** Running Test *****")
    logging.info("  Num examples = %d", len(test_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    # gate_values_text_list = []
    # gate_values_graph_list = []

    # feature_list = [] 
    # label_list = [] 
    for batch in test_dataloader:
        (graph, label, text_inputs_ids, ast_inputs_ids) = [x for x in batch]
        text_inputs_ids = torch.tensor(text_inputs_ids).to(device)
        ast_inputs_ids = torch.tensor(ast_inputs_ids).to(device)

        targets = label.to(device)
        with torch.no_grad():
       
            prop,logit = model(graph, text_inputs_ids, ast_inputs_ids)
           
        
            lm_loss = loss_fct(prop, targets.long())
          
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(targets.cpu().numpy())
        nb_eval_steps += 1

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

    logging.info("***** Test results *****")
    for key in sorted(result.keys()):
        logging.info("  %s = %s", key, str(round(result[key], 4)))
    torch.cuda.empty_cache()
    return result
