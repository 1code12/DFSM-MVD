import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification
import torch.nn.functional as F

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config,args):
        super().__init__()
        self.dense = nn.Linear(args.d_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
#        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x




class Model(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config,args)
        self.args = args
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.args.d_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=config.hidden_size, out_features=self.args.d_size),
            nn.LayerNorm(self.args.d_size),
            self.activation,
            self.dropout

        )
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None,ast_ids=None):


        if input_ids is not None:
            sourcecode_outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            sourcecode_embedding = sourcecode_outputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            print("input_ids is None")
        if ast_ids is not None:
            ast_outputs = self.encoder.roberta(ast_ids, attention_mask=ast_ids.ne(1), output_attentions=output_attentions)[0]
            ast_embedding = ast_outputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            print("ast_ids is None")

        sourcecode_embedding=self.fc(sourcecode_embedding)


        ast_embedding=self.fc(ast_embedding)

        #
        x = torch.stack((sourcecode_embedding, ast_embedding), dim=0)
        h = self.transformer_encoder(x)
        h = torch.cat((h[0], h[1]), dim=1)

        logits=self.classifier(h)
        # print(logits.shape)
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            # print(labels)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob



class CodeTextEncoder(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeTextEncoder, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config,args)
        self.args = args
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.args.d_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
   
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.args.d_size, nhead=2)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(in_features=config.hidden_size, out_features=self.args.d_size),
            nn.LayerNorm(self.args.d_size),
            self.activation,
            self.dropout

        )
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None,ast_ids=None):


        if input_ids is not None:
            sourcecode_outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            sourcecode_embedding = sourcecode_outputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            print("input_ids is None")
        if ast_ids is not None:
            ast_outputs = self.encoder.roberta(ast_ids, attention_mask=ast_ids.ne(1), output_attentions=output_attentions)[0]
            ast_embedding = ast_outputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            print("ast_ids is None")

        sourcecode_embedding=self.fc(sourcecode_embedding)


        ast_embedding=self.fc(ast_embedding)

        #
        x = torch.stack((sourcecode_embedding, ast_embedding), dim=0)
        h = self.transformer_encoder(x)
        h = torch.cat((h[0], h[1]), dim=1)
        return h
        #
        # logits=self.classifier(h)
        # # print(logits.shape)
        # prob = torch.softmax(logits, dim=-1)
        # if labels is not None:
        #     # print(labels)
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits, labels)
        #     return loss, prob
        # else:
        #     return prob
