import torch

from models.Reveal_BG.AMPLE.graph_transformer_layers import GraphTransformerLayer
from models.FuseModel.mlp_readout import MLPReadout

import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class Linear(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.dropout(self.relu(self.linear(x)))


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, hidden_size=128, d_ff=512, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FFN(d_model=self.hidden_size, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(num_layers * 2)])

    def forward(self, x, y):
        for i in range(self.num_layers):
            attn = self.cross_attn_layers[i]
            ffn = self.ffn_layers[i]
            norm1 = self.layer_norms[i * 2]  
            norm2 = self.layer_norms[i * 2 + 1]  

            attn_output, _ = attn(x, y, y)
          
            x = norm1(x + attn_output)
         
            ffn_output = ffn(x)
          
            x = norm2(x + ffn_output)

        return x


class MultiHeadCrossAttentionAdaptive(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=2, hidden_size=128, d_ff=256, dropout=0.1):
        super(MultiHeadCrossAttentionAdaptive, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.hidden_size , num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            FFN(d_model=self.hidden_size , d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_size ) for _ in range(num_layers * 2)])

    def forward(self, x, y):
        for i in range(self.num_layers):
            attn = self.cross_attn_layers[i]
            ffn = self.ffn_layers[i]
            norm1 = self.layer_norms[i * 2]  
            norm2 = self.layer_norms[i * 2 + 1]  

            attn_output, _ = attn(x, y, y)
         
            x = norm1(x + attn_output)
      
            ffn_output = ffn(x)
         
            x = norm2(x + ffn_output)

        return x


class FusionModel(nn.Module):
    def __init__(self, graph_model, text_model, hidden_size=128, num_heads=4, num_layers=2):
        super(FusionModel, self).__init__()
        self.graph_encoder = graph_model
        self.text_encoder = text_model
        self.hidden_size = hidden_size
        self.activation = torch.nn.ReLU()
        self.MPL_layer_softmax = MLPReadout(hidden_size, 2)  # shape (batch,64)
        self.MPL_layer_sigmoid = MLPReadout(hidden_size // 2, 1)
        self.cross_attention_text_to_graph = MultiHeadCrossAttention(hidden_size)
        self.cross_attention_graph_to_text = MultiHeadCrossAttention(hidden_size)
        self.adaptive_fusion = AdaptiveFusion(self.hidden_size)
    
        self.dropout = nn.Dropout(0.5)
        self.linear_layer = Linear(d_model=self.hidden_size * 4, d_ff=self.hidden_size, dropout=0.1)

        self.project_t = nn.Sequential(
       
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
            self.dropout,
            self.activation,
            nn.LayerNorm(self.hidden_size)
        )

        self.project_g = nn.Sequential(
            nn.Linear(in_features=100, out_features=self.hidden_size),
            self.dropout,
            self.activation,
            nn.LayerNorm(self.hidden_size)
        )


        self.private_t = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.dropout,
            self.activation,
            nn.LayerNorm(self.hidden_size)
        )

        self.private_g = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.dropout,
            self.activation,
            nn.LayerNorm(self.hidden_size)
        )

  
        self.shared = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.dropout,
            self.activation,
            nn.LayerNorm(self.hidden_size)
        )


        self.recon_t = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.dropout,
            self.activation,
            nn.LayerNorm(self.hidden_size)
        )
        self.recon_g = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            self.dropout,
            self.activation,
            nn.LayerNorm(self.hidden_size)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2)  # 128//2=64
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def shared_private_CodeSpace(self, graph_features, text_features):
   
        self.orig_t = self.project_t(text_features)
        self.orig_g = self.project_g(graph_features)

        self.private_text = self.private_t(self.orig_t)  
        self.private_graph = self.private_g(self.orig_g)

        self.shared_t = self.shared(self.orig_t)
        self.shared_g = self.shared(self.orig_g)

    def reconstruct(self, ):
        self.code_t = (self.private_text + self.shared_t)
        self.code_g = (self.private_graph + self.shared_g)

        self.code_t_recon = self.recon_t(self.code_t)
        self.code_g_recon = self.recon_g(self.code_g)

    def forward(self, batch, inputs_ids, ast_inputs_ids, activation_type=None):
   
        graph_features = self.graph_encoder(batch, cuda=True)
   
      
        text_features = self.text_encoder(input_ids=inputs_ids, ast_ids=ast_inputs_ids)
      
        self.shared_private_CodeSpace(graph_features, text_features)

     
        self.reconstruct()

     

        cross_text_to_graph = self.cross_attention_text_to_graph(self.orig_t, self.orig_g)
        cross_graph_to_text = self.cross_attention_graph_to_text(self.orig_g, self.orig_t)
        h = torch.stack((cross_text_to_graph, cross_graph_to_text, self.shared_t, self.shared_g), dim=0)
        h = self.transformer_encoder(h)
        combined_shared_code = torch.cat((h[0], h[1], h[2], h[3]), dim=-1)
    
        combined_shared_code = self.linear_layer(combined_shared_code)
  
     
        fused_vector= self.adaptive_fusion(combined_shared_code, self.private_text, self.private_graph)  # 128*2

    



        logit = self.MPL_layer_softmax(fused_vector)
 
        prob = torch.softmax(logit, dim=-1)

        return logit,prob


class AdaptiveFusion(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(AdaptiveFusion, self).__init__()

        self.cross_attention_text_to_combine = MultiHeadCrossAttentionAdaptive(hidden_size)
        self.cross_attention_combine_to_text = MultiHeadCrossAttentionAdaptive(hidden_size)

        self.cross_attention_graph_to_combine = MultiHeadCrossAttentionAdaptive(hidden_size)
        self.cross_attention_combine_to_graph = MultiHeadCrossAttentionAdaptive(hidden_size)
    
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() #0-1
        )

    def forward(self, combined_shared_code, private_text, private_graph):
        cross_combine_to_text = self.cross_attention_combine_to_text(combined_shared_code, private_text)
        #print(combined_shared_code.shape)
        cross_text_to_combine = self.cross_attention_text_to_combine(private_text, combined_shared_code)
        gate_text = torch.cat([cross_combine_to_text, cross_text_to_combine], dim=-1)
        gate_value_text = self.gate(gate_text)

        cross_combine_to_graph = self.cross_attention_combine_to_graph(combined_shared_code, private_graph)
        cross_graph_to_combine = self.cross_attention_graph_to_combine(private_graph, combined_shared_code)
        gate_graph = torch.cat([cross_combine_to_graph, cross_graph_to_combine], dim=-1)
        gate_value_graph = self.gate(gate_graph)


 
        fused_vector = combined_shared_code + gate_value_text * private_text + gate_value_graph * private_graph
        return fused_vector

