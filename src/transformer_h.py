import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 

# class Embedding(nn.Module):
#     def __init__(self, d_model, vocab):
#         super(Embedding, self).__init__()
#         self.embed = nn.Embedding()

class Positional_Encoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(Positional_Encoding, self).__init__()

        self.d_model = d_model
        position_encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        position_encoding[:, 0::2] = torch.sin(pos * div_term)
        position_encoding[:, 1::2] = torch.cos(pos * div_term)

        position_encoding = position_encoding.unsqueeze(0)

        self.register_buffer('pe', position_encoding)

    def forward(self, x):
        # (batch_size, seq_len, hidden_dim)
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return x

class ScaledDotproductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotproductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        '''
        q: (batch_size, n_head, len_q, d_k)
        k: (batch_size, n_head, len_q, d_k)
        v: (batch_size, n_head, len_q, d_v)
        '''
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(2, 3)) / (d_k ** 0.5)
        
        if mask is not None:
            # float('-inf')): nan
            scores = scores.masked_fill(mask==0, -1e9)
        
        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_Heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_Heads = n_Heads
        self.d_k = d_model // n_Heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotproductAttention()

    def forward(self, q, k ,v, mask=None):
        # 定义q,k,v而不是x: 兼容self_attention和cross_attention

        # (batch_size, seq_len, hidden_dim)
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        batch_size = q.size(0)

        # (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_Heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_Heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_Heads, self.d_k).transpose(1, 2)

        context, attn = self.attn(Q, K, V, mask=mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)

        return output, attn

class Position_wise_Feed_Forward_Net(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Position_wise_Feed_Forward_Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.fc(x)
    
class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Add_Norm, self).__init__()
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.add_norm1 = Add_Norm(d_model, dropout)

        self.ffn = Position_wise_Feed_Forward_Net(d_model, d_ff, dropout)
        self.add_norm2 = Add_Norm(d_model, dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        output_1 = self.add_norm1(x, attn_output)

        ffn_output = self.ffn(output_1)
        output_2 = self.add_norm2(output_1, ffn_output)

        return output_2
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1):
        super(Encoder, self).__init__()

        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encode = Positional_Encoding(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        embed_x = self.src_emb(x)
        x_layer = self.pos_encode(embed_x)
        for layer in self.layers:
            x_layer = layer(x_layer, mask)
        
        return self.last_norm(x_layer)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.add_norm1 = Add_Norm(d_model, dropout)

        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.add_norm2 = Add_Norm(d_model, dropout)

        self.ffn = Position_wise_Feed_Forward_Net(d_model, d_ff, dropout)
        self.add_norm3 = Add_Norm(d_model, dropout)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        output1 = self.add_norm1(x, self_attn_out)

        cross_attn_out, _ = self.cross_attn(output1, encoder_output, encoder_output, memory_mask)
        output2 = self.add_norm2(output1, cross_attn_out)

        ffn_out = self.ffn(output2)
        output3 = self.add_norm3(output2, ffn_out)

        return output3

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = Positional_Encoding(d_model)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        embed_tgt = self.embed(tgt)
        tgt_layer = self.pos_embed(embed_tgt)

        for layer in self.layers:
            tgt_layer = layer(tgt_layer, encoder_output, tgt_mask, memory_mask)
        
        return self.last_norm(tgt_layer)

class Encoder_Only_Transformer(nn.Module):
    def __init__(self, src_vocab, num_classes, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1):
        super(Encoder_Only_Transformer, self).__init__()

        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, num_layers, dropout)

        self.fc = nn.Linear(d_model, num_classes)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):
        output = self.encoder(src, src_mask)
        # 取 [CLS] token
        out = self.fc(output[:, 0, :])

        return out


class Encoder_Decoder_Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1):
        super(Encoder_Decoder_Transformer, self).__init__()

        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, d_ff, num_layers, dropout)

        self.output_linear = nn.Linear(d_model, tgt_vocab)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

        logits = self.output_linear(decoder_output)

        return logits



    