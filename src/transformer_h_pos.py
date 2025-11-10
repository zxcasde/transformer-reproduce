import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 

class LearnedPositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 创建可学习的位置编码参数
        self.position_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # 生成位置索引
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # 获取位置编码并加到输入上
        pos_encoding = self.position_embedding(positions)
        return x + pos_encoding

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码（原始Transformer使用）"""
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()

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

class RelativePositionalEncoding(nn.Module):
    """相对位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 相对位置编码通常用于注意力计算中
        # 这里简化实现，使用可学习的相对位置偏置
        self.rel_pos_bias = nn.Embedding(2 * max_len - 1, 1)
        
    def forward(self, x):
        # 相对位置编码通常不直接加到输入上，而是在注意力计算中使用
        # 这里返回原始输入，相对位置编码将在注意力模块中使用
        return x
    
    def get_rel_pos_bias(self, seq_len, device):
        """获取相对位置偏置矩阵"""
        # 生成相对位置索引，确保在正确的设备上
        range_vec = torch.arange(seq_len, device=device)
        range_mat = range_vec.unsqueeze(1).expand(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # 将距离映射到索引
        distance_mat = distance_mat + self.max_len - 1
        
        # 确保嵌入层在正确的设备上
        bias = self.rel_pos_bias(distance_mat).squeeze(-1)
        
        return bias.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
class NoPositionalEncoding(nn.Module):
    """无位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(NoPositionalEncoding, self).__init__()
        self.d_model = d_model
        
    def forward(self, x):
        # 直接返回输入，不添加位置信息
        return x

class ScaledDotproductAttention(nn.Module):
    def __init__(self, use_relative_pos=False, max_len=5000):
        super(ScaledDotproductAttention, self).__init__()
        self.use_relative_pos = use_relative_pos
        
    def forward(self, q, k, v, mask=None, rel_pos_bias=None):
        '''
        q: (batch_size, n_head, len_q, d_k)
        k: (batch_size, n_head, len_q, d_k)
        v: (batch_size, n_head, len_q, d_v)
        '''
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(2, 3)) / (d_k ** 0.5)
        
        # 添加相对位置偏置
        if self.use_relative_pos and rel_pos_bias is not None:
            scores = scores + rel_pos_bias
        
        if mask is not None:
            # float('-inf')): nan
            scores = scores.masked_fill(mask==0, -1e9)
        
        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_Heads, use_relative_pos=False, max_len=5000):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_Heads = n_Heads
        self.d_k = d_model // n_Heads
        self.use_relative_pos = use_relative_pos

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotproductAttention(use_relative_pos, max_len)

    def forward(self, q, k ,v, mask=None, rel_pos_bias=None):
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

        context, attn = self.attn(Q, K, V, mask=mask, rel_pos_bias=rel_pos_bias)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)

        return output, attn

class Position_wise_Feed_Forward_Net(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Position_wise_Feed_Forward_Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.fc(x)
    
class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Add_Norm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1, use_relative_pos=False, max_len=5000):
        super(EncoderLayer, self).__init__()
        self.use_relative_pos = use_relative_pos

        self.self_attn = MultiHeadAttention(d_model, n_heads, use_relative_pos, max_len)
        self.add_norm1 = Add_Norm(d_model, dropout)

        self.ffn = Position_wise_Feed_Forward_Net(d_model, d_ff, dropout)
        self.add_norm2 = Add_Norm(d_model, dropout)

    def forward(self, x, mask=None, rel_pos_bias=None):
        attn_output, _ = self.self_attn(x, x, x, mask, rel_pos_bias)
        output_1 = self.add_norm1(x, attn_output)

        ffn_output = self.ffn(output_1)
        output_2 = self.add_norm2(output_1, ffn_output)

        return output_2
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1, 
                 pos_encoding_type='sinusoidal', max_len=5000):
        super(Encoder, self).__init__()

        self.src_emb = nn.Embedding(vocab_size, d_model)
        
        # 根据类型选择位置编码
        self.pos_encoding_type = pos_encoding_type
        if pos_encoding_type == 'sinusoidal':
            self.pos_encode = SinusoidalPositionalEncoding(d_model, max_len)
            self.use_relative_pos = False
        elif pos_encoding_type == 'learned':
            self.pos_encode = LearnedPositionalEncoding(d_model, max_len)
            self.use_relative_pos = False
        elif pos_encoding_type == 'relative':
            self.pos_encode = RelativePositionalEncoding(d_model, max_len)
            self.use_relative_pos = True
        elif pos_encoding_type == 'none':
            self.pos_encode = NoPositionalEncoding(d_model, max_len)
            self.use_relative_pos = False
        else:
            raise ValueError(f"Unsupported position encoding type: {pos_encoding_type}")

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, self.use_relative_pos, max_len) 
            for _ in range(num_layers)
        ])

        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        embed_x = self.src_emb(x)
        
        # 应用位置编码（相对位置编码除外）
        if self.pos_encoding_type != 'relative':
            x_layer = self.pos_encode(embed_x)
        else:
            x_layer = embed_x
        
        # 如果是相对位置编码，计算相对位置偏置
        rel_pos_bias = None
        if self.pos_encoding_type == 'relative':
            seq_len = x.shape[1]
            # 传递设备信息给相对位置编码
            rel_pos_bias = self.pos_encode.get_rel_pos_bias(seq_len, x.device)
        
        for layer in self.layers:
            x_layer = layer(x_layer, mask, rel_pos_bias)
        
        return self.last_norm(x_layer)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, use_relative_pos=False, max_len=5000):
        super(DecoderLayer, self).__init__()
        self.use_relative_pos = use_relative_pos

        self.self_attn = MultiHeadAttention(d_model, n_heads, use_relative_pos, max_len)
        self.add_norm1 = Add_Norm(d_model, dropout)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, False, max_len)  # cross attention通常不用相对位置编码
        self.add_norm2 = Add_Norm(d_model, dropout)

        self.ffn = Position_wise_Feed_Forward_Net(d_model, d_ff, dropout)
        self.add_norm3 = Add_Norm(d_model, dropout)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None, rel_pos_bias=None):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask, rel_pos_bias)
        output1 = self.add_norm1(x, self_attn_out)

        cross_attn_out, _ = self.cross_attn(output1, encoder_output, encoder_output, memory_mask)
        output2 = self.add_norm2(output1, cross_attn_out)

        ffn_out = self.ffn(output2)
        output3 = self.add_norm3(output2, ffn_out)

        return output3

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1,
                 pos_encoding_type='sinusoidal', max_len=5000):
        super(Decoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        
        # 根据类型选择位置编码
        self.pos_encoding_type = pos_encoding_type
        if pos_encoding_type == 'sinusoidal':
            self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len)
            self.use_relative_pos = False
        elif pos_encoding_type == 'learned':
            self.pos_embed = LearnedPositionalEncoding(d_model, max_len)
            self.use_relative_pos = False
        elif pos_encoding_type == 'relative':
            self.pos_embed = RelativePositionalEncoding(d_model, max_len)
            self.use_relative_pos = True
        elif pos_encoding_type == 'none':
            self.pos_embed = NoPositionalEncoding(d_model, max_len)
            self.use_relative_pos = False
        else:
            raise ValueError(f"Unsupported position encoding type: {pos_encoding_type}")

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, self.use_relative_pos, max_len) 
            for _ in range(num_layers)
        ])

        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        embed_tgt = self.embed(tgt)
        
        # 应用位置编码（相对位置编码除外）
        if self.pos_encoding_type != 'relative':
            tgt_layer = self.pos_embed(embed_tgt)
        else:
            tgt_layer = embed_tgt
        
        # 如果是相对位置编码，计算相对位置偏置
        rel_pos_bias = None
        if self.pos_encoding_type == 'relative':
            seq_len = tgt.shape[1]
            rel_pos_bias = self.pos_embed.get_rel_pos_bias(seq_len)
        
        for layer in self.layers:
            tgt_layer = layer(tgt_layer, encoder_output, tgt_mask, memory_mask, rel_pos_bias)
        
        return self.last_norm(tgt_layer)

class Encoder_Only_MLM(nn.Module):
    def __init__(self, src_vocab, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1,
                 pos_encoding_type='sinusoidal', max_len=5000):
        super(Encoder_Only_MLM, self).__init__()

        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, num_layers, dropout, 
                              pos_encoding_type, max_len)

        self.mlm_head = nn.Linear(d_model, src_vocab)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask):
        output = self.encoder(src, src_mask)
        logits = self.mlm_head(output)

        return logits
    
class Encoder_Only_Transformer(nn.Module):
    def __init__(self, src_vocab, num_classes, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1,
                 pos_encoding_type='sinusoidal', max_len=5000):
        super(Encoder_Only_Transformer, self).__init__()

        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, num_layers, dropout, 
                              pos_encoding_type, max_len)

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
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4, d_ff=512, num_layers=2, dropout=0.1,
                 src_pos_encoding_type='sinusoidal', tgt_pos_encoding_type='sinusoidal', max_len=5000):
        super(Encoder_Decoder_Transformer, self).__init__()

        self.encoder = Encoder(src_vocab, d_model, n_heads, d_ff, num_layers, dropout, 
                              src_pos_encoding_type, max_len)
        self.decoder = Decoder(tgt_vocab, d_model, n_heads, d_ff, num_layers, dropout,
                              tgt_pos_encoding_type, max_len)

        self.generator = nn.Linear(d_model, tgt_vocab)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

        logits = self.generator(decoder_output)

        return logits