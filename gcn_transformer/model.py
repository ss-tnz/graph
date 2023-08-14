import math
import torch.nn.functional as F
import torch.nn as nn
import torch

#位置编码，在图数据中不适用
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=0.1)
#
#         pe = torch.zeros(max_seq_len, d_model)
#         position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1)]
#         return self.dropout(x)


def scaled_dot_product_attention(query, key, value, mask=None):
    matmul = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scaled_attention_logits = matmul / math.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(scaled_attention)
        return output, attention_weights

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.multihead_attention(x, x, x, mask)
        out1 = self.layernorm1(x + self.dropout1(attn_output))

        ffn_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))

        return out2


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        # self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):


        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.layernorm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, enc_mask):
        self_attention_output, _ = self.self_attention(x, x, x, self_mask)
        out1 = self.layernorm1(x + self.dropout1(self_attention_output))

        encoder_attention_output, _ = self.encoder_attention(out1, enc_output, enc_output, enc_mask)
        out2 = self.layernorm2(out1 + self.dropout2(encoder_attention_output))

        ffn_output = self.feed_forward(out2)
        out3 = self.layernorm3(out2 + self.dropout3(ffn_output))

        return out3


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, output_vocab_size, max_seq_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        # self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, output_vocab_size)

    def forward(self, x, enc_output, self_mask, enc_mask):


        for layer in self.decoder_layers:
            x = layer(x, enc_output, self_mask, enc_mask)

        return self.output_layer(x)


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, output_vocab_size, max_seq_len,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, max_seq_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, output_vocab_size, max_seq_len, dropout)

    def forward(self, x, mask):
        encoder_output = self.encoder(x, mask)
        decoder_output = self.decoder(x, encoder_output, mask, mask)
        return decoder_output
