

import math
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv

import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
path = './data/cora'
dataset = Planetoid(root=path, name='Cora')
data = dataset[0]
X = data.x
y = data.y
num_nodes = X.shape[0]
num_features = X.shape[1]


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


edge_index = data.edge_index


hidden_dim = 128
gcn_model = GCN(num_features, hidden_dim)


gcn_output = gcn_model(X, edge_index)
gcn_output_dense = gcn_output
sequences = [gcn_output_dense[i].unsqueeze(0) for i in range(num_nodes)]

num_layers = 4
d_model = 128
num_heads = 8
d_ff = 256
max_seq_len = len(sequences[0])
input_vocab_size = num_features
output_vocab_size = num_features  # You can adjust this based on your task
batch_size = 32
learning_rate = 0.001
num_epochs = 100

#位置编码，固定
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


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
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        x = self.positional_encoding(x)

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
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, output_vocab_size)

    def forward(self, x, enc_output, self_mask, enc_mask):
        x = self.positional_encoding(x)

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



gcn_output_stacked = torch.stack(sequences, dim=0)


train_ratio = 0.8
train_size = int(train_ratio * len(gcn_output_stacked))
train_data = gcn_output_stacked[:train_size]
test_data = gcn_output_stacked[train_size:]

train_dataset = TensorDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(test_data)

model = Transformer(num_layers, d_model, num_heads, d_ff, hidden_dim, hidden_dim, num_nodes)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

train_losses = []  # 存储训练集损失
test_losses = []  # 存储测试集损失

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        inputs = batch[0]

        optimizer.zero_grad()
        decoder_outputs = model(inputs, mask=None)

        loss = criterion(decoder_outputs, inputs)
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # 在测试集上进行评估
    test_loss = 0
    test_data = []
    with torch.no_grad():
        all_decoded_outputs = []
        for batch in test_dataloader:
            inputs = batch[0]
            decoder_outputs = model(inputs, mask=None)
            all_decoded_outputs.append(decoder_outputs)
            loss = criterion(decoder_outputs, inputs)
            test_loss += loss.item()


    avg_test_loss = test_loss / len(test_dataloader)
    test_losses.append(avg_test_loss)
    combined_decoded_outputs = torch.cat(all_decoded_outputs, dim=0)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")
    output_file_path = 'decoded_outputs.pth'
    torch.save(combined_decoded_outputs, output_file_path)




# 绘制损失图
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
plt.savefig('loss_curve.png')