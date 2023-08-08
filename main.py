import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.utils import to_dense_adj
from torch.utils.data import DataLoader, TensorDataset

path = './data/cora'
dataset = Planetoid(root=path, name='Cora')
data = dataset[0]
X = data.x
y = data.y
num_nodes = X.shape[0]
num_features = X.shape[1]
print(data)
print(X)

# 稀疏矩阵转换
adj_matrix = to_dense_adj(data.edge_index).squeeze()
# 求和函数聚合
graph_representations = torch.mm(adj_matrix, X) / (adj_matrix.sum(dim=1, keepdim=True) + 1e-6)

## 构建序列数据集
sequences = [graph_representations[i].unsqueeze(0) for i in range(num_nodes)]
labels = y

# 定义RNN Encoder-Decoder模型
class RNNEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNEncoderDecoder, self).__init__()
        self.encoder_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.decoder_rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, sequences):
        encoder_output, _ = self.encoder_rnn(sequences)
        decoder_output, _ = self.decoder_rnn(encoder_output)
        output = self.linear(decoder_output)
        return output

# 训练模型
hidden_size = 64
num_classes = dataset.num_classes
model = RNNEncoderDecoder(num_features, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32
dataset = [(seq, label) for seq, label in zip(sequences, labels)]
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    total_loss = 0
    for sequences_batch, labels_batch in dataloader:
        optimizer.zero_grad()
        output = model(sequences_batch)
        loss = criterion(output.view(-1, num_classes), labels_batch)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss}')

# 使用训练好的模型进行预测
with torch.no_grad():
    predicted_labels = []
    for i in range(num_nodes):
        test_sequence = graph_representations[i].unsqueeze(0).unsqueeze(0)  # 增加batch维度
        output = model(test_sequence)
        predicted_label = output.argmax(dim=-1).item()
        predicted_labels.append(predicted_label)

predicted_labels = torch.tensor(predicted_labels)
accuracy = (predicted_labels == y).sum().item() / num_nodes

print("Accuracy:", accuracy)