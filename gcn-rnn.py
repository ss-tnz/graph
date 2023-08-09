import torch
from torch_geometric.datasets import Planetoid
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader


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


class RNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNAutoencoder, self).__init__()
        self.encoder_rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.decoder_rnn = nn.RNN(hidden_size, input_size, batch_first=True)
        self.linear = nn.Linear(input_size, input_size)

    def forward(self, sequences):
        encoded, _ = self.encoder_rnn(sequences)
        decoded, _ = self.decoder_rnn(encoded)
        output = self.linear(decoded)
        return output






labels = y


model = RNNAutoencoder(hidden_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32
dataset = [(seq, seq) for seq in sequences]
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    total_loss = 0
    for sequences_batch, _ in dataloader:
        optimizer.zero_grad()
        output = model(sequences_batch)
        loss = criterion(output, sequences_batch)

        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss}')


with torch.no_grad():
    reconstructed_sequences = []
    for i in range(num_nodes):
        test_sequence = gcn_output_dense[i].unsqueeze(0)  # Add batch dimension
        output = model(test_sequence)
        reconstructed_sequences.append(output.squeeze())

reconstructed_sequences = torch.stack(reconstructed_sequences)
reconstruction_error = torch.mean((reconstructed_sequences - gcn_output_dense) ** 2)
print(gcn_output_dense)
print(reconstructed_sequences)

print("Reconstruction Error:", reconstruction_error.item())
