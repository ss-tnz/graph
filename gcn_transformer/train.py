import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gcn import GCN
from model import Transformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = './data/cora'
dataset = Planetoid(root=path, name='Cora')
data = dataset[0]
X = data.x
y = data.y
edge_index = data.edge_index
hidden_dim = 128

num_layers = 4
d_model = 128
num_heads = 8
d_ff = 256
batch_size =32
learning_rate = 0.001
num_epochs = 100

gcn_model = GCN(X.size(1), hidden_dim).to(device)

transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True)
train_data, val_data, test_data = transform(data)
print(train_data)


train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
test_loader = DataLoader([test_data], batch_size=batch_size, shuffle=False)


transformer_model = Transformer(num_layers, d_model, num_heads, d_ff, X.size(1), X.size(1), edge_index.max().item() + 1).to(device)


optimizer = optim.Adam(list(gcn_model.parameters()) + list(transformer_model.parameters()), lr=learning_rate)
criterion = nn.MSELoss()

train_losses = []
test_losses = []
desired_batch_size = 32
desired_sequence_length = 1
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        inputs = batch.x.clone().detach()
        edge_index_batch = batch.edge_index.clone().detach()

        gcn_output = gcn_model(inputs, edge_index_batch)

        decoder_outputs = transformer_model(gcn_output.unsqueeze(0), mask=None)

        loss = criterion(decoder_outputs, inputs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    test_loss = 0
    with torch.no_grad():
        all_decoded_outputs = []
        for batch in test_loader:
            inputs, edge_index_batch = batch.x.clone().detach(), batch.edge_index.clone().detach()
            gcn_output = gcn_model(inputs, edge_index_batch)
            decoder_outputs = transformer_model(gcn_output.unsqueeze(0), mask=None)
            all_decoded_outputs.append(decoder_outputs)
            loss = criterion(decoder_outputs, inputs)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")

# Plot the loss curve
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()
plt.savefig('loss_curve.png')
