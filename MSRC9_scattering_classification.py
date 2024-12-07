from models/geometric_scattering import *
from models/simple_mlp import *
from DSE import *
import scipy
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import phate
import matplotlib.pyplot as plt

np.random.seed(29)
torch.manual_seed(29)

# Load data
msrc9_dataset = TUDataset(root='./data', name='MSRC_9', transform=NormalizeFeatures()) # the dataset has 8 classes
# Each node is equipped with a one-hot vector of size 10

# In the following loop, we extract the node signals, graph labels
# We compute scattering coefficients
signals = []
edge_indices= []
labels = []
adjacency_matrices = []
graph_scattering= []
blis_graph_scattering = []
for i in range(len(msrc9_dataset)):
    data = msrc9_dataset[i] # this gives one graph
    signals.append(data.x) # this is the node signals
    edge_indices.append(data.edge_index) 
    num_nodes = len(data.x)
    # We create the adjacency matrix
    adj_matrix_dense = torch.sparse_coo_tensor(
        data.edge_index,torch.ones(data.edge_index.shape[1]),  # All edges have weight 1 (or use other edge attributes)
        (num_nodes, num_nodes)  # Shape of the adjacency matrix
        ).to_dense()
    adjacency_matrices.append(adj_matrix_dense)
    labels.append(data.y)
    # We compute the scattering coefficients
    graph_level_features = graph_level(data.x,adj_matrix_dense) # this has length 10
    blis_graph_level_features = graph_level_blis(data.x, adj_matrix_dense)
    graph_scattering.append(graph_level_features)
    blis_graph_scattering.append(blis_graph_level_features)

graph_scattering = np.array(graph_scattering)
blis_graph_scattering = np.array(blis_graph_scattering)
labels = np.array(labels)

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(graph_scattering)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = labels)
plt.savefig('msrc9_scattering_phate_plot.png', dpi=300) 

phate_data = phate_op.fit_transform(blis_graph_scattering)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = labels)
plt.savefig('msrc9_blis_scattering_phate_plot.png', dpi=300)

print("DSE for scattering:", diffusion_spectral_entropy(graph_scattering))
print("DSE for BLIS:", diffusion_spectral_entropy(blis_graph_scattering))

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(graph_scattering, labels, test_size=0.2, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(blis_graph_scattering, labels, test_size=0.2, random_state=42)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
X_train_b = torch.tensor(X_train_b, dtype=torch.float32)
X_test_b = torch.tensor(X_test_b, dtype=torch.float32)
y_train_b = torch.tensor(y_train_b, dtype=torch.float32).view(-1, 1)
y_test_b = torch.tensor(y_test_b, dtype=torch.float32).view(-1, 1)

# Create data loaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
train_data_b = TensorDataset(X_train_b, y_train_b)
test_data_b = TensorDataset(X_test_b, y_test_b)
train_loader_b = DataLoader(train_data_b, batch_size=16, shuffle=True)
test_loader_b = DataLoader(test_data_b, batch_size=16, shuffle=False)

input_dim = X_train.shape[1] # this is 10
hidden_dim = 64
output_dim = 8 # this is the number of classes in the dataset
num_epochs = 100

model = MLP_multi(input_dim, hidden_dim, output_dim)
model_b = MLP_multi(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer_b = optim.Adam(model_b.parameters(), lr=0.0001)

embeddings = []
all_labels = []
embeddings_b = []
all_labels_b = []

print("Training on scattering for MSRC_9")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        y_pred,embed = model(X_batch)
        if epoch == num_epochs - 1: 
            for i in range(len(y_batch)):
                embeddings.append(embed[i])
                all_labels.append(y_batch[i])
        # Compute loss
        loss = criterion(y_pred, y_batch.squeeze(-1).long())
    
        # Backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

print("Testing on scattering for MSRC_9")
model.eval()
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for X_batch,y_batch in test_loader:
        y_pred_test, embed = model(X_batch)
        for i in range(len(y_batch)):
            embeddings.append(embed[i])
            all_labels.append(y_batch[i])
        y_pred_test = torch.argmax(y_pred_test, dim = 1)
        total_samples += y_batch.size(0)
        correct_predictions += (y_pred_test == y_batch.squeeze(-1).long()).sum().item()
test_accuracy = 100 * correct_predictions / total_samples
print(f'Accuracy on test set: {test_accuracy:.4f}')


print("Training on BLIS for MSRC_9")
for epoch in range(num_epochs):
    model_b.train()
    total_loss = 0
    for X_batch, y_batch in train_loader_b:
        optimizer_b.zero_grad()

        # Forward pass
        y_pred, embed = model_b(X_batch)

        if epoch == num_epochs - 1: 
            for i in range(len(y_batch)):
                embeddings_b.append(embed[i])
                all_labels_b.append(y_batch[i])

        # Compute loss
        loss = criterion(y_pred, y_batch.squeeze(-1).long())
    
        # Backpropagation
        loss.backward()
        optimizer_b.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader_b):.4f}')

print("Testing on BLIS for MSRC_9")
model_b.eval()
correct_predictions = 0
total_samples = 0
with torch.no_grad():
    for X_batch,y_batch in test_loader_b:
        y_pred_test,embed = model_b(X_batch)
        for i in range(len(y_batch)):
            embeddings_b.append(embed[i])
            all_labels_b.append(y_batch[i])
        y_pred_test = torch.argmax(y_pred_test, dim = 1)
        total_samples += y_batch.size(0)
        correct_predictions += (y_pred_test == y_batch.squeeze(-1).long()).sum().item()
test_accuracy = 100 * correct_predictions / total_samples
print(f'Accuracy on test set: {test_accuracy:.4f}')

embeddings = torch.stack(embeddings).detach().numpy()
all_labels = torch.stack(all_labels).detach().numpy()
embeddings_b = torch.stack(embeddings_b).detach().numpy()
all_labels_b = torch.stack(all_labels_b).detach().numpy()

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(embeddings)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = all_labels)
plt.savefig('msrc9_pen_scattering_phate_plot.png', dpi=300) 

phate_data = phate_op.fit_transform(embeddings_b)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = all_labels_b)
plt.savefig('msrc9__pen_blis_scattering_phate_plot.png', dpi=300)

