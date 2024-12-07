import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv
from torch_geometric.nn import SAGEConv
import os


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)  # Classifier

    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training) 
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = torch.relu(self.conv4(x, edge_index))

        # Pooling to get the graph-level embedding
        graph_embedding = global_mean_pool(x, batch)  # Graph-level embedding before classification

        # Classifier to get the predicted labels
        pred = self.fc(graph_embedding)

        return graph_embedding, pred  # Return both graph embedding and predicted labels

class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv4 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        ))
        self.fc = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x,edge_index,batch):
        # Graph convolution layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training) 
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = torch.relu(self.conv4(x, edge_index))

        # Pooling to get the graph-level embedding
        graph_embedding = global_mean_pool(x, batch)  # Graph-level embedding before classification

        # Classifier to get the predicted labels
        pred = self.fc(graph_embedding)

        return graph_embedding, pred

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        
        # First GraphSAGE convolution layer
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        
        # Second GraphSAGE convolution layer
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels) # classifier
    
    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training) 
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = torch.relu(self.conv3(x, edge_index))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = torch.relu(self.conv4(x, edge_index))

        graph_embedding = global_mean_pool(x, batch)
        pred = self.fc(graph_embedding)

        return graph_embedding, pred


# Utility function to train the model, save embeddings, and test the model
def train_and_test_model(model, train_loader, test_loader, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    test_acc = []
    emb = []
    true_labels = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            graph_embedding, pred = model(data.x, data.edge_index, data.batch)
            if epoch == epochs - 1: # save embedding of training point at last epoch
                for i in range(len(data)):
                    emb.append(graph_embedding[i])
                    true_labels.append(data[i].y)

            loss = criterion(pred, data.y)  # Compute loss based on predicted labels
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Test the model after each epoch and save embeddings
    model.eval()
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            graph_embedding, pred = model(data.x, data.edge_index, data.batch)
            correct += (pred.argmax(dim=1) == data.y).sum().item()
            for i in range(len(data)):
                emb.append(graph_embedding[i])
                true_labels.append(data[i].y)
            accuracy = correct / len(test_loader.dataset)
            test_acc.append(accuracy)
        print(f"Test Accuracy: {np.mean(test_acc):.4f}")


    return np.mean(test_acc), emb, true_labels
