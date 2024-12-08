from learnable_geometric_scattering import *
from DSE import *
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import phate
import matplotlib.pyplot as plt

torch.manual_seed(29)
np.random.seed(29)


# NCI1 dataset

nci1_dataset = TUDataset(root='./data', name='NCI1', transform=T.NormalizeFeatures())
nci1_train_loader = DataLoader(nci1_dataset[:int(0.8*len(nci1_dataset))], batch_size=32, shuffle=True)
nci1_test_loader = DataLoader(nci1_dataset[int(0.8*len(nci1_dataset)):], batch_size=32, shuffle=False)

in_channels = nci1_dataset.num_node_features
out_channels = nci1_dataset.num_classes

nci1_model = TSNet(in_channels, out_channels, trainable_laziness=True, trainable_f=True)
nci1_optimizer = torch.optim.Adam(nci1_model.parameters(), lr=0.001)

num_epochs = 100
nci1_embeddings = []
nci1_true_labels = []

print("Training LEGS on NCI1 dataset")

# training loop
for epoch in range(num_epochs):
  nci1_model.train()
  total_loss = 0
  for data in nci1_train_loader:
    nci1_optimizer.zero_grad()
    out,emb = nci1_model(data)
    if epoch == num_epochs -1:
      for i in range(len(data)):
        nci1_embeddings.append(emb[i])
        nci1_true_labels.append(data[i].y)
    loss = nci1_model.loss_function(out, data.y)
    loss.backward()
    nci1_optimizer.step()
    total_loss += loss.item()
  if(epoch+1)%10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(nci1_train_loader):.4f}")

# testing loop
nci1_model.eval()
test_acc = []
correct = 0
with torch.no_grad():
  for data in nci1_test_loader:
    out,embed = nci1_model(data)
    pred = out.argmax(dim=1)
    correct += (pred == data.y).sum().item()
    for i in range((len(data))):
      nci1_embeddings.append(embed[i])
      nci1_true_labels.append(data[i].y)
    accuracy = correct / len(nci1_test_loader.dataset)
    test_acc.append(accuracy)
print(f"Test Accuracy: {np.mean(test_acc):.4f}")


nci1_embeddings = torch.stack(nci1_embeddings).detach().numpy()
nci1_true_labels = torch.stack(nci1_true_labels).detach().numpy()

print("DSE: ", diffusion_spectral_entropy(nci1_embeddings))

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(nci1_embeddings)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = nci1_true_labels)
plt.savefig('nci1_legs_LRW_phate_plot.png', dpi=300) 
  


#MSRC_9 dataset

msrc9_dataset = TUDataset(root='./data', name='MSRC_9', transform=T.NormalizeFeatures())
msrc9_train_loader = DataLoader(msrc9_dataset[:int(0.8*len(msrc9_dataset))], batch_size=32, shuffle=True)
msrc9_test_loader = DataLoader(msrc9_dataset[int(0.8*len(msrc9_dataset)):], batch_size=32, shuffle=False)

in_channels = msrc9_dataset.num_node_features
out_channels = msrc9_dataset.num_classes

msrc9_model = TSNet(in_channels, out_channels, trainable_laziness=True, trainable_f=True)
msrc9_optimizer = torch.optim.Adam(msrc9_model.parameters(), lr=0.001)

num_epochs = 100
msrc9_embeddings = []
msrc9_true_labels = []

print("Training LEGS on MSRC_9 dataset")

# training loop
for epoch in range(num_epochs):
  msrc9_model.train()
  total_loss = 0
  for data in msrc9_train_loader:
    msrc9_optimizer.zero_grad()
    out,emb = msrc9_model(data)
    if epoch == num_epochs -1:
      for i in range(len(data)):
        msrc9_embeddings.append(emb[i])
        msrc9_true_labels.append(data[i].y)
    loss = msrc9_model.loss_function(out, data.y)
    loss.backward()
    msrc9_optimizer.step()
    total_loss += loss.item()
  if(epoch+1)%10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(msrc9_train_loader):.4f}")

# testing loop
msrc9_model.eval()
test_acc = []
correct = 0
with torch.no_grad():
  for data in msrc9_test_loader:
    out,embed = msrc9_model(data)
    pred = out.argmax(dim=1)
    correct += (pred == data.y).sum().item()
    for i in range((len(data))):
      msrc9_embeddings.append(embed[i])
      msrc9_true_labels.append(data[i].y)
    accuracy = correct / len(msrc9_test_loader.dataset)
    test_acc.append(accuracy)
print(f"Test Accuracy: {np.mean(test_acc):.4f}")


msrc9_embeddings = torch.stack(msrc9_embeddings).detach().numpy()
msrc9_true_labels = torch.stack(msrc9_true_labels).detach().numpy()

print("DSE: ", diffusion_spectral_entropy(msrc9_embeddings))

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(msrc9_embeddings)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = msrc9_true_labels)
plt.savefig('msrc9_legs_LRW_phate_plot.png', dpi=300) 
  

# MSRC_21
msrc21_dataset = TUDataset(root='./data', name='MSRC_21', transform=T.NormalizeFeatures())
msrc21_train_loader = DataLoader(msrc21_dataset[:int(0.8*len(msrc21_dataset))], batch_size=32, shuffle=True)
msrc21_test_loader = DataLoader(msrc21_dataset[int(0.8*len(msrc21_dataset)):], batch_size=32, shuffle=False)

in_channels = msrc21_dataset.num_node_features
out_channels = msrc21_dataset.num_classes

msrc21_model = TSNet(in_channels, out_channels, trainable_laziness=True, trainable_f=True)
msrc21_optimizer = torch.optim.Adam(msrc21_model.parameters(), lr=0.001)

num_epochs = 100
msrc21_embeddings = []
msrc21_true_labels = []

print("Training LEGS on MSRC_21 dataset")

# training loop
for epoch in range(num_epochs):
  msrc21_model.train()
  total_loss = 0
  for data in msrc21_train_loader:
    msrc21_optimizer.zero_grad()
    out,emb = msrc21_model(data)
    if epoch == num_epochs -1:
      for i in range(len(data)):
        msrc21_embeddings.append(emb[i])
        msrc21_true_labels.append(data[i].y)
    loss = msrc21_model.loss_function(out, data.y)
    loss.backward()
    msrc21_optimizer.step()
    total_loss += loss.item()
  if(epoch+1)%10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(msrc21_train_loader):.4f}")

# testing loop
msrc21_model.eval()
test_acc = []
correct = 0
with torch.no_grad():
  for data in msrc21_test_loader:
    out,embed = msrc21_model(data)
    pred = out.argmax(dim=1)
    correct += (pred == data.y).sum().item()
    for i in range((len(data))):
      msrc21_embeddings.append(embed[i])
      msrc21_true_labels.append(data[i].y)
    accuracy = correct / len(msrc21_test_loader.dataset)
    test_acc.append(accuracy)
print(f"Test Accuracy: {np.mean(test_acc):.4f}")


msrc21_embeddings = torch.stack(msrc21_embeddings).detach().numpy()
msrc21_true_labels = torch.stack(msrc21_true_labels).detach().numpy()

print("DSE: ", diffusion_spectral_entropy(msrc21_embeddings))

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(msrc21_embeddings)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = msrc21_true_labels)
plt.savefig('msrc21_legs_LRW_phate_plot.png', dpi=300) 
  

# PROTEINS dataset
proteins_dataset = TUDataset(root='./data', name='PROTEINS', transform=T.NormalizeFeatures())
proteins_train_loader = DataLoader(proteins_dataset[:int(0.8*len(proteins_dataset))], batch_size=32, shuffle=True)
proteins_test_loader = DataLoader(proteins_dataset[int(0.8*len(proteins_dataset)):], batch_size=32, shuffle=False)

in_channels = proteins_dataset.num_node_features
out_channels = proteins_dataset.num_classes

proteins_model = TSNet(in_channels, out_channels, trainable_laziness=True, trainable_f=True)
proteins_optimizer = torch.optim.Adam(proteins_model.parameters(), lr=0.001)

num_epochs = 100
proteins_embeddings = []
proteins_true_labels = []

print("Training LEGS on PROTEINS dataset")

# training loop
for epoch in range(num_epochs):
  proteins_model.train()
  total_loss = 0
  for data in proteins_train_loader:
    proteins_optimizer.zero_grad()
    out,emb = proteins_model(data)
    if epoch == num_epochs -1:
      for i in range(len(data)):
        proteins_embeddings.append(emb[i])
        proteins_true_labels.append(data[i].y)
    loss = proteins_model.loss_function(out, data.y)
    loss.backward()
    proteins_optimizer.step()
    total_loss += loss.item()
  if(epoch+1)%10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(proteins_train_loader):.4f}")

# testing loop
proteins_model.eval()
test_acc = []
correct = 0
with torch.no_grad():
  for data in proteins_test_loader:
    out,embed = proteins_model(data)
    pred = out.argmax(dim=1)
    correct += (pred == data.y).sum().item()
    for i in range((len(data))):
      proteins_embeddings.append(embed[i])
      proteins_true_labels.append(data[i].y)
    accuracy = correct / len(proteins_test_loader.dataset)
    test_acc.append(accuracy)
print(f"Test Accuracy: {np.mean(test_acc):.4f}")


proteins_embeddings = torch.stack(proteins_embeddings).detach().numpy()
proteins_true_labels = torch.stack(proteins_true_labels).detach().numpy()

print("DSE: ", diffusion_spectral_entropy(proteins_embeddings))

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(proteins_embeddings)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = proteins_true_labels)
plt.savefig('proteins_legs_LRW_phate_plot.png', dpi=300) 



# ENZYMES dataset
enzymes_dataset = TUDataset(root='./data', name='ENZYMES', transform=T.NormalizeFeatures())
enzymes_train_loader = DataLoader(enzymes_dataset[:int(0.8*len(enzymes_dataset))], batch_size=32, shuffle=True)
enzymes_test_loader = DataLoader(enzymes_dataset[int(0.8*len(enzymes_dataset)):], batch_size=32, shuffle=False)

in_channels = enzymes_dataset.num_node_features
out_channels = enzymes_dataset.num_classes

enzymes_model = TSNet(in_channels, out_channels, trainable_laziness=True, trainable_f=True)
enzymes_optimizer = torch.optim.Adam(enzymes_model.parameters(), lr=0.001)

num_epochs = 100
enzymes_embeddings = []
enzymes_true_labels = []

print("Training LEGS on ENZYMES dataset")

# training loop
for epoch in range(num_epochs):
  enzymes_model.train()
  total_loss = 0
  for data in enzymes_train_loader:
    enzymes_optimizer.zero_grad()
    out,emb = enzymes_model(data)
    if epoch == num_epochs -1:
      for i in range(len(data)):
        enzymes_embeddings.append(emb[i])
        enzymes_true_labels.append(data[i].y)
    loss = enzymes_model.loss_function(out, data.y)
    loss.backward()
    enzymes_optimizer.step()
    total_loss += loss.item()
  if(epoch+1)%10 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(enzymes_train_loader):.4f}")

# testing loop
enzymes_model.eval()
test_acc = []
correct = 0
with torch.no_grad():
  for data in enzymes_test_loader:
    out,embed = enzymes_model(data)
    pred = out.argmax(dim=1)
    correct += (pred == data.y).sum().item()
    for i in range((len(data))):
      enzymes_embeddings.append(embed[i])
      enzymes_true_labels.append(data[i].y)
    accuracy = correct / len(enzymes_test_loader.dataset)
    test_acc.append(accuracy)
print(f"Test Accuracy: {np.mean(test_acc):.4f}")


enzymes_embeddings = torch.stack(enzymes_embeddings).detach().numpy()
enzymes_true_labels = torch.stack(enzymes_true_labels).detach().numpy()

print("DSE: ", diffusion_spectral_entropy(enzymes_embeddings))

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(enzymes_embeddings)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = enzymes_true_labels)
plt.savefig('enzymes_legs_LRW_phate_plot.png', dpi=300) 
