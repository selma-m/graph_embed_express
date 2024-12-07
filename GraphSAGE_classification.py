import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from models/gcn_gin_models import *
from DSE import *

import phate
import matplotlib.pyplot as plt

torch.manual_seed(29)
np.random.seed(29)

# NCI1 dataset

nci1_dataset = TUDataset(root='./data', name='NCI1', transform=T.NormalizeFeatures())
nci1_train_loader = DataLoader(nci1_dataset[:int(0.8*len(nci1_dataset))], batch_size=32, shuffle=True)
nci1_test_loader = DataLoader(nci1_dataset[int(0.8*len(nci1_dataset)):], batch_size=32, shuffle=False)

print("Training GraphSAGE on NCI1 dataset")
nci1_model = GCNModel(in_channels=nci1_dataset.num_features, hidden_channels=64, out_channels=nci1_dataset.num_classes)
acc_nci1, embed_nci1, all_labels_nci1 = train_and_test_model(nci1_model, nci1_train_loader, nci1_test_loader)

embed_nci1 = torch.stack(embed_nci1).detach().numpy()
all_labels_nci1 = torch.stack(all_labels_nci1).detach().numpy()

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(embed_nci1)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = all_labels_nci1)
plt.savefig('nci1_graphsage_phate_plot.png', dpi=300) 

print("DSE: ", diffusion_spectral_entropy(embed_nci1))

# PROTEINS dataset

proteins_dataset = TUDataset(root='./data', name='PROTEINS', transform=T.NormalizeFeatures())
proteins_train_loader = DataLoader(proteins_dataset[:int(0.8*len(proteins_dataset))], batch_size=32, shuffle=True)
proteins_test_loader = DataLoader(proteins_dataset[int(0.8*len(proteins_dataset)):], batch_size=32, shuffle=False)

print("Training GraphSAGE on PROTEINS dataset")
proteins_model = GraphSAGE(in_channels=proteins_dataset.num_features, hidden_channels=64, out_channels=proteins_dataset.num_classes)
acc_proteins, embed_proteins, all_labels_proteins = train_and_test_model(proteins_model, proteins_train_loader, proteins_test_loader)

embed_proteins = torch.stack(embed_proteins).detach().numpy()
all_labels_proteins = torch.stack(all_labels_proteins).detach().numpy()

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(embed_proteins)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = all_labels_proteins)
plt.savefig('proteins_graphsage_phate_plot.png', dpi=300) 
print("DSE: ", diffusion_spectral_entropy(embed_proteins))



# MSRC9

msrc9_dataset = TUDataset(root='./data', name='MSRC_9', transform=T.NormalizeFeatures())
msrc9_train_loader = DataLoader(msrc9_dataset[:int(0.8*len(msrc9_dataset))], batch_size=32, shuffle=True)
msrc9_test_loader = DataLoader(msrc9_dataset[int(0.8*len(msrc9_dataset)):], batch_size=32, shuffle=False)

print("Training GraphSAGE on MSRC_9 dataset")
msrc9_model = GraphSAGE(in_channels=msrc9_dataset.num_features, hidden_channels=64, out_channels=msrc9_dataset.num_classes)
acc_msrc9, embed_msrc9, all_labels_msrc9 = train_and_test_model(msrc9_model, msrc9_train_loader, msrc9_test_loader)

embed_msrc9 = torch.stack(embed_msrc9).detach().numpy()
all_labels_msrc9 = torch.stack(all_labels_msrc9).detach().numpy()

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(embed_msrc9)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = all_labels_msrc9)
plt.savefig('msrc9_graphsage_phate_plot.png', dpi=300) 
print("DSE: ", diffusion_spectral_entropy(embed_msrc9))


# ENZYMES dataset

enzymes_dataset = TUDataset(root='./data', name='ENZYMES', transform=T.NormalizeFeatures())
enzymes_train_loader = DataLoader(enzymes_dataset[:int(0.8*len(enzymes_dataset))], batch_size=32, shuffle=True)
enzymes_test_loader = DataLoader(enzymes_dataset[int(0.8*len(enzymes_dataset)):], batch_size=32, shuffle=False)

print("Training GraphSAGE on ENZYMES dataset")
enzymes_model = GraphSAGE(in_channels=enzymes_dataset.num_features, hidden_channels=64, out_channels=enzymes_dataset.num_classes)
acc_enzymes, embed_enzymes, all_labels_enzymes = train_and_test_model(enzymes_model, enzymes_train_loader, enzymes_test_loader)

embed_enzymes = torch.stack(embed_enzymes).detach().numpy()
all_labels_enzymes = torch.stack(all_labels_enzymes).detach().numpy()

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(embed_enzymes)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = all_labels_enzymes)
plt.savefig('enzymes_graphsage_phate_plot.png', dpi=300) 
print("DSE: ", diffusion_spectral_entropy(embed_enzymes))


# MSRC21 dataset

msrc21_dataset = TUDataset(root='./data', name='MSRC_21', transform=T.NormalizeFeatures())
msrc21_train_loader = DataLoader(msrc21_dataset[:int(0.8*len(msrc21_dataset))], batch_size=32, shuffle=True)
msrc21_test_loader = DataLoader(msrc21_dataset[int(0.8*len(msrc21_dataset)):], batch_size=32, shuffle=False)

print("Training GraphSAGE on MSRC_21 dataset")
msrc21_model = GraphSAGE(in_channels=msrc21_dataset.num_features, hidden_channels=64, out_channels=msrc21_dataset.num_classes)
acc_msrc21, embed_msrc21, all_labels_msrc21 = train_and_test_model(msrc21_model, msrc21_train_loader, msrc21_test_loader)

embed_msrc21 = torch.stack(embed_msrc21).detach().numpy()
all_labels_msrc21 = torch.stack(all_labels_msrc21).detach().numpy()

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(embed_msrc21)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = all_labels_msrc21)
plt.savefig('msrc21_graphsage_phate_plot.png', dpi=300) 
print("DSE: ", diffusion_spectral_entropy(embed_msrc21))


