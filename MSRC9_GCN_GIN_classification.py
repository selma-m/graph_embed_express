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

dataset = TUDataset(root='./data', name='MSRC_9', transform=T.NormalizeFeatures())

train_loader = DataLoader(dataset[:int(0.8*len(dataset))], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[int(0.8*len(dataset)):], batch_size=32, shuffle=False)

labels = np.array(dataset.y)

print("Training GCN on MSRC_9 dataset")
gcn_model = GCNModel(in_channels=dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes)
gcn_acc_nci1, gcn_embed_nci1, all_labels = train_and_test_model(gcn_model, train_loader, test_loader)

gcn_embed_nci1 = torch.stack(gcn_embed_nci1).detach().numpy()
all_labels = torch.stack(all_labels).detach().numpy()

print("DSE: ", diffusion_spectral_entropy(gcn_embed_nci1))

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(gcn_embed_nci1)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = all_labels)
plt.savefig('msrc9_gcn_phate_plot.png', dpi=300) 

print("Training GIN on MSRC_9 dataset")
gin_model = GINModel(in_channels=dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes)
gin_acc_nci1, gin_embed_nci1, all_labels = train_and_test_model(gin_model, train_loader, test_loader)

gin_embed_nci1 = torch.stack(gin_embed_nci1).detach().numpy()
all_labels = torch.stack(all_labels).detach().numpy()
print("DSE: ", diffusion_spectral_entropy(gin_embed_nci1))

phate_op = phate.PHATE(verbose = 0)
phate_data = phate_op.fit_transform(gin_embed_nci1)
phate.plot.scatter2d(phate_data, figsize=(8,6), c = labels)
plt.savefig('msrc9_gin_phate_plot.png', dpi=300) 
