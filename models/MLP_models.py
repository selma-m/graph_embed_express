import torch
import torch.nn as nn

"""
    Simple MLP classifiers
"""

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.bn1 = nn.BatchNorm1d(hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.dropout = nn.Dropout(p=0.2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2) 
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.Tanh()  
        self.softmax = nn.Softmax()  

    def forward(self, x):
        x = self.bn0(x)
        x = self.relu(self.fc1(x)) 
        x = self.bn1(x) # add dropout layer?
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.relu(self.fc3(x))
        embed = x
        x = self.dropout(x)
        x = self.softmax(self.fc4(x))   
        return x, embed
    
class MLP_multi(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_multi, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, output_dim)
        self.relu = nn.ReLU()  
        self.softmax = nn.Softmax(dim = 0)  

    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.bn1(x)
        x = self.relu(self.fc2(x))
        embed = x
        x = self.bn2(x)
        x = self.fc3(x)  
        return self.softmax(x), embed
    
class MLP_multi2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_multi2, self).__init__()
        self.dropout = nn.Dropout(p = 0.2)
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()  
        self.softmax = nn.Softmax(dim = 0)  

    def forward(self, x):
        x = self.tanh(self.fc1(x)) 
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.tanh(self.fc2(x))
        x = self.bn2(x)
        x  = self.dropout(x)
        x = self.tanh(self.fc3(x)) 
        embed = x
        x = self.fc4(x)
        return self.softmax(x), embed
