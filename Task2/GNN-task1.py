import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,BatchNorm
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import pickle
import torch.nn as nn
import os
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.bn3 = BatchNorm(out_channels)
        self.fc1 = torch.nn.Linear(out_channels, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 1)  # Output is a single number
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Mean of node features
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
   
def evalwithGNN(state):
    num_node_features = 1  # 根据具体情况调整
    hidden_channels = 16  # 隐藏层大小，可以根据需要调整
    output_dim = 8
    model = GNN(num_node_features, hidden_channels, output_dim)

    model.load_state_dict(torch.load("modelpath.pth",))

    data=getdata(state)
    test_data=Data(x=torch.tensor(torch.cat((data['node_type'], data['num_inverted_predecessors']), dim=0), dtype=torch.float32).unsqueeze(1), 
                edge_index=data['edge_index'])
    output = model(test_data)
    predicted_value = output.item()
    return predicted_value
