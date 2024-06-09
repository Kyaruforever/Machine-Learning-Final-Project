import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import pickle
import os
from torch_geometric.nn import GCNConv,BatchNorm


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
        x_ori = x.clone()
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F. gelu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F. gelu(x)
        x = x_ori + x
        x = torch.mean(x, dim=0)  # Mean of node features
        x = self.fc1(x)
        x = F. gelu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = F. gelu(x)
        x = self.fc3(x)
        x = F. gelu(x)
        x = self.fc4(x)
        return x
    
class CustomDataset(Dataset):
    def __init__(self, data_list, targets):
        self.data_list = data_list
        self.targets = targets

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        target = self.targets[idx]
        return data, target

def load_data(directory):
    data_list = []
    targets = []
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 构建文件的完整路径
        filepath = os.path.join(directory, filename)
        
        # 检查路径是否为文件
        if os.path.isfile(filepath):
            
            file = open(filepath,'rb')
            data = pickle.load(file)
            target = targetDic[filename]
            
            
            data_list.append(data)
            targets.append(target)
    return data_list , targets

targetDic_path = open('targetDic1.pkl','rb')
targetDic = pickle.load(targetDic_path)

# test，test构建
test_file_path = './100data'
test_data_lists , test_targets = load_data(test_file_path)
test_data_list = [Data(x=torch.tensor(torch.cat((data['node_type'], data['num_inverted_predecessors']), dim=0), dtype=torch.float32).unsqueeze(1), 
                  edge_index=data['edge_index']) for data in test_data_lists]
test_target = torch.tensor(test_targets, dtype=torch.float32)
test_target = test_target

test_dataset = CustomDataset(test_data_list, test_target)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 定义模型、损失函数和优化器
num_node_features = 1  # 根据具体情况调整
hidden_channels = 16  # 隐藏层大小，可以根据需要调整
output_dim = 8
model = GNN(num_node_features, hidden_channels, output_dim)
model.load_state_dict(torch.load('./model/best_model.pth'))
model.eval()
with torch.no_grad():
    i = 0
    for data, target in test_loader:
        i += 1
        if i>= 1000:
            break
        data = data
        target = target
        output = model(data)
        predicted_value = output.item()
        target_value = target.item()
        print("Predicted value:", predicted_value, "Target value:", target_value)