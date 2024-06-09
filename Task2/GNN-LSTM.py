import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import pickle
import numpy as np
import torch.nn as nn
from torch.nn import BatchNorm1d as BatchNorm

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_list, targets, operation_sequences, max_operations):
        self.data_list = data_list
        self.targets = targets
        self.operation_sequences = operation_sequences
        self.max_operations = max_operations

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        target = self.targets[idx]
        operations = self.operation_sequences[idx]
        # Pad operations to max length
        padded_operations = torch.full((self.max_operations,), 0, dtype=torch.long)
        padded_operations[:len(operations)] = torch.tensor(operations, dtype=torch.long)
        return data, target, padded_operations

# 定义模型
class GCN_LSTM_Model(nn.Module):
    def __init__(self, num_node_features, hidden_channels, operation_embedding_dim, lstm_hidden_dim, num_operations, max_operations):
        super(GCN_LSTM_Model, self).__init__()
        
        # 图神经网络部分
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        
        # LSTM部分
        self.embedding = nn.Embedding(num_operations + 1, operation_embedding_dim)  # +1 for padding
        self.lstm = nn.LSTM(operation_embedding_dim, lstm_hidden_dim, batch_first=True)
        
        # 全连接层
        self.fc1 = torch.nn.Linear(hidden_channels + lstm_hidden_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer

    def forward(self, data, operations):
        # 图神经网络部分
        x, edge_index = data.x, data.edge_index
        x_ori = x.clone()
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.gelu(x)
        
        x = x_ori + x
        x = torch.mean(x, dim=0, keepdim=True)  # 保持批次维度
        
        # LSTM部分
        embedded_operations = self.embedding(operations)
        lstm_out, _ = self.lstm(embedded_operations)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 结合图表示和操作序列表示
        x = x.expand(lstm_out.size(0), -1)  # 扩展x的批次维度
        combined = torch.cat([x, lstm_out], dim=-1)
        
        # 全连接层
        x = self.fc1(combined)
        x = F.gelu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        out = self.fc4(x)
        
        return out.squeeze()

# 加载数据的函数
def load_data(directory, targetDic):
    data_list = []
    targets = []
    operations = []
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 构建文件的完整路径
        filepath = os.path.join(directory, filename)
        parts = filename.split('_')  
        number_part = list(map(int, parts[1].split('.')[0]))  # 将操作序列转换为整数列表
        # 检查路径是否为文件
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
                target = targetDic[filename]
                data_list.append(data)
                targets.append(target)
                operations.append(number_part)
    return data_list, targets, operations

# target字典读入
targetDic_path = open('targetDic1.pkl', 'rb')
targetDic = pickle.load(targetDic_path)

# 加载训练和测试数据
train_file_path = './500data'
test_file_path = './100data'
train_data_list, train_targets, train_operations = load_data(train_file_path, targetDic)
test_data_list, test_targets, test_operations = load_data(test_file_path, targetDic)

max_operations = 10

train_data_list = [Data(x=torch.tensor(data['node_type'], dtype=torch.float32).unsqueeze(1), 
                        edge_index=torch.tensor(data['edge_index'], dtype=torch.long)) for data in train_data_list]

test_data_list = [Data(x=torch.tensor(data['node_type'], dtype=torch.float32).unsqueeze(1), 
                       edge_index=torch.tensor(data['edge_index'], dtype=torch.long)) for data in test_data_list]

# 将目标转换为tensor
train_targets = torch.tensor(train_targets, dtype=torch.float32)
test_targets = torch.tensor(test_targets, dtype=torch.float32)

# 创建自定义数据集和数据加载器
train_dataset = CustomDataset(train_data_list, train_targets, train_operations, max_operations)
test_dataset = CustomDataset(test_data_list, test_targets, test_operations, max_operations)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型、损失函数和优化器
num_node_features = 1  # 节点特征维度
hidden_channels = 16  # 隐藏层大小
operation_embedding_dim = 8  # 操作嵌入维度
lstm_hidden_dim = 16  # LSTM隐藏层维度
num_operations = 7  # 操作种类数（0到6）

model = GCN_LSTM_Model(num_node_features, hidden_channels, operation_embedding_dim, lstm_hidden_dim, num_operations, max_operations)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train(loader):
    model.train()
    total_loss = 0
    for data, target, operations in loader:
        optimizer.zero_grad()
        out = model(data, operations)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target, operations in loader:
            out = model(data, operations)
            loss = criterion(out, target)
            total_loss += loss.item()
    return total_loss / len(loader)

# 训练和测试模型
num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train(train_loader)
    test_loss = test(test_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
