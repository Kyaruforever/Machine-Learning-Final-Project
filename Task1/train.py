import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import pickle
import torch.nn as nn
import os
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv,BatchNorm
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# class GCN(torch.nn.Module):
#     def __init__(self, num_node_features, hidden_channels, output_dim):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, output_dim)
#         self.fc = nn.Linear(output_dim, 1)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)
#         x = torch.mean(x, dim = 0)
#         x = self.fc(x)
#         return x # 将输出张量降维，以适应回归任务

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


# data_file = open("example.pkl",'rb')
# for i in range(500):
#     data_lists.append(pickle.load(data_file))
    
# targets = np.zeros(500)

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
            if target <=1e-6:
                continue
            
            data_list.append(data)
            targets.append(target)
    return data_list , targets

# target字典读入
targetDic_path = open('targetDic1.pkl','rb')
targetDic = pickle.load(targetDic_path)

# train，test构建
train_file_path = './500data'
# test_file_path = './100data'
# train_data_lists , train_targets = load_data(train_file_path)
# test_data_lists , test_targets = load_data(test_file_path)

all_data_lists , all_targets = load_data(train_file_path)
train_data_lists, test_data_lists, train_targets, test_targets = train_test_split(all_data_lists, all_targets, test_size=0.1, random_state=42)
# train_targets, test_targets = train_test_split(all_targets, test_size=0.1, random_state=42)


# 假设 data_list 是一个包含多个 Data 对象的列表，每个对象包含节点特征、边索引等
train_data_list = [Data(x=torch.tensor(torch.cat((data['node_type'], data['num_inverted_predecessors']), dim=0), dtype=torch.float32).unsqueeze(1), 
                  edge_index=data['edge_index']) for data in train_data_lists]
train_target = torch.tensor(train_targets, dtype=torch.float32)
train_data_list = [data.to(device) for data in train_data_list]
train_target = train_target.to(device)

# 创建自定义数据集和数据加载器
dataset = CustomDataset(train_data_list, train_target)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
# 定义模型、损失函数和优化器
num_node_features = 1  # 根据具体情况调整
hidden_channels = 16  # 隐藏层大小，可以根据需要调整
output_dim = 8
model = GNN(num_node_features, hidden_channels, output_dim)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss().to(device)

# 训练模型
def train(loader,  epochs):
    model.train()
    total_loss = 0
    loop = tqdm(loader,leave = False, total = len(loader), desc=f'Epoch [{epoch}/{epochs - 1}]')
    for data, target in loop:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        out = model(data)
        # print(out.shape)
        # print(target.shape)
        # target_tenser = torch.full(out.size(), target)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 训练多个 epoch
num_epochs = 3

for epoch in range(num_epochs):
    loss = train(loader, num_epochs)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# torch.save(model.state_dict(), 'model_weights.pth')

# Test
test_data_list = [Data(x=torch.tensor(torch.cat((data['node_type'], data['num_inverted_predecessors']), dim=0), dtype=torch.float32).unsqueeze(1), 
                  edge_index=data['edge_index']) for data in test_data_lists]
# 假设 targets 是对应的目标标签
test_target = torch.tensor(test_targets, dtype=torch.float32)

# 创建自定义数据集和数据加载器
test_dataset = CustomDataset(test_data_list, test_target)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
model.eval()
with torch.no_grad():
    i = 0
    for data, target in test_loader:
        i += 1
        if i>= 1000:
            break
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        predicted_value = output.item()
        target_value = target.item()
        print("Predicted value:", predicted_value, "Target value:", target_value)