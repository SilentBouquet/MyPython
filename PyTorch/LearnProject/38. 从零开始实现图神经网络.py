import matplotlib
import torch
import math
import numpy as np
import networkx as nx
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader


# 全局池化层
def global_sum_pool(X, batch_mat):
    if batch_mat is None or batch_mat.dim() == 1:
        return torch.sum(X, dim=0).unsqueeze(0)
    else:
        return torch.mm(batch_mat, X)


def build_graph_color_label_representation(G, mapping_dict):
    # 获取图中每个节点的颜色，并通过映射字典将其转换为对应的整数索引
    one_hot_idxs = np.array([mapping_dict[v] for v in
                             nx.get_node_attributes(G, 'color').values()])
    one_hot_encoding = np.zeros((one_hot_idxs.size, len(mapping_dict)))
    one_hot_encoding[np.arange(one_hot_idxs.size), one_hot_idxs] = 1
    return one_hot_encoding


# 生成掩码矩阵
def get_batch_tensor(graph_sizes):
    starts = [sum(graph_sizes[:i]) for i in range(len(graph_sizes))]
    stops = [starts[i] + graph_sizes[i] for i in range(len(graph_sizes))]
    total_len = sum(graph_sizes)
    batch_size = len(graph_sizes)
    batch_mat = torch.zeros(batch_size, total_len).float()
    for idx, starts_and_stops in enumerate(zip(starts, stops)):
        starts = starts_and_stops[0]
        stops = starts_and_stops[1]
        batch_mat[idx, starts:stops] = 1
    return batch_mat


# 将一个批次中的多个图数据整合到一起，生成批量化后的邻接矩阵、特征矩阵、标签和掩码矩阵
def collate_graphs(batch):
    adj_mats = [graph['A'] for graph in batch]
    sizes = [A.size(0) for A in adj_mats]
    total_size = sum(sizes)
    batch_mat = get_batch_tensor(sizes)
    # 将每个图的特征矩阵 X 按行拼接，生成批量化后的特征矩阵
    feat_mats = torch.cat([graph['X'] for graph in batch], dim=0)
    labels = torch.cat([graph['y'] for graph in batch], dim=0)
    batch_adj = torch.zeros(total_size, total_size, dtype=torch.float32)
    accum = 0
    # 填充批量化邻接矩阵
    for adj in adj_mats:
        g_size = adj.shape[0]
        batch_adj[accum:accum + g_size, accum:accum + g_size] = adj
        accum += g_size
    repr_and_label = {
        'A': batch_adj,
        'X': feat_mats,
        'y': labels,
        'batch': batch_mat
    }
    return repr_and_label


# 图数据字典化
def get_graph_dict(G, mapping_dict):
    A = torch.from_numpy(np.asarray(nx.adjacency_matrix(G).todense())).float()
    X = torch.from_numpy(build_graph_color_label_representation(G, mapping_dict)).float()
    y = torch.tensor([[1, 0]]).float()
    return {'A': A, 'X': X, 'y': y, 'batch': None}


# 图卷积层编码
class BasicGraphConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W2 = Parameter(torch.rand((in_channels, out_channels), dtype=torch.float32))
        self.W1 = Parameter(torch.rand((in_channels, out_channels), dtype=torch.float32))
        self.bias = Parameter(torch.zeros(out_channels, dtype=torch.float32))

    def forward(self, X, A):
        potential_msgs = torch.mm(X, self.W2)
        propagated_msgs = torch.mm(A, potential_msgs)
        root_update = torch.mm(X, self.W1)
        output = propagated_msgs + root_update + self.bias
        return output


# 定义图神经网络结构
class NodeNetwork(torch.nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.conv1 = BasicGraphConvolutionLayer(input_features, 32)
        self.conv2 = BasicGraphConvolutionLayer(32, 32)
        self.fc1 = torch.nn.Linear(32, 16)
        self.out_layer = torch.nn.Linear(16, 2)

    def forward(self, X, A, batch_mat):
        x = F.relu(self.conv1(X, A))
        x = F.relu(self.conv2(x, A))
        output = global_sum_pool(x, batch_mat)
        output = self.fc1(output)
        output = self.out_layer(output)
        return F.softmax(output, dim=1)


# 构建图数据
blue, orange, green = "#1f77b4", "#ff7f0e", "#2ca02c"
mapping_dict = {green: 0, blue: 1, orange: 2}
G1 = nx.DiGraph()
G1.add_nodes_from([
    (1, {"color": blue}),
    (2, {"color": orange}),
    (3, {"color": blue}),
    (4, {"color": green}),
])
G1.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4)])
G2 = nx.DiGraph()
G2.add_nodes_from([
    (1, {"color": green}),
    (2, {"color": green}),
    (3, {"color": orange}),
    (4, {"color": orange}),
    (5, {"color": blue}),
])
G2.add_edges_from([(2, 3), (3, 4), (3, 1), (5, 1)])
G3 = nx.DiGraph()
G3.add_nodes_from([
    (1, {"color": orange}),
    (2, {"color": orange}),
    (3, {"color": green}),
    (4, {"color": green}),
    (5, {"color": blue}),
    (6, {"color": orange}),
])
G3.add_edges_from([(2, 3), (3, 4), (3, 1), (5, 1), (2, 5), (6, 1)])
G4 = nx.DiGraph()
G4.add_nodes_from([
    (1, {"color": blue}),
    (2, {"color": blue}),
    (3, {"color": green}),
])
G4.add_edges_from([(1, 2), (2, 3)])
graph_list = [get_graph_dict(graph, mapping_dict) for graph in [G1, G2, G3, G4]]
matplotlib.use('Qt5Agg')
plt.figure(figsize=(14, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
for i, G in enumerate([G1, G2, G3, G4]):
    plt.subplot(2, 2, i + 1)
    color_map = nx.get_node_attributes(G, 'color').values()
    nx.draw(G, with_labels=True, node_color=color_map, node_size=700)
    plt.title(f'G{i+1}')
plt.show()


# 自定义数据集
class ExampleDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        mol_rep = graph_list[idx]
        return mol_rep


dataset = ExampleDataset(graph_list)
# collate_fn 将从数据集中获取的多个样本数据整合成一个批次，以便于批量训练
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_graphs)

# 使用NodeNetwork进行预测
node_features = 3
net = NodeNetwork(node_features)
batch_results = []
for b in dataloader:
    batch_results.append(net(b['X'], b['A'], b['batch']).detach())
print(batch_results)

# 对比单个图和批量处理的预测结果
G1_rep = dataset[1]
G1_single = net(G1_rep['X'], G1_rep['A'], G1_rep['batch']).detach()
G1_batch = batch_results[0][1]
print(torch.all(torch.isclose(G1_single, G1_batch)))