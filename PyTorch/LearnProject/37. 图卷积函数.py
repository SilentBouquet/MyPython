import matplotlib
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

# 构建图数据
G = nx.Graph()
blue, orange, green = "#1f77b4", "#ff7f0e", "#2ca02c"
# 使用 add_nodes_from 方法向图 G 中添加节点。
# 每个节点是一个元组，格式为 (节点ID, 节点属性字典)
G.add_nodes_from([
    (1, {"color": blue}),
    (2, {"color": orange}),
    (3, {"color": blue}),
    (4, {"color": green}),
])
# 使用 add_edges_from 方法向图 G 中添加边
G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4)])
# nx.adjacency_matrix(G) 返回一个稀疏矩阵表示的邻接矩阵
# todense() 将稀疏矩阵转换为密集矩阵（NumPy 数组）
A = np.asarray(nx.adjacency_matrix(G).todense())
print(A)


def build_graph_color_label_representation(G, mapping_dict):
    # 获取图中每个节点的颜色，并通过映射字典将其转换为对应的整数索引
    one_hot_idxs = np.array([mapping_dict[v] for v in
                             nx.get_node_attributes(G, 'color').values()])
    one_hot_encoding = np.zeros((one_hot_idxs.size, len(mapping_dict)))
    one_hot_encoding[np.arange(one_hot_idxs.size), one_hot_idxs] = 1
    return one_hot_encoding


# 将图的节点颜色转换为独热编码矩阵
X = build_graph_color_label_representation(G, {green: 0, blue: 1, orange: 2})
print(X)
color_map = nx.get_node_attributes(G, 'color').values()
matplotlib.use('Qt5Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color=color_map, node_size=700)
plt.show()

# 前向传播
f_in, f_out = X.shape[1], 6
W1 = np.random.rand(f_in, f_out)
W2 = np.random.rand(f_in, f_out)
h = np.dot(X, W1) + np.dot(np.dot(A, X), W2)
print(h)