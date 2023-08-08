import torch
from torch_geometric.utils import to_dense_adj

# 示例的稀疏邻接矩阵，使用COO格式表示
edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]], dtype=torch.long)
num_nodes = 5

# 将稀疏邻接矩阵转换为密集邻接矩阵
dense_adj = to_dense_adj(edge_index)[0]

print("稀疏邻接矩阵：")
print(edge_index)

print("密集邻接矩阵：")
print(dense_adj)
