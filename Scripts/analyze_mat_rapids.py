import numpy as np
import networkx as nx 
import pickle as pk
import cugraph 
import cudf 
import time 
import torch 
#HYPERPARAMETERS TO TUNE
perc_edges_to_keep = 0.5 
num_start_nodes = 500
walk_length = 100
p = 0.2 
q = 0.5 
f = open('trnMat.pkl','rb')
train = pk.load(f).astype(np.int32)
num_users = train.shape[0]
all_edges = train.getnnz()
G = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(train)
adj_matrix = nx.adjacency_matrix(G).tocoo()
G = cugraph.Graph()
G.from_cudf_edgelist(cudf.DataFrame({'src':adj_matrix.row, 'dst':adj_matrix.col}), source='src', destination='dst', renumber=False, legacy_renum_only=True)
start = np.random.randint(0, adj_matrix.shape[0], (num_start_nodes,)).tolist()
vertex_paths = cugraph.node2vec(G, start, walk_length, compress_result=False, p=p, q=q)[0]
vertex_paths = torch.from_numpy(vertex_paths.to_numpy()).long().cuda()
vertex_paths = vertex_paths.reshape(-1, walk_length)
src = vertex_paths[:, :-1].flatten()
total_edges = src.shape[0]
dst = vertex_paths[:, 1:].flatten()
edge_index = torch.stack([src, dst], dim=0)
edge_index2 = torch.stack([dst, src], dim=0)
edge_index = torch.cat([edge_index, edge_index2], dim=1)
edge_index = edge_index[:, edge_index[0] < num_users]
edge_index[1] = edge_index[1] - num_users
ret = torch.sparse_coo_tensor(edge_index, (torch.ones(edge_index.shape[1]) * (all_edges * perc_edges_to_keep / total_edges)).cuda(), train.shape).coalesce()
print(torch.count_nonzero(ret.values()))
values = torch.bernoulli(torch.clamp(ret.values(), max=1))
print(torch.count_nonzero(values))
ret = torch.sparse_coo_tensor(ret.indices(), values, train.shape)



