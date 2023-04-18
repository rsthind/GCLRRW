import dgl 
import pickle as pk
import numpy as np
import torch 
import random 
#HYPERPARAMETERS TO TUNE
perc_edges_to_keep = 0.5
num_start_nodes = 500
restart_prob = 0.001
walk_length = 100
f = open('trnMat.pkl','rb')
train = pk.load(f).astype(np.int32)
all_edges = train.getnnz()

row_tensor = torch.tensor(train.row).cuda()
col_tensor = torch.tensor(train.col).cuda()
data_dict = {
    ('user', 'at', 'location'): (row_tensor, col_tensor)
}
g = dgl.heterograph(data_dict)
transform = dgl.AddReverse()
g = transform(g)
g = g.to('cuda')
num_start_users = int(num_start_nodes * random.random())
users = torch.randint(0, g.number_of_nodes('user'), (num_start_users,), dtype=torch.int32).cuda()
items = torch.randint(0, g.number_of_nodes('location'), (num_start_nodes - num_start_users,), dtype=torch.int32).cuda()
#run the rw
random_walk_users, eids_users, _ = dgl.sampling.random_walk(g, users, metapath=['at', 'rev_at'] * int(walk_length / 2), return_eids=True, restart_prob = restart_prob)
random_walk_items, eids_items, _ = dgl.sampling.random_walk(g, items, metapath=['rev_at', 'at'] * int(walk_length / 2), return_eids=True, restart_prob = restart_prob)

eids = torch.cat([eids_users.flatten(), eids_items.flatten()])
eids = eids[eids >= 0]
total_edges = eids.shape[0]
edge_index = torch.stack(([row_tensor[eids], col_tensor[eids]]), dim=0)
ret = torch.sparse_coo_tensor(edge_index, (torch.ones(edge_index.shape[1]) * (all_edges * perc_edges_to_keep / total_edges)).cuda(), train.shape).coalesce()
print(torch.count_nonzero(ret.values()))
values = torch.bernoulli(torch.clamp(ret.values(), max=1))
print(torch.count_nonzero(values))
ret = torch.sparse_coo_tensor(ret.indices(), values, train.shape) #new adj matrix




