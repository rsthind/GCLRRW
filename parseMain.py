import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData

import numpy as np
import networkx as nx 
import cugraph 
import cudf 
import torch 

from scipy.sparse import coo_matrix

def start(args, p, q):
    device = 'cuda:' + args.cuda

    # hyperparameters
    d = 64
    l = 2
    temp = 0.2
    batch_user = 256
    epoch_no = 100
    max_samp = 40
    lambda_1 = 0.2
    lambda_2 = args.lambda2
    dropout = args.dropout
    lr = args.lr
    decay = args.decay
    svd_q = args.q

    #RW hyperparameters
    perc_edges_to_keep = args.perc_edges #0.9
    num_start_nodes = args.start_nodes #5000
    walk_length = args.walk_len #4000
    #p = args.p
    #q = args.q_val

    # load data
    path = 'data/' + args.data + '/'
    #path = '/'
    f = open(path+'trnMat.pkl','rb')
    train = pickle.load(f)
    train_csr = (train!=0).astype(np.float32)
    f = open(path+'tstMat.pkl','rb')
    test = pickle.load(f)
    print('Data loaded.')

    print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)

    epoch_user = min(train.shape[0], 30000)

    num_users = train.shape[0]
    all_edges = train.getnnz()
    G = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(train)
    adj_matrix = nx.adjacency_matrix(G).tocoo()
    G = cugraph.Graph()
    G.from_cudf_edgelist(cudf.DataFrame({'src':adj_matrix.row, 'dst':adj_matrix.col}), source='src', destination='dst', renumber=False, legacy_renum_only=True)

    start = np.random.randint(0, adj_matrix.shape[0], (num_start_nodes,)).tolist() #nodes to start from
    vertex_paths = cugraph.node2vec(G, start, walk_length, compress_result=False, p=p, q=q)[0] #running rw

    #splitting path into edges
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
    # counts = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).cuda(), train.shape).coalesce()
    # print(torch.sort(counts.values(), descending=True)[0][999])
    values = torch.bernoulli(torch.clamp(ret.values(), max=1)).bool()
    # print(values.size(0))
    ret_ind = ret.indices()[:, values]
    values = values[values].type(torch.float32)
    # print(values.size(0))

    ret_ind = ret_ind.cpu().numpy()
    val_np = values.cpu().numpy()
    # normalizing the ret adj matrix
    ret = coo_matrix((val_np, ret_ind), shape=train.shape)
    rowD = np.array(ret.sum(1)).squeeze()
    colD = np.array(ret.sum(0)).squeeze()
    for i in range(len(ret.data)):
        ret.data[i] = ret.data[i] / pow(rowD[ret.row[i]]*colD[ret.col[i]], 0.5)


    # normalizing the adj matrix
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

    # construct data loader
    train = train.tocoo()
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    adj_norm = adj_norm.coalesce().cuda(torch.device(device))

    ret = ret.tocoo()
    ret_adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(ret)
    ret_adj_norm = ret_adj_norm.coalesce().cuda(torch.device(device))
    print('Adj matrix normalized.')

    # process test set
    test_labels = [[] for i in range(test.shape[0])]
    for i in range(len(test.data)):
        row = test.row[i]
        col = test.col[i]
        test_labels[row].append(col)
    print('Test data processed.')

    loss_list = []
    loss_r_list = []
    loss_s_list = []
    recall_20_x = []
    recall_20_y = []
    ndcg_20_y = []
    recall_40_y = []
    ndcg_40_y = []

    model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device, ret_adj_norm)
    #model.load_state_dict(torch.load('saved_model.pt'))
    model.cuda(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
    #optimizer.load_state_dict(torch.load('saved_optim.pt'))

    current_lr = lr

    for epoch in range(epoch_no):
        # if (epoch+1)%50 == 0:
        #     torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
        #     torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')

        epoch_loss = 0
        epoch_loss_r = 0
        epoch_loss_s = 0
        train_loader.dataset.neg_sampling()
        for i, batch in enumerate(tqdm(train_loader)):
            uids, pos, neg = batch
            uids = uids.long().cuda(torch.device(device))
            pos = pos.long().cuda(torch.device(device))
            neg = neg.long().cuda(torch.device(device))
            iids = torch.concat([pos, neg], dim=0)

            # feed
            optimizer.zero_grad()
            loss, loss_r, loss_s= model(uids, iids, pos, neg)
            loss.backward()
            optimizer.step()
            #print('batch',batch)
            epoch_loss += loss.cpu().item()
            epoch_loss_r += loss_r.cpu().item()
            epoch_loss_s += loss_s.cpu().item()

            torch.cuda.empty_cache()
            #print(i, len(train_loader), end='\r')

        batch_no = len(train_loader)
        epoch_loss = epoch_loss/batch_no
        epoch_loss_r = epoch_loss_r/batch_no
        epoch_loss_s = epoch_loss_s/batch_no
        loss_list.append(epoch_loss)
        loss_r_list.append(epoch_loss_r)
        loss_s_list.append(epoch_loss_s)
        print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

        if epoch % 3 == 0:  # test every 10 epochs
            test_uids = np.array([i for i in range(adj_norm.shape[0])])
            batch_no = int(np.ceil(len(test_uids)/batch_user))

            all_recall_20 = 0
            all_ndcg_20 = 0
            all_recall_40 = 0
            all_ndcg_40 = 0
            for batch in tqdm(range(batch_no)):
                start = batch*batch_user
                end = min((batch+1)*batch_user,len(test_uids))

                test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
                predictions = model(test_uids_input,None,None,None,test=True)
                predictions = np.array(predictions.cpu())

                #top@20
                recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
                #top@40
                recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

                all_recall_20+=recall_20
                all_ndcg_20+=ndcg_20
                all_recall_40+=recall_40
                all_ndcg_40+=ndcg_40
                #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
            print('-------------------------------------------')
            print('Test of epoch',epoch,':','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)
            recall_20_x.append(epoch)
            recall_20_y.append(all_recall_20/batch_no)
            ndcg_20_y.append(all_ndcg_20/batch_no)
            recall_40_y.append(all_recall_40/batch_no)
            ndcg_40_y.append(all_ndcg_40/batch_no)

    # final test
    test_uids = np.array([i for i in range(adj_norm.shape[0])])
    batch_no = int(np.ceil(len(test_uids)/batch_user))

    all_recall_20 = 0
    all_ndcg_20 = 0
    all_recall_40 = 0
    all_ndcg_40 = 0
    for batch in range(batch_no):
        start = batch*batch_user
        end = min((batch+1)*batch_user,len(test_uids))

        test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
        predictions = model(test_uids_input,None,None,None,test=True)
        predictions = np.array(predictions.cpu())

        #top@20
        recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
        #top@40
        recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

        all_recall_20+=recall_20
        all_ndcg_20+=ndcg_20
        all_recall_40+=recall_40
        all_ndcg_40+=ndcg_40
        #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
    print('-------------------------------------------')
    print('Final test:','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)

    recall_20_x.append('Final')
    recall_20_y.append(all_recall_20/batch_no)
    ndcg_20_y.append(all_ndcg_20/batch_no)
    recall_40_y.append(all_recall_40/batch_no)
    ndcg_40_y.append(all_ndcg_40/batch_no)

    metric = pd.DataFrame({
        'epoch':recall_20_x,
        'recall@20':recall_20_y,
        'ndcg@20':ndcg_20_y,
        'recall@40':recall_40_y,
        'ndcg@40':ndcg_40_y
    })
    current_t = time.gmtime()
    metric.to_csv('log/result_'+ 'p_' + str(p) + '_q_' + str(q) + '_' + args.data +'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.csv')

    #torch.save(model.state_dict(),'saved_model/saved_model_'+ args.data +'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')
    #torch.save(optimizer.state_dict(),'saved_model/saved_optim_'+ args.data +'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')
