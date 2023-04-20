import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--inter_batch', default=4096, type=int, help='batch size')
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding size')
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')

    parser.add_argument('--perc_edges', default=0.5, type=float, help='perc_edges_to_keep')
    parser.add_argument('--start_nodes', default=500, type=int, help='num_start_nodes')
    parser.add_argument('--walk_len', default=100, type=int, help='walk_length')
    parser.add_argument('--p', default=1.0, type=float, help='p for random walk')
    parser.add_argument('--q_val', default=1.0, type=float, help='q for random walk')
    parser.add_argument('--restart', default=0.001, type=float, help='random walk restart probability random walk')
    return parser.parse_args()
args = parse_args()

#python3 main.py --data gowalla --lambda1 1e-7 --lambda2 1e-5 --temp 0.3 --dropout 0 --q 5
#ssh -X rthind3@coc-ice.pace.gatech.edu