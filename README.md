# GCLRRW

### 1. Running environment

We develope our codes in the following environment:

```
Python version 3.9.12
cugraph
cudf
dgl
torch
networkx
numpy
tqdm
```

### 2. How to run the codes

* Gowalla

```
python main.py
```

### 3. Some configurable arguments

* `--cuda` specifies which GPU to run on if there are more than one.
* `--data` selects the dataset to use.
* `--lambda1` specifies $\lambda_1$, the regularization weight for CL loss.
* `--lambda2` is $\lambda_2$, the L2 regularization weight.
* `--temp` specifies $\tau$, the temperature in CL loss.
* `--dropout` is the edge dropout rate.
* `--q` decides the rank q for SVD.
* `--p` p for random walk.
* `--q_val` q for random walk.
* `--restart` random walk restart probability.
* `--perc_edges` percent of edges to keep for random walk.
* `--start_nodes` number of start nodes for random walk.
* `--walk_len` the length of the walk.