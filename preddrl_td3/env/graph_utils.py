import torch

import numpy as np
from collections import defaultdict

def seq_to_graph(seq, frames, ped_ids, norm_lap_matr=True):
    '''
    seq: list of states (N, T, state_dim)
    '''
    num_nodes = seq.shape[0]
    seq_len = seq.shape[1]
    
    nodes_dist = np.zeros((seq_len, num_nodes, num_nodes))#[T, num_nodes, num_nodes]
    for t in range(seq_len):
        nodes_coord = seq[:, :, t]
        # Compute distance matrix
        for h in range(len(nodes_coord)): 
            nodes_dist[t, h, h] = 1
            
            for k in range(h+1, len(nodes_coord)):
                
                l2_norm = anorm(nodes_coord[h], nodes_coord[k])
                
                nodes_dist[t, h, k] = l2_norm
                nodes_dist[t, k, h] = l2_norm
                 
        if norm_lap_matr:
            G = nx.from_numpy_matrix(nodes_dist[t, :, :])
            nodes_dist[t, :, :] = nx.normalized_laplacian_matrix(G).toarray()
    
    # Construct the DGL graph
    # g = dgl.DGLGraph()
    # Construct the DGL graph
    g = dgl.graph((edges_data['src'], edges_data['des']))
    g.add_nodes(num_nodes)        
    
    edge_feats = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i!=j:
                g.add_edge(i, j)
                edge_feats.append(nodes_dist[:, i, j])
    
    assert len(edge_feats) == g.number_of_edges()
    
    # Add edge features
    g.edata['dist'] = torch.DoubleTensor(edge_feats)
    g.ndata['pos'] = torch.DoubleTensor(seq)
    g.ndata['vel'] = torch.DoubleTensor(seq_rel)
    g.ndata['frames'] = torch.DoubleTensor(frames)
    g.ndata['pid'] = torch.DoubleTensor(ped_ids)
    
    return g