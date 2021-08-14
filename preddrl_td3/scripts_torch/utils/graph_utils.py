import dgl
import torch

import numpy as np
from collections import defaultdict




def seq_to_graph(sequence_dict, seq_len=20, interaction_radius=5):
    '''
    seq_pos : masked array of size [N, 2, 20]
    seq_vel : masked array of size [N, 2, 20]
    frame_id : masked array of size [N, 20]
    peds_id : array containing id of N peds [N, ]
    traj_id : array containing unique trajectory number [N, ]
    loss_mask : masked array of size [N, 20] 

    '''

    nodes_data = defaultdict(list)
    edges_data = defaultdict(list)
    nodes_neighors = defaultdict(list)
    N = 0 #total nodes counter
    previous_nodes = None
    for t in range(seq_len):
        current_mask = sequence_dict['msk'][:, t].astype(bool) # indicates nodes present at current time
        current_nodes = [n for n in range(N, N + current_mask.sum())]
        if not current_nodes:
            continue
        # print(current_nodes)
        current_pos = sequence_dict['pos'][:, :, t][current_mask]
        current_rel = sequence_dict['rel'][:, :, t][current_mask]
        current_vel = sequence_dict['vel'][:, :, t][current_mask]
        current_acc = sequence_dict['acc'][:, :, t][current_mask]
        current_hed = sequence_dict['hed'][:, :, t][current_mask]
        current_dir = sequence_dict['dir'][:, :, t][current_mask]
        current_vnorm = sequence_dict['vnorm'][:, :, t][current_mask]
        current_anorm = sequence_dict['anorm'][:, :, t][current_mask]

        current_fid = sequence_dict['fid'][:, t][current_mask]
        
        current_tid = sequence_dict['tid'][current_mask]
        current_cid = sequence_dict['cid'][current_mask]
        current_sid = sequence_dict['sid'][current_mask]
        
        # temporal edges
        unmatched_nodes = copy(current_nodes) # new nodes at current time
        if t>0 and previous_nodes is not None:
            for j, v in enumerate(current_nodes):
                for i, u in enumerate(previous_nodes):
                    if previous_tid[i]!=current_tid[j]:
                        continue

                    dist = norm(current_pos[j], previous_pos[i])
                    diff = current_pos[j] - previous_pos[i] # diff is measured from destination
                    
                    edges_data['src'].extend([u])
                    edges_data['des'].extend([v])

                    edges_data['dist'].extend([dist])
                    edges_data['diff'].extend([diff])

                    edges_data['acc'].extend([previous_acc[i]])
                    edges_data['anorm'].extend([previous_anorm[i]])

                    edges_data['spatial_mask'].extend([0.0])
                    nodes_neighors[v].append(u) # consider logic of using incoming edges for neighbors, u->v means, u is neighbors of v
                    
                    # # bidirectional edges
                    # if random.random()>0.5:
                    #     edges_data['src'].extend([v])
                    #     edges_data['des'].extend([u])
                    #     edges_data['dist'].extend([dist])
                    #     edges_data['diff'].extend([-diff])
                    #     edges_data['spatial_mask'].extend([0.0])
                    #     nodes_neighors[u].append(v)
                        
                    # remove current node
                    unmatched_nodes.remove(v)
                    # only single node from previous step can match to single node at current step, thus break if matched
                    break 
        # make sure that nodes at last time step is not added to the graph as it will not have any temporal edges
        if t==seq_len-1:
            for node in unmatched_nodes:
                current_nodes.remove(node)
        
        # nodes
        for i in range(len(current_nodes)):
            u = current_nodes[i]
            nodes_data['ntx'].append(t)
            nodes_data['pos'].append(current_pos[i])
            nodes_data['rel'].append(current_rel[i])
            nodes_data['vel'].append(current_vel[i])
            nodes_data['acc'].append(current_acc[i])
            nodes_data['hed'].append(current_hed[i])
            nodes_data['dir'].append(current_dir[i])
            nodes_data['vnorm'].append(current_vnorm[i])
            nodes_data['anorm'].append(current_anorm[i])
            
            nodes_data['tid'].append(current_tid[i])
            nodes_data['cid'].append(current_cid[i])
            nodes_data['sid'].append(current_sid[i])
            nodes_data['fid'].append(current_fid[i])
            
            # spatial edges
            for j in range(i+1, len(current_nodes)):
                dist = norm(current_pos[i], current_pos[j]) 
                
                if dist > get_interaction_radius(current_cid[i], current_cid[j], interaction_radius):
                    continue
                
                diff = current_pos[i] - current_pos[j]

                v = current_nodes[j]
                edges_data['src'].extend([u, v])
                edges_data['des'].extend([v, u])
                edges_data['dist'].extend([dist, dist])
                edges_data['diff'].extend([-diff, diff]) # difference measured from destination
                
                # dummy acclerations
                edges_data['acc'].extend([current_acc[i], current_acc[j]])
                edges_data['anorm'].extend([current_anorm[j], current_anorm[i]])

                edges_data['spatial_mask'].extend([1.0, 1.0])
                
                nodes_neighors[u].append(v)
                nodes_neighors[v].append(u)
                            
        previous_nodes = current_nodes
        previous_pos = current_pos
        previous_tid = current_tid
        previous_acc = current_acc
        previous_anorm = current_anorm

        N+=len(current_pos)
        
    # Construct the DGL graph
    g = dgl.graph((edges_data['src'], edges_data['des']))
  
    # Add  features
    g.ndata['ntx'] = torch.tensor(nodes_data['ntx'], dtype=torch.int64)
    g.ndata['tid'] = torch.tensor(nodes_data['tid'], dtype=torch.int64) #unique id for a traj, NOTE! this is different from ped id
    g.ndata['cid'] = torch.tensor(nodes_data['cid'], dtype=torch.int64)
    g.ndata['sid'] = torch.tensor(nodes_data['sid'], dtype=torch.int64)
    g.ndata['fid'] = torch.tensor(nodes_data['fid'], dtype=torch.int64)
    
    g.ndata['pos'] = torch.DoubleTensor(np.stack(nodes_data['pos'], axis=0))
    g.ndata['rel'] = torch.DoubleTensor(np.stack(nodes_data['rel'], axis=0))
    g.ndata['vel'] = torch.DoubleTensor(np.stack(nodes_data['vel'], axis=0))
    g.ndata['acc'] = torch.DoubleTensor(np.stack(nodes_data['acc'], axis=0))
    g.ndata['hed'] = torch.DoubleTensor(np.stack(nodes_data['hed'], axis=0))
    g.ndata['dir'] = torch.DoubleTensor(np.stack(nodes_data['dir'], axis=0))
    g.ndata['vnorm'] = torch.DoubleTensor(np.stack(nodes_data['vnorm'], axis=0))
    g.ndata['anorm'] = torch.DoubleTensor(np.stack(nodes_data['anorm'], axis=0))
    
    g.edata['dist'] = torch.DoubleTensor(np.reshape(edges_data['dist'], (len(edges_data['dist']), -1)))
    g.edata['diff'] = torch.DoubleTensor(np.reshape(edges_data['diff'], (len(edges_data['diff']), -1)))
    g.edata['acc'] = torch.DoubleTensor(np.reshape(edges_data['acc'], (len(edges_data['acc']), -1)))
    g.edata['anorm'] = torch.DoubleTensor(np.reshape(edges_data['anorm'], (len(edges_data['anorm']), -1)))
    g.edata['spatial_mask'] = torch.DoubleTensor(np.expand_dims(edges_data['spatial_mask'], axis=-1))

#%%
    # nodes_neighors_list = [nodes_neighors[n] for n in g.nodes().numpy()]
    # nodes_neighors_list = [neighbors + [-1]*(max_neighbors-len(neighbors)) for neighbors in nodes_neighors_list]
    # g.ndata['neighbors_nodes'] = torch.tensor(nodes_neighors_list, dtype=torch.int64)
    
    return g


def node_sequence(node_pos, dt, pad_front=0, pad_end=20, seq_len=20, rel_idx=0, frames=None):
    
    p, r, v, v_norm, a, a_norm, h, d = motion_kinematics(node_pos, dt, rel_idx)
    
    mask = np.zeros((seq_len,))
    fid = np.zeros((seq_len,))
    
    state_dict = {}
    for s in ['pos', 'rel', 'vel', 'acc', 'hed', 'dir']:
        state_dict[s] = np.zeros((2, seq_len))
        
    for s in ['vnorm', 'anorm']:
        state_dict[s] = np.zeros((1, seq_len))
        
    state_dict['pos'][:, pad_front:pad_end] = p.T
    state_dict['rel'][:, pad_front:pad_end] = r.T
    state_dict['vel'][:, pad_front:pad_end] = v.T
    state_dict['vnorm'][:, pad_front:pad_end] = v_norm.T
    state_dict['acc'][:, pad_front:pad_end] = a.T
    state_dict['anorm'][:, pad_front:pad_end] = a_norm.T
    state_dict['hed'][:, pad_front:pad_end] = h.T
    state_dict['hed'][:, pad_front:pad_end] = d.T

    fid[pad_front:pad_end] = frames
    mask[pad_front:pad_end] = 1
    
    return state_dict, fid, mask
    
def create_graph(nodes):
    # prepare data for the graph
    sequence_dict = defaultdict()
    sequence_dict = defaultdict