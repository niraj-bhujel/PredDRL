import dgl
import torch
from copy import copy

import numpy as np
from collections import defaultdict

from .vis_graph import network_draw

node_type_list = ['robot', 'pedestrian', 'obstacle', 'goal']

INTERACTION_RADIUS = {
    ('robot', 'robot'): 0.0,
    ('robot', 'pedestrian'): 5.0,
    ('robot', 'obstacle'): 5.0,
    ('robot', 'goal'): 20.0,

    ('pedestrian', 'pedestrian'): 5.0,
    ('pedestrian', 'obstacle') : 5.0,
    ('pedestrian', 'goal'): 20.0,

    ('obstacle', 'obstacle'): 0.00,
    ('obstacle', 'goal'): 0.0,
    
}


def create_graph(nodes, interaction_radius=5):
    '''
        Create a graphs with node representing a pedestrians/robot/obstacle.
    '''

    nodes_data = defaultdict(list)
    edges_data = defaultdict(list)
    # N = len(nodes)
    for i in range(len(nodes)):

        src_node = nodes[i]

        # print(node_type_list.index(src_node._type), src_node._type)
        # src_node_states = np.concatenate(src_node.states_at(src_node.last_timestep), axis=-1)

        nodes_data['pos'].append(src_node._pos[src_node.last_timestep])
        nodes_data['vel'].append(src_node._vel[src_node.last_timestep])
        nodes_data['acc'].append(src_node._acc[src_node.last_timestep])
        nodes_data['hed'].append(src_node.heading(src_node.last_timestep))
        # nodes_data['goal'].append(src_node._goal)
        nodes_data['dir'].append(src_node.distance_to_goal(src_node.last_timestep))
        
        nodes_data['tid'].append(src_node._id)
        nodes_data['cid'].append(node_type_list.index(src_node._type))
        
        # spatial edges
        for j in range(i+1, len(nodes)):

            dst_node = nodes[j]
            
            diff = np.array(src_node._pos[src_node.last_timestep]) - np.array(dst_node._pos[dst_node.last_timestep])

            dist = np.linalg.norm(diff, keepdims=True)

            try:
                rad = INTERACTION_RADIUS[src_node._type, dst_node._type]
            except:
                rad = INTERACTION_RADIUS[dst_node._type, src_node._type]

            if dist > rad:
                continue
            
            # bidirectional edges
            edges_data['src'].extend([i, j])
            edges_data['des'].extend([j, i])
            edges_data['dist'].extend([dist, dist])
            edges_data['diff'].extend([-diff, diff]) # difference measured from destination

            edges_data['spatial_mask'].extend([1.0, 1.0])

    # Construct the DGL graph
    g = dgl.graph((edges_data['src'], edges_data['des']), num_nodes=len(nodes))
  
    # Add  features
    g.ndata['tid'] = torch.tensor(nodes_data['tid'], dtype=torch.int32)
    g.ndata['cid'] = torch.tensor(nodes_data['cid'], dtype=torch.int32)
    
    g.ndata['pos'] = torch.tensor(np.stack(nodes_data['pos'], axis=0), dtype=torch.float32)
    g.ndata['vel'] = torch.tensor(np.stack(nodes_data['vel'], axis=0), dtype=torch.float32)
    g.ndata['acc'] = torch.tensor(np.stack(nodes_data['acc'], axis=0), dtype=torch.float32)
    g.ndata['hed'] = torch.tensor(np.stack(nodes_data['hed'], axis=0), dtype=torch.float32).unsqueeze(1)
    # g.ndata['goal'] = torch.tensor(np.stack(nodes_data['goal'], axis=0))
    g.ndata['dir'] = torch.tensor(np.stack(nodes_data['dir'], axis=0), dtype=torch.float32).unsqueeze(1)
    
    g.edata['dist'] = torch.tensor(np.stack(edges_data['dist'], axis=0), dtype=torch.float32)
    g.edata['diff'] = torch.tensor(np.stack(edges_data['diff'], axis=0), dtype=torch.float32)
    g.edata['spatial_mask'] = torch.tensor(np.reshape(edges_data['spatial_mask'], (-1, 1)), dtype=torch.int32)
    
    return g

def create_st_graph(nodes, seq_len=2, interaction_radius=5):
    '''
    nodes: list of node object.
        The algorithm assume that number of nodes are fixed over seq_len.
        i.e. if there are 2 nodes at t=0, it will have 2 nodes at t = seq_len-1 as well.
        It creates nodes at each time step. 
        
        !!WORK on PROGRESS
    '''

    nodes_data = defaultdict(list)
    edges_data = defaultdict(list)

    N = 0 #total nodes counter for the graph

    for t in range(seq_len):
        # get the last pos
        # current_nodes = [n for n in range(N, N + len(nodes))]

        # temporal edges -> from previous node to current nodes
        if t>0 and previous_nodes is not None:
            for j, v in enumerate(current_nodes):
                for i, u in enumerate(previous_nodes):
                    if previous_tid[i]!=current_tid[j]:
                        continue

                    dist = np.linalg.norm(current_pos[j] - previous_pos[i])
                    diff = current_pos[j] - previous_pos[i] # diff is measured from destination
                    
                    edges_data['src'].extend([u])
                    edges_data['des'].extend([v])

                    edges_data['dist'].extend([dist])
                    edges_data['diff'].extend([diff])

                    edges_data['spatial_mask'].extend([0.0])
                        
                    # only single node from previous step can match to single node at current step, thus break if matched
                    break

        # spatial nodes
        for i in range(len(nodes)):

            # src_node = nodes[i]
            nodes_data['states'] = np.concatenate(nodes[i].states_at(-1), axis=-1) # (13, )
            
            nodes_data['ntx'].append(t)
            nodes_data['tid'].append(nodes[i]._id)
            nodes_data['cid'].append(node_type_list.index(nodes[i]._type))
            
            # spatial edges
            for j in range(i+1, len(nodes)):

                # use last pos
                if t==0:
                    diff = nodes[i]._pos[-1] - nodes[j]._pos[-1]
                # use predicted pos
                else:
                    diff = nodes[i]._preds[t-1] - nodes[j]._preds[t-1]

                dist = np.linalg.norm(diff) 
                if dist > interaction_radius:
                    continue

                u = current_nodes[i]
                v = current_nodes[j]
                
                edges_data['src'].extend([u, v])
                edges_data['des'].extend([v, u])
                edges_data['dist'].extend([dist, dist])
                edges_data['diff'].extend([-diff, diff]) # difference measured from destination
    
                edges_data['spatial_mask'].extend([1.0, 1.0])

        previous_nodes = current_nodes
        N+=len(current_nodes)
        