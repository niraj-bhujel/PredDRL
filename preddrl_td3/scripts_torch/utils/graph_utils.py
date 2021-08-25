import dgl
import torch
from copy import copy

import numpy as np
from collections import defaultdict

from .vis_graph import network_draw

node_type_list = ['robot', 'pedestrian', 'obstacle', 'goal']

# define edges direction, and threshold value for interaction distance
interaction_direction = {
    ('robot', 'pedestrian'): 10.0,
    ('robot', 'goal'): 1e6,

    ('pedestrian', 'pedestrian'): 5.0,
    ('pedestrian', 'robot') : 10.0,
    ('pedestrian', 'goal'): 1e6,

    ('obstacle', 'pedestrian'): 20.0, # large distance to prevent graph creatiion error due to zero edges
    ('obstacle', 'robot'): 20.0,

    ('goal', 'robot'): 1e6,
    ('goal', 'pedestrian'): 1e6,
    
}

state_dims = {
        "pos": 2,
        "vel": 2,
        "acc": 2,
        "rot": 1,
        "yaw": 1,
        "hed": 1,
        "gdist": 1,
        "diff": 2,
        "dist": 1,
        "action": 2,
        "goal": 2,
    }

def neighbor_eids(g, node):
    in_nodes, nodes = g.in_edges(node)
    in_eids = g.edge_ids(in_nodes, nodes)

    nodes, out_nodes = g.out_edges(node)
    out_eids = g.edge_ids(nodes, out_nodes)

    return torch.cat([in_eids, out_eids]).unique()

def create_graph(nodes, bidirectional=False):
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
        # nodes_data['acc'].append(src_node._acc[src_node.last_timestep])
        # nodes_data['rot'].append(src_node._rot[src_node.last_timestep])
        nodes_data['yaw'].append(src_node._yaw[src_node.last_timestep])
        nodes_data['hed'].append(src_node.heading(src_node.last_timestep))

        nodes_data['action'].append(src_node._action)
        nodes_data['goal'].append(src_node._goal) # call this after heading
        nodes_data['gdist'].append(src_node.distance_to_goal(src_node.last_timestep))
        
        nodes_data['time_step'].append(src_node.time_step)

        nodes_data['tid'].append(src_node._id)
        nodes_data['cid'].append(node_type_list.index(src_node._type))
        
        # spatial edges
        for j in range(len(nodes)):
            dst_node = nodes[j]
            try:
                rad = interaction_direction[src_node._type, dst_node._type]
            except:
                continue
            
            diff = np.array(src_node._pos[src_node.last_timestep]) - np.array(dst_node._pos[dst_node.last_timestep])

            dist = np.linalg.norm(diff, keepdims=True)

            if dist > rad:
                continue

            # edges from source to dest
            edges_data['src'].extend([i])
            edges_data['des'].extend([j])
            edges_data['dist'].extend([dist])
            edges_data['diff'].extend([diff])
            edges_data['spatial_mask'].extend([1.0])

            if bidirectional:
                edges_data['src'].extend([j])
                edges_data['des'].extend([i])
                edges_data['dist'].extend([dist])
                edges_data['diff'].extend([-diff])
                edges_data['spatial_mask'].extend([1.0])

    # Construct the DGL graph
    g = dgl.graph((edges_data['src'], edges_data['des']), num_nodes=len(nodes))
  
    # Add  features
    g.ndata['tid'] = torch.tensor(nodes_data['tid'], dtype=torch.int32)
    g.ndata['cid'] = torch.tensor(nodes_data['cid'], dtype=torch.int32)
    

    g.ndata['pos'] = torch.tensor(np.stack(nodes_data['pos'], axis=0), dtype=torch.float32).view(-1, state_dims['pos'])
    g.ndata['vel'] = torch.tensor(np.stack(nodes_data['vel'], axis=0), dtype=torch.float32).view(-1, state_dims['vel'])
    # g.ndata['acc'] = torch.tensor(np.stack(nodes_data['acc'], axis=0), dtype=torch.float32).view(-1, state_dims['acc'])

    # g.ndata['rot'] = torch.tensor(np.stack(nodes_data['rot'], axis=0), dtype=torch.float32).view(-1, state_dims['rot'])
    g.ndata['yaw'] = torch.tensor(np.stack(nodes_data['yaw'], axis=0), dtype=torch.float32).view(-1, state_dims['yaw'])

    g.ndata['action'] = torch.tensor(np.stack(nodes_data['action'], axis=0), dtype=torch.float32).view(-1, state_dims['action'])
    
    g.ndata['hed'] = torch.tensor(np.stack(nodes_data['hed'], axis=0), dtype=torch.float32).view(-1, state_dims['hed'])    
    g.ndata['goal'] = torch.tensor(np.stack(nodes_data['goal'], axis=0), dtype=torch.float32).view(-1, state_dims['goal'])
    g.ndata['gdist'] = torch.tensor(np.stack(nodes_data['gdist'], axis=0), dtype=torch.float32).view(-1, state_dims['gdist'])

    g.ndata['time_step'] = torch.tensor(np.stack(nodes_data['time_step'], axis=0), dtype=torch.float32).view(-1, 1)
    
    g.edata['dist'] = torch.tensor(np.array(edges_data['dist']), dtype=torch.float32).view(-1, state_dims['dist'])
    g.edata['diff'] = torch.tensor(np.array(edges_data['diff']), dtype=torch.float32).view(-1, state_dims['diff'])
    g.edata['spatial_mask'] = torch.tensor(np.array(edges_data['spatial_mask']), dtype=torch.int32).view(-1, 1)
    
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
        