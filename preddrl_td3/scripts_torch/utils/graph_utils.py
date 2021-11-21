import dgl
import torch
from copy import copy, deepcopy

import numpy as np
from collections import defaultdict

from .vis_graph import network_draw

node_type_list = ['robot', 'pedestrian', 'obstacle', 'robot_goal']

# define edges direction, and threshold value for interaction distance
interaction_direction = {
    ('robot', 'pedestrian'): 3.0,

    ('robot', 'robot_goal'): 1e6,
    ('robot_goal', 'robot'): 1e6,

    ('pedestrian', 'pedestrian'): 3.0,
    ('pedestrian', 'robot') : 3.0,

    ('obstacle', 'pedestrian'): 5.0, # large distance to prevent graph creatiion error due to zero edges
    ('obstacle', 'robot'): 15,
    # ('obstacle', 'obstacle'): 2.5,

}

state_dims = {
        "pos": 2,
        "vel": 2,
        "rel_vel": 2,
        "acc": 2,
        "rot": 1,
        "yaw": 1,
        "hed": 1,
        "dir": 2,
        "vpref": 2,
        "theta": 1, 
        "gdist": 1,
        "diff": 2,
        "dist": 1,
        "action": 2,
        "goal": 2,
        "state":7,
        "future": 2*4, # 4 is future steps
        "spatial_mask": 1,
    }

def min_neighbor_distance(g, node, mask_nodes=[]):
    if len(mask_nodes)>0:
        g = dgl.remove_nodes(deepcopy(g), mask_nodes)

    in_nodes, nodes = g.in_edges(node)
    in_eids = g.edge_ids(in_nodes, nodes)

    nodes, out_nodes = g.out_edges(node)
    out_eids = g.edge_ids(nodes, out_nodes)

    node_neibhbors_eids = torch.cat([in_eids, out_eids]).unique()

    if node_neibhbors_eids.size()[0]>0:
        return g.edata['dist'][node_neibhbors_eids].min()
    else:
        # no nodes nearby, thus infinite distance
        return 1e3

def n1_has_n2_in_sight(n1, n2, fov=57):
    '''
    Check if node2 is in the field of node
    https://stackoverflow.com/questions/22542821/how-to-calculate-if-something-lies-in-someones-field-of-vision
    '''
    alpha = math.atan2(n1.pos[1], n1.pos[0])
    
    # angle betwen node
    d = (n1.pos[0] - n2.pos[0], n1.pos[1] - n2.pos[1])
    beta = math.atan2(d[1], d[0])
    
    angle = beta - alpha
    
    if angle > math.pi:
        angle = angle - 2*math.pi
    if angle < -np.pi:
        angle = angle + 2*math.pi
    
    return abs(angle)<fov*math.pi/180

def create_graph(nodes, bidirectional=False):
    '''
        Create a graphs with node representing a pedestrians/robot/obstacle.
    '''

    nodes_data = defaultdict(list)
    edges_data = defaultdict(list)
    # N = len(nodes)

    src_nodes = []
    dst_nodes = []
    for i, src_node in enumerate(nodes):
        
        num_neighbors = 0
        # spatial edges
        for j, dst_node in enumerate(nodes):
            
            try:
                rad = interaction_direction[src_node.type, dst_node.type]
            except Exception:
                continue
            
            diff = np.array(src_node.pos) - np.array(dst_node.pos)

            dist = np.linalg.norm(diff, keepdims=True)

            if dist > rad:
                continue

            # if n1_has_n2_in_sight(src_node, dst_node):
            # edges from source to dest
            src_nodes.extend([i])
            dst_nodes.extend([j])

            edges_data['dist'].extend([dist])
            edges_data['diff'].extend([diff])

            edges_data['rel_vel'].extend([np.array(src_node.vel) - np.array(dst_node.vel)])

            edges_data['spatial_mask'].extend([1.0])

            if bidirectional:
                src_nodes.extend([j])
                dst_nodes.extend([i])
                edges_data['dist'].extend([dist])
                edges_data['diff'].extend([-diff])
                edges_data['spatial_mask'].extend([1.0])

            num_neighbors+=1

    robot_node = [node for node in nodes if node.type=='robot'][0]
    # prepare node data, discard node without edges
    valid_nodes = np.unique(src_nodes + dst_nodes)
    for n in valid_nodes:
        node = nodes[n]

        nodes_data['pos'].append(node.pos)
        nodes_data['vel'].append(node.vel)
        nodes_data['vpref'].append(node.preferred_vel())

        nodes_data['theta'].append(node.theta)
        nodes_data['dir'].append([node.gx - node.px, node.gy - node.py])
        nodes_data['hed'].append(node.heading)

        nodes_data['action'].append(node.action)
        nodes_data['goal'].append(node.goal)
        
        nodes_data['state'].append(node.state)
        nodes_data['future'].append([s for futures in node.futures for s in futures])

        nodes_data['tid'].append(node.id)
        nodes_data['cid'].append(node_type_list.index(node.type))


    # Construct the DGL graph
    g = dgl.graph((src_nodes, dst_nodes))
    g = dgl.node_subgraph(g, valid_nodes)
    
    # Add  features

    
    for node_attr, ndata in nodes_data.items():
        if node_attr in ['tid', 'cid']:
            g.ndata[node_attr] = torch.tensor(ndata, dtype=torch.int32)
        else:
            g.ndata[node_attr] = torch.tensor(np.stack(ndata, axis=0), dtype=torch.float32).view(-1, state_dims[node_attr])

    for edge_attr, edata in edges_data.items():
        if edge_attr=='spatial_mask':
            g.edata[edge_attr] = torch.tensor(np.array(edata), dtype=torch.int32).view(-1, state_dims[edge_attr])
        else:
            g.edata[edge_attr] = torch.tensor(np.array(edata), dtype=torch.float32).view(-1, state_dims[edge_attr])
    
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
                    edges_data['dst'].extend([v])

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
            nodes_data['tid'].append(nodes[i].id)
            nodes_data['cid'].append(node_type_list.index(nodes[i].type))
            
            # spatial edges
            for j in range(i+1, len(nodes)):

                # use last pos
                if t==0:
                    diff = nodes[i].pos[-1] - nodes[j].pos[-1]
                # use predicted pos
                else:
                    diff = nodes[i]._preds[t-1] - nodes[j]._preds[t-1]

                dist = np.linalg.norm(diff) 
                if dist > interaction_radius:
                    continue

                u = current_nodes[i]
                v = current_nodes[j]
                
                edges_data['src'].extend([u, v])
                edges_data['dst'].extend([v, u])
                edges_data['dist'].extend([dist, dist])
                edges_data['diff'].extend([-diff, diff]) # difference measured from destination
    
                edges_data['spatial_mask'].extend([1.0, 1.0])

        previous_nodes = current_nodes
        N+=len(current_nodes)
        