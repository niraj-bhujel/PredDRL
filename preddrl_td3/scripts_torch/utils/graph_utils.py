import dgl
import torch
from copy import copy, deepcopy

import numpy as np
from collections import defaultdict

from .vis_graph import network_draw

node_type_list = ['robot', 'pedestrian', 'obstacle', 'robot_goal']

# define edges direction, and threshold value for interaction distance
interaction_direction = {
    ('robot', 'pedestrian'): 5.0,

    ('robot', 'robot_goal'): 1e6,
    ('robot_goal', 'robot'): 1e6,

    ('pedestrian', 'pedestrian'): 3.0,
    ('pedestrian', 'robot') : 5.0,

    ('obstacle', 'pedestrian'): 5.0, # large distance to prevent graph creatiion error due to zero edges
    ('obstacle', 'robot'): 15,
    # ('obstacle', 'obstacle'): 2.5,

}

state_dims = {
        "pos": 2,
        "rel": 2,
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
        "current_states":7,
        "future_states": 2*4, # 4 is future steps
    }


def remove_uncommon_nodes(g1, g2):
    g1_tid = g1.ndata['tid'].cpu().numpy()
    g2_tid = g2.ndata['tid'].cpu().numpy()

    comm_tid = np.sort(np.intersect1d(g1_tid, g2_tid))

    # remove node belonging to uncommon tid
    g1_redundant_nodes = [g1.nodes()[g1.ndata['tid']==tid] for tid in g1_tid if tid not in comm_tid]
    g1_node_idx = [i for i, node in enumerate(g1.nodes()) if node not in g1_redundant_nodes]
    
    if len(g1_redundant_nodes)>0:
        g1.remove_nodes(g1_redundant_nodes)

    g2_redundant_nodes = [g2.nodes()[g2.ndata['tid']==tid] for tid in g2_tid if tid not in comm_tid]
    g2_node_idx = [i for i, node in enumerate(g2.nodes()) if node not in g2_redundant_nodes]
    
    if len(g2_redundant_nodes)>0:
        g2.remove_nodes(g2_redundant_nodes)

    return g1, g2, g1_node_idx, g2_node_idx

def find_collision_nodes(g, mask_nodes=[], collision_threshold=0.2):
    src_nodes, dst_nodes = g.edges()
    collision_edges = g.edge_ids(src_nodes, dst_nodes)[g.edata['dist'].squeeze(1)<collision_threshold]
    # find the src and dst nodes of the collision edges
    collision_src_nodes, collision_dst_nodes = g.find_edges(collision_edges)

    collision_src_nodes = [node for node in collision_src_nodes if node not in mask_nodes]
    collision_dst_nodes = [node for node in collision_dst_nodes if node not in mask_nodes]

    collision_nodes = np.unique([collision_src_nodes, collision_dst_nodes])
    collision_nidx = [i for i, node in enumerate(g.nodes()) if node in collision_nodes]

    return collision_nodes, collision_nidx

def min_neighbor_distance(g, node, mask_nodes=[]):
    if len(mask_nodes)>0:
        g = dgl.remove_nodes(g, mask_nodes)

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

def create_graph(nodes, ref_pos=(0., 0.), bidirectional=False):
    '''
        Create a graphs with node representing a pedestrians/robot/obstacle.
    '''

    nodes_data = defaultdict(list)
    edges_data = defaultdict(list)
    # N = len(nodes)

    for i, src_node in enumerate(nodes):

        nodes_data['pos'].append(src_node.pos)
        nodes_data['rel'].append([src_node.pos[0]-ref_pos[0], src_node.pos[1]-ref_pos[1]])

        nodes_data['yaw'].append(src_node.theta)
        nodes_data['hed'].append(src_node.heading)
        
        nodes_data['action'].append(src_node.action)
        nodes_data['goal'].append(src_node.goal)
        
        nodes_data['current_states'].append(src_node.get_states())
        nodes_data['future_states'].append([s for futures in src_node.get_futures() for s in futures])

        nodes_data['gdist'].append(src_node.distance_to_goal())
        
        nodes_data['tid'].append(src_node.id)
        nodes_data['cid'].append(node_type_list.index(src_node.type))

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
            edges_data['src'].extend([i])
            edges_data['dst'].extend([j])
            edges_data['dist'].extend([dist])
            edges_data['diff'].extend([diff])
            edges_data['spatial_mask'].extend([1.0])

            if bidirectional:
                edges_data['src'].extend([j])
                edges_data['dst'].extend([i])
                edges_data['dist'].extend([dist])
                edges_data['diff'].extend([-diff])
                edges_data['spatial_mask'].extend([1.0])

            num_neighbors+=1


    # Construct the DGL graph
    g = dgl.graph((edges_data['src'], edges_data['dst']))

    # Add  features
    g.ndata['tid'] = torch.tensor(nodes_data['tid'], dtype=torch.int32)
    g.ndata['cid'] = torch.tensor(nodes_data['cid'], dtype=torch.int32)
    
    g.ndata['pos'] = torch.tensor(np.stack(nodes_data['pos'], axis=0), dtype=torch.float32).view(-1, state_dims['pos'])
    g.ndata['rel'] = torch.tensor(np.stack(nodes_data['rel'], axis=0), dtype=torch.float32).view(-1, state_dims['rel'])


    g.ndata['action'] = torch.tensor(np.stack(nodes_data['action'], axis=0), dtype=torch.float32).view(-1, state_dims['action'])
    g.ndata['yaw'] = torch.tensor(np.stack(nodes_data['yaw'], axis=0), dtype=torch.float32).view(-1, state_dims['yaw'])
    g.ndata['hed'] = torch.tensor(np.stack(nodes_data['hed'], axis=0), dtype=torch.float32).view(-1, state_dims['hed'])    
    g.ndata['goal'] = torch.tensor(np.stack(nodes_data['goal'], axis=0), dtype=torch.float32).view(-1, state_dims['goal'])

    g.ndata['gdist'] = torch.tensor(np.stack(nodes_data['gdist'], axis=0), dtype=torch.float32).view(-1, state_dims['gdist'])

    g.ndata["current_states"] = torch.tensor(np.stack(nodes_data['current_states'], axis=0), dtype=torch.float32).view(-1, state_dims['current_states'])
    g.ndata["future_states"] = torch.tensor(np.stack(nodes_data['future_states'], axis=0), dtype=torch.float32).view(-1, state_dims['future_states'])

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
        