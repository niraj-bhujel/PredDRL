#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:01:29 2021

@author: dl-asoro
"""

from utils.graph_utils import *
import pickle

import warnings
warnings.filterwarnings("ignore")

g_path = '/home/loc/preddrl_ws/src/preddrl_td3/results/2021_12_24_ddpg_graph_warm_20_bs10_ht4_ft4_pt4_in_pos_vpref_pred_vel_h256_l2/vis_graphs/train'
for file in os.listdir(g_path):
    g = pickle.load(open(g_path + "/" + file, "rb"))

print(gs.ndata['cid'])
robot_node = state.nodes()[gs.ndata['cid']==node_type_list.index('robot')]
goal_node = state.nodes()[gs.ndata['cid']==node_type_list.index('robot_goal')]

# remove goal node
g = dgl.remove_nodes(deepcopy(gs), goal_node)
