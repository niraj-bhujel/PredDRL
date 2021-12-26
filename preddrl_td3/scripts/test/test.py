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
run_dir = '../results/'
run = '2021_12_25_ddpg_graph_warm_200_bs10_ht4_ft4_pt2_in_pos_vpref_pred_vel_h256_l2'
run_path = os.path.join(run_dir, run, 'vis_graphs/train')

g = pickle.load(open(run_path + '/step31_episode_step31.pkl',  "rb"))
#%%
for file in os.listdir(run_path):
    if '.pkl' in file:
        g = pickle.load(open(g_path + "/" + file, "rb"))
        print(g.node_attr_schemes().keys())

print(gs.ndata['cid'])
robot_node = state.nodes()[gs.ndata['cid']==node_type_list.index('robot')]
goal_node = state.nodes()[gs.ndata['cid']==node_type_list.index('robot_goal')]

# remove goal node
g = dgl.remove_nodes(deepcopy(gs), goal_node)
