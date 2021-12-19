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


gs = pickle.load(open("/home/dl-asoro/Desktop/PredDRL/preddrl_td3/results/2021_12_16_run0_ddpg_graph_warmup_20_bs10_input_history_disp_vel_vpref_h256_l2_pred_future/graphs/train/step1_episode_step0.pkl", "rb"))

print(gs.ndata['cid'])
robot_node = state.nodes()[gs.ndata['cid']==node_type_list.index('robot')]
goal_node = state.nodes()[gs.ndata['cid']==node_type_list.index('robot_goal')]

# remove goal node
g = dgl.remove_nodes(deepcopy(gs), goal_node)
