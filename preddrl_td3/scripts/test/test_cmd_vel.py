import os
import sys
import time
import logging
import shutil
import random
import numpy as np
from collections import deque
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

import dgl
from dgl.heterograph import DGLHeteroGraph

import rospy

from gym.spaces import Box

if './' not in sys.path: 
    sys.path.insert(0, './')


import args
import yaml

from utils.utils import *

from env.environment import Env
from utils.graph_utils import node_type_list

from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelStates

rospy.init_node('pred_drl', disable_signals=True)

parser = args.get_argument()
args = parser.parse_args()

model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)

env = Env(test=False, stage=args.stage, graph_state=True, dataset=args.dataset)

try:
    total_steps = 0
    obs = env.reset(initGoal=True) # add initGoal arg by niraj
    while total_steps < args.max_steps:

        print('Step - {}/{}'.format(total_steps, args.max_steps))

        action  = env.sample_robot_action(args.sampling_method)

        if isinstance(obs, DGLHeteroGraph):
            robot_action = action
            action = obs.ndata['action'].numpy()
            action[obs.ndata['cid']==node_type_list.index('robot')] = robot_action

        next_obs, reward, done, success = env.step(action, obs)

        if isinstance(obs, DGLHeteroGraph):
            robot_action = action[obs.ndata['cid']==node_type_list.index('robot')].flatten()
        else:
            robot_action = action

        print('Robot Action:', np.round(robot_action, 2))

        obs = next_obs
        if done:
            obs = env.reset()

        total_steps += 1

except KeyboardInterrupt:

    print("Waiting for gazebo delete_model services...")
    rospy.wait_for_service("gazebo/delete_model")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    print('Clearning existing pedestrians models from', model_states.name)
    [delete_model(model_name) for model_name in model_states.name if 'pedestrian' in model_name]  

