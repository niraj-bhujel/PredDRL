#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


import os
import sys


import math
import numpy as np

from scipy.interpolate import interp1d

from preddrl_tracker.scripts.node import Node

        
def angleToQuaternion(theta, angles=True):

    # if not angles:
    theta *= np.pi/180

    R = np.zeros((9))
    R[0] = np.cos(theta)
    R[1] = -np.sin(theta)
    R[3] = np.sin(theta)
    R[4] = np.cos(theta)
    R[8] = 1
    
    w = math.sqrt(R[0]+R[4]+R[8]+1)/2.0
    x = math.sqrt(R[0]-R[4]-R[8]+1)/2.0
    y = math.sqrt(-R[0]+R[4]-R[8]+1)/2.0
    z = math.sqrt(-R[0]-R[4]+R[8]+1)/2.0
    
    q = [w,x,y,z]

    return q

def interpolate(pos, method='quadratic', num_points=1):

    x, y = pos[:, 0], pos[:, 1]

    x_intp = interp1d(np.arange(len(x)), x, kind=method)
    y_intp = interp1d(np.arange(len(y)), y, kind=method)

    points = np.linspace(0, len(x)-1, num_points)

    return np.stack([x_intp(points), y_intp(points)], axis=-1)

def _gradient(x, dt=0.4, axis=0):
    '''
    x - array of shape (T, dim)
    '''

    g =  np.diff(x, axis=axis) / dt
    g = np.concatenate([g[0][None, :], g], axis=axis)
    
    return g

def prepare_data(data_path, target_frame_rate=25, max_peds=20):
    print('Preparing data .. ')

    target_frame_rate =  max(target_frame_rate, 2.5)
    
    data = np.loadtxt(data_path).round(2)

    # convert frame rate into 25 fps from 2.5fps
    data_frames = np.unique(data[:, 0])
    frame_rate_multiplier = target_frame_rate/2.5
    
    # keep the original key frames, sample between frame intervals
    interframes = [np.linspace(0, diff, num=int(frame_rate_multiplier), endpoint=False) for diff in np.diff(data_frames)]
    intp_data_frames = np.concatenate([int_f + key_f for int_f, key_f in zip(interframes, data_frames)] +
                                      [np.linspace(data_frames[-1], data_frames[-1]+10, num=int(frame_rate_multiplier), endpoint=False)])

    ped_nodes = {}
    num_ped_considered = 0
    for pid in np.unique(data[:, 1]):

        ped_frames = data[data[:, 1]==pid, 0]        
        num_intp_points = int(len(ped_frames)*frame_rate_multiplier)

        ped_pos = data[data[:, 1]==pid, 2:4]
        
        intp_ped_pos = interpolate(ped_pos, 'quadratic', num_intp_points).round(2)

        intp_ped_vel = np.gradient(intp_ped_pos, 1.0/target_frame_rate, axis=0).round(2)
        # ped_acc = np.gradient(ped_vel, 1.0/target_frame_rate, axis=0).round(2)

        start_idx = intp_data_frames.tolist().index(ped_frames[0])

        
        node = Node(pid, start_idx, node_type='pedestrian', max_len=num_intp_points)
        
        for i in range(len(intp_ped_pos)):
            
            p = [intp_ped_pos[i][0], intp_ped_pos[i][1], 0.]
            v = [intp_ped_vel[i][0], intp_ped_vel[i][1], 0.]
            
            theta = math.atan2(v[1], v[0]) # radians
            q = angleToQuaternion(theta)
            
            r = [0., 0., 0.]
            
            node.update_states(p, v, q, r)

        ped_nodes[pid] = node
                
        num_ped_considered+=1
        
        if num_ped_considered>max_peds:
            break

    return ped_nodes