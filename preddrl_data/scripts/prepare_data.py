import os
import sys
if not './' in sys.path:
    sys.path.insert(0, './')

import math
import numpy as np

from scipy.interpolate import interp1d
from preddrl_td3.scripts.utils.agent import Agent

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
    print('Loading data ... ', data_path)
    target_frame_rate =  np.clip(2.5, target_frame_rate, 25)
    
    data = np.loadtxt(data_path).round(2)

    # convert frame rate into 25 fps from 2.5fps
    data_frames = np.unique(data[:, 0])
    frame_rate_multiplier = target_frame_rate/2.5
    
    # keep the original key frames, sample between frame intervals
    interframes = [np.linspace(0, diff, num=int(frame_rate_multiplier), endpoint=False) for diff in np.diff(data_frames)]
    intp_data_frames = np.concatenate([int_f + key_f for int_f, key_f in zip(interframes, data_frames)] +
                                      [np.linspace(data_frames[-1], data_frames[-1]+10, num=int(frame_rate_multiplier), endpoint=False)])

    ped_nodes = []
    ped_intp_frames = []
    num_ped_considered = 0
    for pid in np.unique(data[:, 1]):

        ped_frames = data[data[:, 1]==pid, 0]        
        num_intp_points = int(len(ped_frames)*frame_rate_multiplier)

        ped_pos = data[data[:, 1]==pid, 2:4]
        ped_vel = (ped_pos[1:] - ped_pos[:-1])*2.5
        
        if not len(ped_pos)>2:
            continue
            
        intp_ped_pos = interpolate(ped_pos, 'quadratic', num_intp_points).round(2)
        
        intp_ped_vel = np.round((intp_ped_pos[1:] - intp_ped_pos[:-1]) * target_frame_rate, 2)

        # intp_ped_vel = np.gradient(intp_ped_pos, 1.0/target_frame_rate, axis=0).round(2)

        start_idx = intp_data_frames.tolist().index(ped_frames[0])
        
        node = Agent(pid, first_timestep=start_idx, 
                     time_step=1./target_frame_rate, 
                     node_type='pedestrian', 
                     pref_speed = np.linalg.norm(np.mean(ped_vel, axis=0)),
                     history_len=num_intp_points)
        
        for i in range(len(intp_ped_pos)-1):
            
            px, py = intp_ped_pos[i+1][0], intp_ped_pos[i+1][1]
            vx, vy = intp_ped_vel[i][0], intp_ped_vel[i][1]

            gx, gy = intp_ped_pos[-1][0], intp_ped_pos[-1][1]

            theta = round(math.atan2(vy, vx), 2) # radians
            
            # node.update_states(px, v, q, r)
            node.update_history(px, py, vx, vy, gx, gy, theta)

        ped_nodes.append(node)
        
        ped_intp_frames.append(intp_data_frames[start_idx:start_idx+num_intp_points])
        
        num_ped_considered+=1
        
        # break
    
        if num_ped_considered>max_peds:
            break
    
    peds_frames = []
    peds_per_frame = []
    for t, frame in enumerate(intp_data_frames):
        curr_ped = []
        for i, node in enumerate(ped_nodes):

            if t>=node.first_timestep and t<=node.last_timestep:
                
                curr_ped.append(ped_nodes[i].id)
        
        if len(curr_ped)>0:
            peds_frames.append(frame)
            peds_per_frame.append(curr_ped)
        
        
    return ped_nodes, peds_frames, peds_per_frame

