'''
 Class to represent a single pedestrian. 
'''
import math
import numpy as np
from collections import deque 

class Node(object):
    def __init__(self, node_id=0, first_timestep=0, node_type='pedestrian', max_len=100, frame_rate=10):
        # self.data = data
        self.first_timestep = first_timestep
        self.id = node_id
        self.node_type = node_type
        self.frame_rate = frame_rate
        
        # self.states = deque([], maxlen=max_len)
        self.pos = deque([], maxlen=max_len)
        self.vel = deque([], maxlen=max_len)
        self.quat = deque([], maxlen=max_len)
        self.rot = deque([], maxlen=max_len)
        
        
    def update_states(self, p, v, q, r):
        # self.states.append([p, v, q, r])
        self.pos.append(p)
        self.vel.append(v)
        self.quat.append(q)
        self.rot.append(r)
        
        
    def cv_prediction(self, t, prediction_horizon=12, frame_rate=2.5):
        
        # past_pos, past_vel = self.history_at(t, history_timesteps=8*, frame_rate=frame_rate)
        
        pass
    
    
    def states_at(self, t):
        
        ts_idx = self.timestep_index(t)
        
        p = self.pos[ts_idx]
        v = self.vel[ts_idx]
        
        q = self.quat[ts_idx]
        r = self.rot[ts_idx]
        
        return p, v, q, r



    def get_history(self, history_timesteps=8, frame_rate=2.5):

        frame_interval = int(self.frame_rate/frame_rate)
        
        if frame_interval<1:
            raise Exception('Unable to perform upsampling at the moment')
            
        sampled_idx = np.arange(-1, -self.timesteps, step=-frame_interval)[::-1]
        
        sampled_pos = np.array(self.pos)[sampled_idx]
        
        return sampled_pos[-history_timesteps:]

    def timestep_index(self, t):
        return t-self.first_timestep

    @property
    def timesteps(self):
        return len(self.pos)

    @property
    def last_timestep(self):
        return self.first_timestep + self.timesteps - 1