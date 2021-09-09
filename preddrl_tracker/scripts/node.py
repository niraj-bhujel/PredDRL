'''
 Class to represent a single pedestrian. 
'''
import time
import math
import numpy as np
from collections import deque 

def euler_from_quaternion(orientation_list):
    '''
    orientation_list: [w, x, y, z]
    '''
    w, x, y, z = orientation_list
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = math.asin(2 * (w * y - z * x))
    y = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

    return r, p, y


class Node(object):
    def __init__(self, node_id=0, first_timestep=0, node_type='pedestrian', max_len=100):
        # self.data = data
        self._first_timestep = int(first_timestep)
        self._id = int(node_id)
        self._type = node_type
        
        self._max_len = max_len 
        
        self._pos = deque([], max_len)
        self._vel = deque([], max_len)
        self._acc = deque([], max_len)
        self._quat = deque([], max_len)
        self._rot = deque([], max_len)
        
        self._yaw = deque([], max_len)
        self._time_stamp = deque([], max_len)


    def __len__(self):
        return len(self._pos)

    def update_states(self, p, v, q, r):
        '''
        p: position, could be (x, y) or (x, y, z)
        v: linear velocity, (vx, vy) or (vx, vy, vz)
        q: quaternion (w, x, y, z)
        r: angular z velocity

        '''
        
        curr_timestamp = time.time()

        # if len(self)>0:

        #     dt = self._time_stamp[-1] - curr_timestamp

        #     last_p = self._pos[-1]
        #     last_v = self._vel[-1]

        #     v = (np.array(p) - np.array(last_p))/dt
        #     a = (np.array(v) - np.array(last_v))/dt

        # else:
        #     v = np.zeros_like(p)
        #     a = np.zeros_like(p)

        self._pos.append(p)
        self._vel.append(v)
        # self._acc.append(a)

        self._quat.append(q)
        self._rot.append(r)
        

        self._time_stamp.append(curr_timestamp)
    
    def get_states_at(self, t):
        
        ts_idx = self.timestep_index(t)
        
        p = self._pos[ts_idx]
        v = self._vel[ts_idx]
        # a = self._acc[ts_idx]
        
        q = self._quat[ts_idx]
        r = self._rot[ts_idx]
        
        return p, v, q, r

    def get_history(self, history_timesteps=8, desired_fps=2.5):

        frame_interval = int(self.frame_rate/desired_fps)
        
        if frame_interval<1:
            raise Exception('Unable to perform upsampling at the moment')
            
        sampled_idx = np.arange(-1, -self.timesteps, step=-frame_interval)[::-1]
        
        sampled_pos = np.array(self._pos)[sampled_idx]
        
        return sampled_pos[-history_timesteps:]

    def timestep_index(self, t):
        return t-self._first_timestep


    @property
    def time_step(self):
        if len(self)>1:
            return np.mean(np.diff(self._time_stamp)[-4:])
        # elif len(self)==1:
        #     fps = 1
        else:
            return 1

    @property
    def id(self):
        return self._id

    @property
    def timesteps(self):
        return len(self._pos)

    @property
    def first_timestep(self):
        return self._first_timestep

    @property
    def last_timestep(self):
        return self._first_timestep + self.timesteps - 1
