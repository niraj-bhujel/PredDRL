'''
 Class to represent a single pedestrian. 
'''
import time
import math
import numpy as np
from collections import deque 

class Node(object):
    def __init__(self, node_id=0, first_timestep=0, node_type='pedestrian', max_len=100, goal=None):
        # self.data = data
        self._first_timestep = first_timestep
        self._id = node_id
        self._type = node_type
        self._length = 0
        self._goal = goal

        self._pos = deque([], maxlen=max_len)
        self._vel = deque([], maxlen=max_len)
        self._acc = deque([], maxlen=max_len)
        self._quat = deque([], maxlen=max_len)
        self._rot = deque([], maxlen=max_len)
        self._time_stamp = deque([], maxlen=max_len)

    def __len__(self):
        return len(self._pos)
        
    def update_states(self, p, v, q, r):
        '''
        p: position, could be (x, y) or (x, y, z)
        v: linear velocity, (vx, vy) or (vx, vy, vz)
        q: quaternion (w, x, y, z)
        r: angular velocity (x, y) or (x, y, z) as in Twist
        '''
        
        curr_timestamp = time.time()

        if len(self)>0:
            a = np.asarray(self._vel[-1]) - np.asarray(v)
            a = a/(self._time_stamp[-1] - curr_timestamp)
        else:
            a = np.zeros_like(v)

        self._pos.append(p)
        self._vel.append(v)
        self._acc.append(a)
        self._quat.append(q)
        self._rot.append(r)
        
        self._time_stamp.append(curr_timestamp)
        
    # def update_goal(self, t, future_step=12):
    #     preds = self.cv_prediction(t, future_step, time_step=1.0/self.frame_rate)
    #     self._goal = preds[-1]

    def cv_prediction(self, t, pred_steps=20, time_step=None):
        if time_step is None:
            time_step = 1.0/self.frame_rate

        _idx = self.timestep_index(t)

        preds = []
        x, y, z = self._pos[_idx]
        vx, vy, vz = self._vel[_idx]
        ax, ay, az = self._acc[_idx]
        sec_from_now = pred_steps * time_step
        for time in np.arange(time_step, sec_from_now + time_step, time_step):
            half_time_squared = 0.5 * time * time
            preds.append((x + time * vx + half_time_squared * ax,
                          y + time * vy + half_time_squared * ay,
                          z + time * vz + half_time_squared * az))
        return preds
    
    def states_at(self, t):
        
        ts_idx = self.timestep_index(t)
        
        p = self._pos[ts_idx]
        v = self._vel[ts_idx]
        
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
    def frame_rate(self):
        if len(self)>1:
            fps = 1/np.mean(np.diff(self._time_stamp))
        elif len(self)==1:
            fps = 1
        else:
            fps = 0
        return fps

    @ property
    def goal(self):
        return self._goal

    @property
    def timesteps(self):
        return len(self._pos)

    @property
    def last_timestep(self):
        return self._first_timestep + self.timesteps - 1

    @goal.setter   #property-name.setter decorator
    def goal(self, value):
        self._goal = value
