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
    def __init__(self, node_id=0, first_timestep=0, node_type='pedestrian', max_len=100, goal=None):
        # self.data = data
        self._first_timestep = first_timestep
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

        self._goal = None
        self._action = None

    def __len__(self):
        return len(self._pos)
    
    def update_action(self, action):
        self._action = action

    def update_goal(self, goal):
        self._goal = goal

    def update_states(self, p, q, r):
        '''
        p: position, could be (x, y) or (x, y, z)
        v: linear velocity, (vx, vy) or (vx, vy, vz)
        q: quaternion (w, x, y, z)
        r: angular z velocity

        '''
        
        curr_timestamp = time.time()

        if len(self)>0:

            dt = self._time_stamp[-1] - curr_timestamp

            last_p = self._pos[-1]
            last_v = self._vel[-1]

            v = (np.array(p) - np.array(last_p))/dt
            a = (np.array(v) - np.array(last_v))/dt

        else:
            v = np.zeros_like(p)
            a = np.zeros_like(p)

        self._pos.append(p)
        self._vel.append(v)
        self._acc.append(a)

        self._quat.append(q)
        self._rot.append(r)
        
        self._yaw.append(euler_from_quaternion(q)[-1])
        
        self._time_stamp.append(curr_timestamp)
    
    def distance_to_goal(self, t):

        return round(math.hypot(self._goal[0] - self._pos[t][0], self._goal[1] - self._pos[t][1]), 2)

    def heading(self, t):
        
        inc_y = self._goal[1] - self._pos[t][1]
        inc_x = self._goal[0] - self._pos[t][0]
        goal_angle = math.atan2(inc_y, inc_x)

        yaw = euler_from_quaternion(self._quat[t])[-1]
        heading = goal_angle - yaw

        if heading > np.pi:
            heading -= 2 * np.pi

        if heading < -np.pi:
            heading += 2 * np.pi

        return round(heading, 2)

    def cv_prediction(self, t, pred_steps=12, time_step=None):

        if time_step is None:
            time_step = self.time_step

        _idx = self.timestep_index(t)

        preds = []
        x, y = self._pos[_idx]
        vx, vy = self._vel[_idx]
        ax, ay = self._acc[_idx]
        sec_from_now = pred_steps * time_step
        for time in np.arange(time_step, sec_from_now + time_step, time_step):
            half_time_squared = 0.5 * time * time
            preds.append([x + time * vx + half_time_squared * ax,
                          y + time * vy + half_time_squared * ay])
        return preds
    
    def states_at(self, t):
        
        ts_idx = self.timestep_index(t)
        
        p = self._pos[ts_idx]
        v = self._vel[ts_idx]
        a = self._acc[ts_idx]
        q = self._quat[ts_idx]
        r = self._rot[ts_idx]
        
        return p, v, a, q, r

    def get_history(self, history_timesteps=8, desired_fps=2.5):

        frame_interval = int(self.frame_rate/desired_fps)
        
        if frame_interval<1:
            raise Exception('Unable to perform upsampling at the moment')
            
        sampled_idx = np.arange(-1, -self.timesteps, step=-frame_interval)[::-1]
        
        sampled_pos = np.array(self._pos)[sampled_idx]
        
        return sampled_pos[-history_timesteps:]

    def timestep_index(self, t):
        return t-self._first_timestep


    def compute_position(self, action):

        theta = self._rot[-1] + action[1]
        px, py = self._pos[-1]
        px = px + np.cos(theta) * action[0] * self.time_step
        py = py + np.sin(theta) * action[0] * self.time_step

        return px, py


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
    def goal(self):
        return self._goal

    @property
    def timesteps(self):
        return len(self._pos)

    @property
    def first_timestep(self):
        return self._first_timestep

    @property
    def last_timestep(self):
        return self._first_timestep + self.timesteps - 1

    @goal.setter   #property-name.setter decorator
    def goal(self, value):
        self._goal = value
