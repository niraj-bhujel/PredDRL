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


class Agent(object):
    def __init__(self, node_id=0, node_type='robot', time_step=0.1):
        # self.data = data

        self._id = int(node_id)
        self._type = node_type
        self._time_step = time_step

        self._pos = None
        self._vel = None
        self._acc = None

        self._quat = None
        self._rot = None
        
        self._yaw = None

        self._goal = None
        self._action = None

        self._time_stamp = None

    def __len__(self):
        return len(self._pos)
    
    def update_action(self, action):
        self._action = action

    def update_goal(self, goal):
        self._goal = goal
    def update_heading(self, heading):
        self._heading = heading
        
    def update_states(self, p, q, r):
        '''
        p: position, could be (x, y) or (x, y, z)
        v: linear velocity, (vx, vy) or (vx, vy, vz)
        q: quaternion (w, x, y, z)
        r: angular z velocity

        '''

        if self._pos is not None:
            v = (p - self._pos)/self._time_step
            a = (v - self._vel)/self._time_step
        else:
            v = np.zeros_like(p)
            a = np.zeros_like(v)

        self._pos = p
        self._vel = v
        self._acc = a

        self._quat = q
        self._rot = r
        
        self._yaw = euler_from_quaternion(q)[-1]

        if self._time_stamp is not None:
            self._time_step = self._time_stamp - time.time()

        self._time_stamp = time.time()

    def cv_prediction(self, pred_steps=4, time_step=0.5):

        preds = []
        x, y = self._pos
        vx, vy = self._vel
        ax, ay = self._acc
        sec_from_now = pred_steps * time_step
        for time in np.arange(time_step, sec_from_now + time_step, time_step):
            half_time_squared = 0.5 * time * time
            preds.append([x + time * vx + half_time_squared * ax,
                          y + time * vy + half_time_squared * ay])
        return preds
    

    def compute_position(self, action):

        theta = self._rot + action[1]
        px, py = self._pos
        px = px + np.cos(theta) * action[0] * self.time_step
        py = py + np.sin(theta) * action[0] * self.time_step

        return px, py

    @property
    def distance_to_goal(self,):
        return round(math.hypot(self._goal[0] - self._pos[0], self._goal[1] - self._pos[1]), 2)

    @property
    def heading(self, ):
        
        inc_y = self._goal[1] - self._pos[1]
        inc_x = self._goal[0] - self._pos[0]
        goal_angle = math.atan2(inc_y, inc_x)

        # yaw = euler_from_quaternion(self._quat)
        heading = goal_angle - self._yaw

        if heading > np.pi:
            heading -= 2 * np.pi

        if heading < -np.pi:
            heading += 2 * np.pi

        return round(heading, 2)

    @property
    def time_step(self):
        return self._time_step

    @property
    def id(self):
        return self._id

    @property
    def goal(self):
        return self._goal
