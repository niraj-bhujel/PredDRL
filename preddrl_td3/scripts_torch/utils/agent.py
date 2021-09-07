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
    def __init__(self, node_id=0, node_type='robot', time_step=0.1, vpref=0.4, radius=0.2):
        # self.data = data

        self.id = node_id
        self.type = node_type
        self.time_step = time_step
        self.radius = radius
        self.vpref = vpref

        self.px = None
        self.py = None

        self.gx = None
        self.gy = None

        self.vx = None
        self.vy = None

        self.theta = None
        self.action = (0., 0.)

    def __len__(self):
        return len(self._pos)

    def update_action(self, action):
        self.action = action

    def update_goal(self, gx, gy):
        self.gx = gx
        self.gy = gy
        
    def update_states(self, px, py, gx, gy, theta):

        if self.px is not None:
            vx = (px - self.px)/self.time_step
            vy = (py - self.py)/self.time_step

            ax = (vx - self.vx)/self.time_step
            ay = (vy - self.vy)/self.time_step

        else:
            vx, vy = 0., 0.
            ax, ay = 0., 0.

        self.px = px
        self.py = py

        self.vx = vx
        self.vy = vy

        self.ax = ax
        self.ay = ay

        self.gx = gx
        self.gy = gy
        
        self._theta = theta

        self._time_stamp = time.time()

    def cv_prediction(self, pred_steps=4, time_step=0.5):

        preds = []
        sec_from_now = pred_steps * time_step
        for time in np.arange(time_step, sec_from_now + time_step, time_step):
            half_time_squared = 0.5 * time * time
            preds.append([self.px + time * self.vx + half_time_squared * self.ax,
                          self.py + time * self.vy + half_time_squared * self.ay])
        return preds
    

    def compute_position(self, action):

        theta = self._theta + action[1]
        px, py = self._pos
        px = px + np.cos(theta) * action[0] * self.time_step
        py = py + np.sin(theta) * action[0] * self.time_step

        return px, py


    def distance_to_goal(self,):
        return round(math.hypot(self.gx - self.px, self.gy - self.py), 2)

    def preferred_vel(self, v_pref=0.4):
        velocity = np.array((self.gx - self.px, self.gy - self.py))
        pref_vel = v_pref * velocity / np.linalg.norm(velocity)
        return pref_vel

    @property
    def pos(self, ):
        return (self.px, self.py)

    @property
    def vel(self, ):
        return (self.vx, self.vy)

    @property
    def goal(self, ):
        return (self.gx, self.gy)

    @property
    def heading(self, ):
        
        goal_angle = math.atan2(self.gy - self.py, self.gx-self.px)

        heading = goal_angle - self._theta

        if heading > np.pi:
            heading -= 2 * np.pi

        if heading < -np.pi:
            heading += 2 * np.pi

        return round(heading, 2)
