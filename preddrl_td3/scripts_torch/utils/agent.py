'''
 Class to represent a single pedestrian. 
'''
import time
import math
import numpy as np
from collections import deque, namedtuple

State = namedtuple('State', ['px', 'py', 'vx', 'vy', 'gx', 'gy', 'theta'])

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
    def __init__(self, node_id, node_type='robot', first_timestep=0, time_step=0.1, vpref=0.4, radius=0.2, history_len=100):

        self.first_timestep = int(first_timestep)
        self.id = int(node_id)
        self.type = node_type
        self.time_step = time_step
        self.radius = radius
        self.vpref = vpref # speed

        self.px = None
        self.py = None

        self.gx = None
        self.gy = None

        self.vx = None
        self.vy = None

        self.theta = None
        self.action = (0., 0.)

        self.state_history = []

    def __len__(self):
        return len(self.state_history)

    def set_action(self, action):
        self.action = action

    def set_goal(self, gx, gy):
        self.gx = gx
        self.gy = gy

    def set_futures(self, futures):
        self.futures = futures

    def set_state(self, px, py, vx, vy, gx, gy, theta):

        # if self.px is not None:
        #     self.ax = (vx - self.vx)/self.time_step
        #     self.ay = (vy - self.vy)/self.time_step
        # else:
        #     self.ax, self.ay = 0., 0.

        self.px = px
        self.py = py

        self.vx = vx
        self.vy = vy

        self.gx = gx
        self.gy = gy
        
        self.theta = theta

    def update_history(self, px, py, vx, vy, gx, gy, theta):
        self.state_history.append(State(px, py, vx, vy, gx, gy, theta))

    def get_state(self, ):
        return State(self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.theta)

    def get_state_at(self, t):
        _idx = t - self.first_timestep
        return self.state_history[_idx]

    def get_futures_at(self, t, future_steps=4):
        _idx = t - self.first_timestep
        future_states = [self.state_history[i] for i in range(_idx, min(_idx+future_steps, self.timesteps))]
        return future_states

    def cv_prediction(self, pred_steps=4):

        preds = []
        sec_from_now = pred_steps * self.time_step

        for time in np.arange(self.time_step, sec_from_now + self.time_step, self.time_step):
            half_time_squared = 0.5 * time * time

            preds.append([self.px + time * self.vx + half_time_squared * self.ax, 
                          self.py + time * self.vy + half_time_squared * self.ay,
                          ])
        return preds
    

    def compute_position(self, action):

        theta = self.theta + action[1]

        px = self.px + np.cos(theta) * action[0] * self.time_step
        py = self.py + np.sin(theta) * action[0] * self.time_step

        return px, py

    def compute_heading(self, ):

        goal_angle = math.atan2(self.gy - self.py, self.gx-self.px)

        heading = goal_angle - self.theta

        if heading > np.pi:
            heading -= 2 * np.pi

        if heading < -np.pi:
            heading += 2 * np.pi

        return round(heading, 2)

    def distance_to_goal(self,):
        return round(math.hypot(self.gx - self.px, self.gy - self.py), 2)

    def preferred_vel(self, speed=0.4):
        goal_vec = np.array((self.gx - self.px, self.gy - self.py))
        norm = np.linalg.norm(goal_vec)
        pref_vel = goal_vec/norm if norm>1 else goal_vec
        return pref_vel

    def serialize_state(self, s):
        return State(s)

    def deserialize_state(self, s):
        return (s.px, s.py, s.vx, s.vy, s.gx, s.gy, s.theta)

    @property
    def state(self, ):
        return (self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.theta)
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
        return self.compute_heading()

    @property
    def timesteps(self):
        return len(self.state_history)

    @property
    def last_timestep(self):
        return self.first_timestep + self.timesteps - 1
