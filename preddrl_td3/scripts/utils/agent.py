'''
 Class to represent a single pedestrian. 
'''
import time
import math
import numpy as np
from collections import deque, namedtuple

State = namedtuple('State', ['px', 'py', 'vx', 'vy', 'gx', 'gy', 'theta'])

class Agent(object):
    def __init__(self, node_id, node_type='robot', first_timestep=0, time_step=0.1, vpref=0.4, radius=0.2, history_len=1000):

        self.first_timestep = int(first_timestep)
        self.id = int(node_id)
        self.type = node_type
        self.time_step = time_step
        self.radius = radius
        self.vpref = vpref # speed
        self.history_len = history_len

        self.px = None
        self.py = None

        self.gx = None
        self.gy = None

        self.vx = None
        self.vy = None

        self.theta = None

        self.action = (0., 0.)

        self.state_history = []
        self.futures = None
        self.history=None

    def __len__(self):
        return len(self.state_history)

    def update_history(self, px, py, vx, vy, gx, gy, theta):
        self.state_history.append(State(px, py, vx, vy, gx, gy, theta))
        if len(self.state_history)>self.history_len:
            del self.state_history[0]

    def set_action(self, action):
        self.action = action

    def set_goal(self, gx, gy):
        self.gx = gx
        self.gy = gy

    def set_state(self, px, py, vx, vy, gx, gy, theta):

        self.px = px
        self.py = py

        self.vx = vx
        self.vy = vy

        self.gx = gx
        self.gy = gy
        
        self.theta = theta

    def set_history(self, history):
        self.history = history

    def set_futures(self, futures):
        self.futures = futures

    def get_state(self, ):
        return State(self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.theta)

    def get_state_at(self, t):
        _idx = t - self.first_timestep
        return self.state_history[_idx]

    def get_history(self, history_steps=4):
        history = np.zeros((history_steps, 7))
        for t in range(-1, max(-history_steps, -len(self.state_history))-1, -1):
            history[t] = self.deserialize_state(self.state_history[t])
        return history

    def get_futures(self, future_steps=4):
        futures = np.zeros((future_steps, 7))
        for t in range(future_steps):
            self.futures[t] = self.deserialize_state(self.state_history[t])
        return futures

    def get_history_at(self, t, history_steps=4):
        history = np.zeros((history_steps, 7))
        for i, ts in enumerate(range(t, max(self.first_timestep, t - history_steps), -1)):
            history[history_steps-i-1] = self.deserialize_state(self.get_state_at(ts))
        return history

    def get_futures_at(self, t, future_steps=4):
        futures = np.zeros((future_steps, 7))
        for i, ts in enumerate(range(t+1, min(t+future_steps+1, self.last_timestep))):
            futures[i] = self.deserialize_state(self.get_state_at(ts))
        return futures

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
        '''
        action: (v, w)
        '''
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

    def preferred_vel(self, ):
        goal_vec = np.array((self.gx - self.px, self.gy - self.py))
        norm = np.linalg.norm(goal_vec)
        if norm>1:
            return goal_vec/norm * self.vpref
        else:
            return goal_vec

    def serialize_state(self, s):
        return State(s)

    def deserialize_state(self, s):
        return (s.px, s.py, s.vx, s.vy, s.gx, s.gy, s.theta)

    def reset(self, ):
        self.px = None
        self.py = None

        self.gx = None
        self.gy = None

        self.vx = None
        self.vy = None

        self.theta = None

        self.state_history = []
        self.futures = None
        self.history = None

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
