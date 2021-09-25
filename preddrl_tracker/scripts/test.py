#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 23:30:04 2021

@author: loc
"""
#%% class method overwriding
class Base(object):
    
    def __init__(self, x=1, y=1):
        self.x = x
        self.y=y
        self(1.5, 1.5)
    def func_add(self,):
        self.add()
        
    def add(self, x=2, y=2):
        print(x+y)
        
class Child(Base):
    def __init__(self, x=1, y=1):
        super().__init__()
        # self.x = x
        # self.y = y
    def add(self, x=5, y=5):
        print(x + y)
        
        
base_class = Base()
child_class = Child()
child_class.func_add()



#%%
future_steps = 10
t_list = []
for global_step in range(1000):

    if global_step>len(peds_frames):
        t = global_step%len(peds_frames)
    else:
        t = global_step
    
    t_list.append(t)
    curr_peds = [ped for ped in ped_nodes if t>=ped.first_timestep and t<ped.last_timestep]
    
    ped_states = {}
    # update action of the current peds
    for ped in curr_peds:
        # ground truth state
        state = ped.get_state_at(t) 
        ped.set_state(state.px, state.py, state.vx, state.vy, state.gx, state.gy, state.theta)
    
        ped_futures = np.zeros((future_steps, 2))
        for i, ts in enumerate(range(t, min(t+future_steps, ped.last_timestep))):
            _s = ped.get_state_at(ts)
            ped_futures[i] = (_s.px, _s.py)
        ped.set_futures(ped_futures)
        print('future:', ped.futures)
    
    if t == 615:
        sys.exit()
    # break
            