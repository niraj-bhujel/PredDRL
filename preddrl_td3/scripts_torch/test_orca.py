#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:29:53 2021

@author: loc
"""

#!/usr/bin/env python

import rvo2

sim = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 2)

# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
# a0 = sim.addAgent((0, 0))
# a1 = sim.addAgent((1, 0))
# a2 = sim.addAgent((1, 1))
a3 = sim.addAgent((0, 1), 1.5, 5, 1.5, 2, 0.4, 2, (0, 0))



# Obstacles are also supported.
sim.addObstacle([(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)])
sim.processObstacles()

# sim.setAgentPrefVelocity(a0, (1, 1))
# sim.setAgentPrefVelocity(a1, (-1, 1))
# sim.setAgentPrefVelocity(a2, (-1, -1))
sim.setAgentPrefVelocity(a3, (1, -1))

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

for step in range(20):
    sim.doStep()

    # positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no) for agent_no in (a0, a1, a2, a3)]
    
    position = sim.getAgentPosition(a3)
    # print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '(%5.3f, %5.3f)'%position))

    velocity = sim.getAgentVelocity(a3)
    print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '(%5.3f, %5.3f)'%velocity))
    
    # sim.setAgentPosition(a3, position)
    # sim.setAgentVelocity(a3, velocity)

#%% compare with previous cell
from preddrl_td3.scripts_torch.utils.agent import Agent
from preddrl_td3.scripts_torch.policy.orca import ORCA

robot = Agent()

robot._pos = (0., 1.)
robot._radius = 0.4
robot._vpref = 2.
robot._vel = (0., 0.)

robot_policy = ORCA()
robot_policy.configure({'time_step':1/60, 
                        'neighbor_dist':1.5, 
                        'max_neighbors':5, 
                        'time_horizon':1.5, 
                        'time_horizon_obst':2, 
                        'radius':0.4, 
                        'max_speed':2})

obstacle_pos = [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)]

for step in range(20):
    action, pos = robot_policy.predict(robot, obstacle_pos=obstacle_pos)
    
    robot._pos = pos
    robot._vel = action
    
    print('(%5.3f, %5.3f)'%action)
    # print('(%5.3f, %5.3f)'%pos)
    
#%%
from preddrl_td3.scripts_torch.utils.agent import Agent
from preddrl_td3.scripts_torch.policy.orca import ORCA

robot = Agent()

robot._pos = (0., 0.)
robot._vel = (0., 0.)
robot.update_goal((0., 1.))
robot_policy = ORCA(time_step=0.25)

obstacle_pos = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0), (-2.5, -2.5), (-2.5, 2.5), (2.5, -2.5), (2.5, 2.5), (-5.0, -5.0), (-5.0, 5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 0.0), (5.0, 0.0), (-0.0, -5.0), (-0.0, 5.0)]

for step in range(10):
    action, pos = robot_policy.predict(robot, obstacle_pos=obstacle_pos)
    
    # robot._pos = pos
    # robot._vel = action
    
    print('Act: (%5.3f, %5.3f)'%action)
    # print('Pos: (%5.3f, %5.3f)'%pos)
    
