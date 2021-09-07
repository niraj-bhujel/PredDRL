from __future__ import division
import numpy as np
import rvo2



class ORCA(object):
    def __init__(self, time_step=0.5):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super(ORCA, self).__init__()
        
        
        
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = None
        # self.kinematics = 'holomonic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.2
        self.max_speed = 1.
        self.sim = None
        self.time_step = time_step
        

    def configure(self, config):
        self.time_step = config.get('time_step', self.time_step)
        self.neighbor_dist = config.get('neighbor_dist', self.neighbor_dist)
        self.max_neighbors = config.get('max_neighbors', self.max_neighbors)
        self.time_horizon = config.get('time_horizon', self.time_horizon)
        self.time_horizon_obst = config.get('time_horizon_obst', self.time_horizon_obst)
        self.radius = config.get('radius', self.radius)
        self.max_speed = config.get('max_speed', self.max_speed)
        

    def set_phase(self, phase):
        return

    def predict(self, self_state, humans=[], obstacle_pos=[]):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        # self_state = state.self_state
        # params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        
        if self.sim is not None:
            if self.sim.getNumAgents() != len(humans) + 1:
                del self.sim
                self.sim = None
        
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, 
                                           self.neighbor_dist, 
                                           self.max_neighbors, 
                                           self.time_horizon, 
                                           self.time_horizon_obst,
                                           self.radius, 
                                           self.max_speed)

            self.sim.addAgent(self_state._pos, 
                              self.neighbor_dist, 
                              self.max_neighbors, 
                              self.time_horizon, 
                              self.time_horizon_obst,
                              self_state._radius + 0.01 + self.safety_space,
                              self_state._vpref, 
                              self_state._vel)
            
            if len(humans)>0:
                for human_state in humans:
                    self.sim.addAgent(human_state._pos, 
                                      self.neighbor_dist, 
                                      self.max_neighbors, 
                                      self.time_horizon, 
                                      self.time_horizon_obst,
                                      human_state._radius + 0.01 + self.safety_space,
                                      self.max_speed, 
                                      human_state._vel)
            if len(obstacle_pos)>0:
                self.sim.addObstacle(obstacle_pos)
                self.sim.processObstacles()

        else:
            self.sim.setAgentPosition(0, self_state._pos)
            self.sim.setAgentVelocity(0, self_state._vel)

            if len(humans)>0:
                for i, human_state in enumerate(humans):
                    self.sim.setAgentPosition(i + 1, human_state._pos)
                    self.sim.setAgentVelocity(i + 1, human_state._vel)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        # velocity = np.array((self_state._goal[0] - self_state._pos[0], self_state._goal[1] - self_state._pos[1]))
        # speed = np.linalg.norm(velocity)
        # pref_vel = velocity / speed if speed > 1 else velocity
        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, (1, -1)) # use this during testing
        # self.sim.setAgentPrefVelocity(0, tuple(self_state.preferred_vel))
        
        if len(humans)>0:
            for i, human_state in enumerate(humans):
                # unknown goal position of other humans
                self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        
        action = self.sim.getAgentVelocity(0)
        position = self.sim.getAgentPosition(0)
        
        # self_state._pos = position
        # self_state._vel = action
        # self.last_state = state

        return action, position
    
