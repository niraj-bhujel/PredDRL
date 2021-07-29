'''
 Class to represent a single pedestrian. 
'''

class Node(object):
    def __init__(self, data, first_timestep, pid, dt=0.4):
        self.data = data
        self.first_timestep = first_timestep
        self.pid = pid

    def points_at(self, t):
        return self.data[t-self.first_timestep]

    def history_points_at(self, t):

        pass

    @property
    def timesteps(self):
        return self.data.shape[0]

    @property
    def last_timestep(self):
        return self.first_timestep + self.timesteps - 1