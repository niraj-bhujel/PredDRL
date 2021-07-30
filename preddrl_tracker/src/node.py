'''
 Class to represent a single pedestrian. 
'''

class Node(object):
    def __init__(self, data, first_timestep, node_id, node_type='ped'):
        self.data = data
        self.first_timestep = first_timestep
        self.id = node_id
        self.node_type = node_type

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