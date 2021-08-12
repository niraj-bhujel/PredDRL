class Scene(object):
    def __init__(self, frames, dt=0.4):
        self.nodes = []
        self.dt = dt
        self.frames = frames

    def get_nodes_at_t(self, t, min_history_timesteps=0):
        
        pass

    def get_node_by_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node