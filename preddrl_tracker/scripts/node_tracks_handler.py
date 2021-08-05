import sys
import numpy as np

from node import Node

import rospy

from gazebo_msgs.msg import ModelStates

'''
This node is responsible to maintain past trajectory of each nodes present in the gazebo environment. 
Each node is a class object of attribute pedestrian, robot or static objects in the environment. 
It subscribes to the ModelStates, and create a Node object for each unique model. 
Finally, it will publish the tracks for each nodes. 
'''
def vector3_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])

def quat_to_numpy(msg):
	return np.array([msg.x, msg.y, msg.z, msg.w])

def point_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])
    
class NodeTracksHandler(object):

    def __init__(self, rate=10):
        # self.track_pub = rospy.Publisher('/tracking/node_tracks', AgentTracks, queue_size=10)
        self.nodes = {}
        self.rate = rate

    def callback(self, model_states):
        self.model_states = model_states
        print(model_states)
        
        _N = len(model_states.name)
        
        for i in range(_N):
            m_name = model_states.name[i]
            
            if m_name=='ground_plane':
                continue
            
            m_pos = point_to_numpy(model_states.pose[i].position)
            m_quat = quat_to_numpy(model_states.pose[i].orientation)
                        
            m_vel = vector3_to_numpy(model_states.twist[i].linear)
            m_rot = vector3_to_numpy(model_states.twist[i].angular)
            
            if m_name in self.nodes.keys():
                node = self.nodes[m_name]
                node.update(m_pos, m_vel, m_quat, m_rot)
                
            else:
                node = Node(node_id=str(m_name), frame_rate=self.rate)
                node.update(m_pos, m_vel, m_quat, m_rot)
        
            
            node_history = node.get_history(history_timesteps=8, frame_rate=2.5)
            
            # prepare message for tracks
            
        # pass

if __name__ == '__main__':
    

    rospy.init_node('node_tracks', anonymous=True, disable_signals=True)
    nh = NodeTracksHandler()

    print('Subscribing gazebo/model_states ...')
    rospy.Subscriber('/gazebo/model_states', ModelStates, nh.callback)
    
    rospy.spin()
    
    # try:
    #     rospy.spin()

    # except KeyboardInterrupt:
        
    #     print("Keyboard interrupt ...")
