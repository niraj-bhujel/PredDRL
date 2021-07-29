import sys
# sys.path.append('../../preddrl_msgs/')

import math
import numpy as np
import rospy

from preddrl_msgs.msg import AgentStates, AgentState
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3

from pyquaternion import Quaternion



def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class PED_STATES_PUBLISHER(object):
    def __init__(self, queue_size=10):

        self.state_pub = rospy.Publisher('/preddrl_tracker/ped_states', AgentStates, queue_size=queue_size)
        
    def angleToQuaternion(self, theta, degrees=True):

        if not degrees:
            theta *= 180/np.pi

        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)
        R[1, 1] = np.cos(theta)
        R[2, 2] = 1

        q = Quaternion(matrix=R)

        return q

    def createMsgHeader(self, frame_id='odom'):

        h = Header()
        h.stamp =  rospy.Time.now()
        h.frame_id = frame_id

        return h

    
    def publish_states(self, ped_states, t=0, max_angle_update=0.1):

        # print(t)
        curr_agent_states = AgentStates()
        h = self.createMsgHeader()
        curr_agent_states.header = h

        for pid, pstate in ped_states.items():

            x, y, vx, vy, ax, ay = pstate

            theta = math.atan2(vy, vx) # radians

            q = self.angleToQuaternion(theta, degrees=False)

            state = AgentState()

            state.header = h
            state.id = int(pid)
            state.type = 1

            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.position.z = 0.0

            state.pose.orientation.x = q.x
            state.pose.orientation.y = q.y
            state.pose.orientation.z = q.z
            state.pose.orientation.w = q.w

            state.twist.linear.x = vx
            state.twist.linear.y = vy
            state.twist.linear.z = 0

            curr_agent_states.agent_states.append(state)

        self.state_pub.publish(curr_agent_states)


if __name__ == '__main__':

    data_rate = 2.5

    data = read_file('../data/crowds_zara01.txt')

    frames = np.unique(data[:, 0])
    peds_per_frame = []
    for frame in frames:
        peds_per_frame.append(data[data[:, 0]==frame, 1])

    peds_data = {}
    for pid in np.unique(data[:, 1]):
        ped_state = np.full((len(frames), 6), np.nan)

        ped_frames = data[data[:, 1]==pid, 0]

        ped_frame_idx = np.amax(frames[:, None] == ped_frames[None, :], axis=-1)

        ped_pos = data[data[:, 1]==pid, 2:4]
        ped_vel = np.gradient(ped_pos, 1/data_rate, axis=0)
        ped_acc = np.gradient(ped_vel, 1/data_rate, axis=0)

        ped_state[ped_frame_idx] = np.concatenate([ped_pos, ped_vel, ped_acc], axis=-1)

        peds_data[pid] = ped_state

    print('Inititating pedestrain state publisher node ...')
    rospy.init_node('pedestrain_states_publisher', anonymous=True)
    
    ped_state_publisher = PED_STATES_PUBLISHER()

    print('Publishing pedestrian states ...')
    t = 0
    while not rospy.is_shutdown():

        # get date at current time
        curr_peds = peds_per_frame[t]
        curr_states = {pid:peds_data[pid][t] for pid in curr_peds}
        
        print(t, curr_peds)
        ped_state_publisher.publish_states(curr_states, t)

        if t>=len(frames)-1:
            t = 0
        else:
            t += 1

        # must turn of use_sim_time for r.sleep() to work
        rospy.sleep(1/data_rate)




