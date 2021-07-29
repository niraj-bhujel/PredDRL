import sys
# sys.path.append('../../preddrl_msgs/')

import math
import numpy as np
import rospy

from preddrl_msgs.msg import AgentStates, AgentState
from std_msgs.msg import Header
from geometry_msgs.msg import *

from gazebo_msgs.srv import SpawnModel, DeleteModel

from rospkg import RosPack

from pyquaternion import Quaternion

from node import Node
from scene import Scene

        
def angleToQuaternion(theta, degrees=True):

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

def createMsgHeader(frame_id='odom'):

    h = Header()
    h.stamp =  rospy.Time.now()
    h.frame_id = frame_id

    return h


def create_actor_msg(nodes, t):

    # print(t)
    agents = AgentStates()
    h = createMsgHeader()
    agents.header = h

    for node in nodes:

        x, y, vx, vy, ax, ay = node.points_at(t)

        print(t, node.pid, x, y, vx, vy, ax, ay)

        theta = math.atan2(vy, vx) # radians

        q = angleToQuaternion(theta, degrees=False)

        state = AgentState()

        state.header = h
        state.id = int(node.pid)

        if t>node.last_timestep-1:
            state.type = int(4)
        else:
            state.type = 0

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

        agents.agent_states.append(state)

    return agents

def prepare_data(data_path, frame_rate=2.5):
    print('Preparing data .. ')
    
    data = np.loadtxt(data_path)
    data = data.round(3)

    frames = np.unique(data[:, 0])
    peds_per_frame = []
    for frame in frames:
        peds_per_frame.append(data[data[:, 0]==frame, 1])

    ped_nodes = []

    for pid in np.unique(data[:, 1]):
        # ped_states = np.full((len(frames), 6), np.nan)

        ped_frames = data[data[:, 1]==pid, 0]

        # ped_frame_idx = np.amax(frames[:, None] == ped_frames[None, :], axis=-1)
        start_idx = frames.tolist().index(ped_frames[0])

        ped_pos = data[data[:, 1]==pid, 2:4]
        ped_vel = np.gradient(ped_pos, 1/frame_rate, axis=0).round(2)
        ped_acc = np.gradient(ped_vel, 1/frame_rate, axis=0).round(2)

        # ped_states[start_idx:start_idx+len(ped_frames)] = np.concatenate([ped_pos, ped_vel, ped_acc], axis=-1)

        ped_states = np.concatenate([ped_pos, ped_vel, ped_acc], axis=-1)

        ped_nodes.append(Node(ped_states, start_idx, pid))

    return frames, peds_per_frame, ped_nodes

if __name__ == '__main__':
    

    data_rate = 2.5

    frames, peds_per_frame, ped_nodes = prepare_data('crowds_zara01.txt', data_rate)

    # prepare gazebo plugin
    rospy.init_node("spawn_preddrl_agents")

    rospack1 = RosPack()
    pkg_path = rospack1.get_path('preddrl_gazebo_plugin')
    default_actor_model_file = pkg_path + "/models/actor_model.sdf"

    actor_model_file = rospy.get_param('~actor_model_file', default_actor_model_file)
    file_xml = open(actor_model_file)
    xml_string = file_xml.read()

    print("Waiting for gazebo spawn_sdf_model services...")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
    print("service: spawn_sdf_model is available ....")

    print("Waiting for gazebo delete_model services...")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)

    # print('Inititating pedestrain state publisher node ...')
    # rospy.init_node('pedestrain_states_publisher', anonymous=True)

    print('Publishing pedestrian states ...')
    state_pub = rospy.Publisher('/preddrl_tracker/ped_states', AgentStates, queue_size=10)
    t = 0

    actors_id_list = []
    while not rospy.is_shutdown():

        # get pids at current time
        curr_ped_ids = peds_per_frame[t]
        curr_ped_nodes = [node for node in ped_nodes if node.pid in curr_ped_ids]

        actors = create_actor_msg(curr_ped_nodes, t)
        state_pub.publish(actors)

        for actor in actors.agent_states:
            actor_id = str( actor.id)
            actor_pose = actor.pose
            model_pose = Pose(Point(x= actor_pose.position.x,
                                   y= actor_pose.position.y,
                                   z= actor_pose.position.z),
                             Quaternion(actor_pose.orientation.x,
                                        actor_pose.orientation.y,
                                        actor_pose.orientation.z,
                                        actor_pose.orientation.w) )

            if actor_id not in actors_id_list:
                rospy.loginfo("Spawning model: actor_id = %s", actor_id)
                spawn_model(actor_id, xml_string, "", model_pose, "world")
                actors_id_list.append(actor_id)

            if actor.type==int(4):
                delete_model(actor_id)

        if t>=len(frames)-1:
            t = 0
        else:
            t += 1

        # must turn of use_sim_time for r.sleep() to work
        rospy.sleep(1/data_rate)




