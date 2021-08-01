import sys
# sys.path.append('../../preddrl_msgs/')

import math
import numpy as np
from scipy.interpolate import interp1d
import rospy

from preddrl_msgs.msg import AgentStates, AgentState
from std_msgs.msg import Header
from geometry_msgs.msg import *

from gazebo_msgs.srv import SpawnModel, DeleteModel

from rospkg import RosPack

from pyquaternion import Quaternion

from node import Node
from scene import Scene

        
def angleToQuaternion(theta, radians=True):

    if not radians:
        theta *= np.pi/180

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

        print(t, node.id, x, y, vx, vy, ax, ay)

        theta = math.atan2(vy, vx) # radians

        q = angleToQuaternion(theta)

        state = AgentState()

        state.header = h
        state.id = int(node.id)

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

def _interpolate(pos, method='quadratic', num_points=1):

    x, y = pos[:, 0], pos[:, 1]

    x_intp = interp1d(np.arange(len(x)), x, kind=method)
    y_intp = interp1d(np.arange(len(y)), y, kind=method)

    points = np.linspace(0, len(x)-1, num_points)

    return np.stack([x_intp(points), y_intp(points)], axis=-1)

def prepare_data(data_path, target_frame_rate=25):
    print('Preparing data .. ')
    target_frame_rate = max(target_frame_rate, 25)
    
    data = np.loadtxt(data_path)
    data = data.round(3)

    # convert frame rate into 25 fps from 2.5fps
    data_frames = np.unique(data[:, 0])
    frame_rate_multiplier = target_frame_rate/2.5
    
    # keep the original key frames, sample between frame intervals
    interframes = [np.linspace(0, diff, num=frame_rate_multiplier, endpoint=False) for diff in np.diff(data_frames)]
    intp_data_frames = np.concatenate([int_f + key_f for int_f, key_f in zip(interframes, data_frames)] +
                                      [np.linspace(data_frames[-1], data_frames[-1]+10, num=frame_rate_multiplier, endpoint=False)])

    ped_nodes = []
    ped_intp_frames = []

    for pid in np.unique(data[:, 1]):

        ped_frames = data[data[:, 1]==pid, 0]        
        num_intp_points = int(len(ped_frames)*frame_rate_multiplier)

        ped_pos = _interpolate(data[data[:, 1]==pid, 2:4], 'quadratic', num_intp_points)

        ped_vel = np.gradient(ped_pos, 1/target_frame_rate, axis=0).round(2)
        ped_acc = np.gradient(ped_vel, 1/target_frame_rate, axis=0).round(2)

        start_idx = intp_data_frames.tolist().index(ped_frames[0])

        ped_states = np.concatenate([ped_pos, ped_vel, ped_acc], axis=-1)

        ped_nodes.append(Node(ped_states, start_idx, pid))
        
        ped_intp_frames.append(intp_data_frames[start_idx:start_idx+num_intp_points])
    
    peds_per_frame = []
    for t, frame in enumerate(intp_data_frames):
        curr_ped = []
        for i, node in enumerate(ped_nodes):

            if t>=node.first_timestep and t<=node.last_timestep:
                
                curr_ped.append(ped_nodes[i].id)
                
        peds_per_frame.append(curr_ped)
        
        
    return intp_data_frames, peds_per_frame, ped_nodes

#%%
if __name__ == '__main__':
    
    ros_rate = 10
    
    frames, peds_per_frame, ped_nodes = prepare_data('./preddrl_tracker/data/crowds_zara01.txt', target_frame_rate=ros_rate)

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
        curr_ped_nodes = [node for node in ped_nodes if node.id in curr_ped_ids]

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
        rospy.sleep(1/ros_rate)




