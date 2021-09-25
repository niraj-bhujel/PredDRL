#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

'''
This is equivalen to tracker that publish the current position of the agents only. 
'''

import os
import sys
if not './' in sys.path:
    sys.path.insert(0, './')

import math
import numpy as np

from scipy.interpolate import interp1d

import rospy

from std_msgs.msg import Header
from geometry_msgs.msg import *

from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState, SetModelState
from gazebo_msgs.msg import ModelStates, ModelState

from rospkg import RosPack

from preddrl_msgs.msg import AgentStates, AgentState

# from .node import Node
from preddrl_td3.scripts_torch.utils.agent import Agent
        
def angleToQuaternion(theta, angles=True):

    # if not angles:
    theta *= np.pi/180

    R = np.zeros((9))
    R[0] = np.cos(theta)
    R[1] = -np.sin(theta)
    R[3] = np.sin(theta)
    R[4] = np.cos(theta)
    R[8] = 1
    
    w = math.sqrt(R[0]+R[4]+R[8]+1)/2.0
    x = math.sqrt(R[0]-R[4]-R[8]+1)/2.0
    y = math.sqrt(-R[0]+R[4]-R[8]+1)/2.0
    z = math.sqrt(-R[0]-R[4]+R[8]+1)/2.0
    
    q = [w,x,y,z]

    return q

def createMsgHeader(frame_id='world'):

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

        px, py, vx, vy, ax, ay, theta = node.get_states_at(t)

        q = angleToQuaternion(theta)

        state = AgentState()

        state.header = h
        state.id = int(node.id)

        if t>node.last_timestep-1:
            state.type = int(4)

        state.pose.position.x = px
        state.pose.position.y = py
        state.pose.position.z = 0.

        state.pose.orientation.w = q[0]
        state.pose.orientation.x = q[1]
        state.pose.orientation.y = q[2]
        state.pose.orientation.z = q[3]

        state.twist.linear.x = vx
        state.twist.linear.y = vy
        state.twist.linear.z = 0.

        agents.agent_states.append(state)

    return agents

def interpolate(pos, method='quadratic', num_points=1):

    x, y = pos[:, 0], pos[:, 1]

    x_intp = interp1d(np.arange(len(x)), x, kind=method)
    y_intp = interp1d(np.arange(len(y)), y, kind=method)

    points = np.linspace(0, len(x)-1, num_points)

    return np.stack([x_intp(points), y_intp(points)], axis=-1)

def _gradient(x, dt=0.4, axis=0):
    '''
    x - array of shape (T, dim)
    '''

    g =  np.diff(x, axis=axis) / dt
    g = np.concatenate([g[0][None, :], g], axis=axis)
    
    return g

def prepare_data(data_path, target_frame_rate=25, max_peds=20):
    
    target_frame_rate =  np.clip(2.5, target_frame_rate, 25)
    
    data = np.loadtxt(data_path).round(2)

    # convert frame rate into 25 fps from 2.5fps
    data_frames = np.unique(data[:, 0])
    frame_rate_multiplier = target_frame_rate/2.5
    
    # keep the original key frames, sample between frame intervals
    interframes = [np.linspace(0, diff, num=int(frame_rate_multiplier), endpoint=False) for diff in np.diff(data_frames)]
    intp_data_frames = np.concatenate([int_f + key_f for int_f, key_f in zip(interframes, data_frames)] +
                                      [np.linspace(data_frames[-1], data_frames[-1]+10, num=int(frame_rate_multiplier), endpoint=False)])

    ped_nodes = []
    ped_intp_frames = []
    num_ped_considered = 0
    for pid in np.unique(data[:, 1]):

        ped_frames = data[data[:, 1]==pid, 0]        
        num_intp_points = int(len(ped_frames)*frame_rate_multiplier)

        ped_pos = data[data[:, 1]==pid, 2:4]
        
        intp_ped_pos = interpolate(ped_pos, 'quadratic', num_intp_points).round(2)
        
        intp_ped_vel = np.round((intp_ped_pos[1:] - intp_ped_pos[:-1]) * target_frame_rate, 2)

        # intp_ped_vel = np.gradient(intp_ped_pos, 1.0/target_frame_rate, axis=0).round(2)

        start_idx = intp_data_frames.tolist().index(ped_frames[0])
        
        node = Agent(pid, first_timestep=start_idx, time_step=1./target_frame_rate, node_type='pedestrian', history_len=num_intp_points)
        
        for i in range(len(intp_ped_pos)-1):
            
            px, py = intp_ped_pos[i+1][0], intp_ped_pos[i+1][1]
            vx, vy = intp_ped_vel[i][0], intp_ped_vel[i][1]

            gx, gy = intp_ped_pos[-1][0], intp_ped_pos[-1][1]

            theta = round(math.atan2(vy, vx), 2) # radians
            
            # node.update_states(px, v, q, r)
            node.update_history(px, py, vx, vy, gx, gy, theta)

        ped_nodes.append(node)
        
        ped_intp_frames.append(intp_data_frames[start_idx:start_idx+num_intp_points])
        
        num_ped_considered+=1
        
        # break
    
        if num_ped_considered>max_peds:
            break
    
    peds_frames = []
    peds_per_frame = []
    for t, frame in enumerate(intp_data_frames):
        curr_ped = []
        for i, node in enumerate(ped_nodes):

            if t>=node.first_timestep and t<=node.last_timestep:
                
                curr_ped.append(ped_nodes[i].id)
        
        if len(curr_ped)>0:
            peds_frames.append(frame)
            peds_per_frame.append(curr_ped)
        
        
    return ped_nodes, peds_frames, peds_per_frame


#%%
if __name__ == '__main__':
    
    ros_rate = 25.
    rospack = RosPack()
    data_root = rospack.get_path('preddrl_tracker')
    # data_root = rospy.myargv(argv=sys.argv)[0]
    # data_path = data_root + '/crowds_zara01.txt'

    # data_root = '/home/loc/peddrl_ws/src'
    data_path = data_root + '/data/crowds_zara01.txt'
    print('Preparing data from: ', data_path)
    ped_nodes, frames, peds_per_frame = prepare_data(data_path, target_frame_rate=ros_rate)

    # prepare gazebo plugin
    rospy.init_node("spawn_preddrl_agents", anonymous=True, disable_signals=True)

    rospack1 = RosPack()
    pkg_path = rospack1.get_path('preddrl_gazebo')
    default_actor_model_file = pkg_path + "/models/actor_model.sdf"

    actor_model_file = rospy.get_param('~actor_model_file', default_actor_model_file)
    file_xml = open(actor_model_file)
    xml_string = file_xml.read()

    print("Waiting for gazebo spawn_sdf_model services...")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
    print("service: spawn_sdf_model is available ....")

    print("Waiting for gazebo delete_model services...")
    rospy.wait_for_service("gazebo/delete_model")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)

    print("Waiting for gazebo delete_model services...")
    rospy.wait_for_service("/gazebo/set_model_state")
    set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    # print('Inititating pedestrain state publisher node ...')
    r = rospy.Rate(ros_rate, reset=True)
    state_pub = rospy.Publisher('/preddrl_tracker/ped_states', AgentStates, queue_size=100)
    print('Publishing pedestrian states for {} frames'.format(len(frames)))
    
    t = 0
    actors_id_list = []

    while True:

        try:

            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)

            # get pids at current time
            curr_ped_ids = peds_per_frame[t]
            curr_ped_nodes = [node for node in ped_nodes if node.id in curr_ped_ids]
            
            # print(curr_ped_nodes[0].states_at(t))
            actors = create_actor_msg(curr_ped_nodes, t)
            state_pub.publish(actors)
                
            # print(actors.agent_states[0])
            for actor in actors.agent_states:
                
                actor_id = 'pedestrian_' + str(actor.id)
                actor_pose = actor.pose
                model_pose = Pose(Point(x= actor_pose.position.x,
                                       y= actor_pose.position.y,
                                       z= actor_pose.position.z),
                                 Quaternion(actor_pose.orientation.x,
                                            actor_pose.orientation.y,
                                            actor_pose.orientation.z,
                                            actor_pose.orientation.w))

                if actor_id not in actors_id_list:
                    rospy.loginfo("[Frame-%d] Spawning %s"%(t, actor_id))
                    spawn_model(actor_id, xml_string, "", model_pose, "world")
                    actors_id_list.append(actor_id)
                    
                elif actor_id in model_states.name:
                    
                    tmp_state = ModelState()
                    tmp_state.model_name = actor_id
                    tmp_state.pose = model_pose
                    tmp_state.reference_frame ="world"
                    
                    set_model_state(tmp_state)

                if actor.type==int(4) and actor_id in model_states.name:
                    
                    rospy.loginfo("[Frame-%d] Deleting %s"%(t, actor_id))
                    resp = delete_model(actor_id)
                    actors_id_list.remove(actor_id)

            if t>=len(frames)-1:
            # if t>100:
                rospy.loginfo('[Frame-%d] Resetting frame to 0. '%(t))

                [delete_model(actor_id) for actor_id in actors_id_list]
                t = 0
                actors_id_list = []

            else:
                t += 1

            # rospy.sleep(0.5) # this doen't work well in python2
            rospy.sleep(1/ros_rate) # this doen't work well in python2
            # r.sleep() # turn of use_sim_time if r.sleep() doesn't work
            
        except KeyboardInterrupt:
            print('Closing down .. ')
            # delete all model at exit
            
            try:
                model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
            except rospy.ROSException:
                rospy.logerr('ModelStates timeout')
                raise ValueError 

            print('Deleting existing pedestrians models from', model_states.name)
            [delete_model(actor_id) for actor_id in actors_id_list if actor_id in model_states.name]  
            
            break
        
        except Exception as e:
            print(e)