#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('./')

import math
import numpy as np

from scipy.interpolate import interp1d
from pyquaternion import Quaternion

from node import Node
from scene import Scene

import rospy

from std_msgs.msg import Header
from geometry_msgs.msg import *

from gazebo_msgs.srv import SpawnModel, DeleteModel

from rospkg import RosPack

from preddrl_msgs.msg import AgentStates, AgentState

        
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

        p, v, q, r = node.states_at(t)

        state = AgentState()

        state.header = h
        state.id = int(node.id)

        if t>node.last_timestep-1:
            state.type = int(4)
        else:
            state.type = 0

        state.pose.position.x = p[0]
        state.pose.position.y = p[1]
        state.pose.position.z = p[2]

        state.pose.orientation.w = q[0]
        state.pose.orientation.x = q[1]
        state.pose.orientation.y = q[2]
        state.pose.orientation.z = q[3]

        state.twist.linear.x = v[0]
        state.twist.linear.y = v[1]
        state.twist.linear.z = v[2]
        
        state.twist.angular.x = r[0]
        state.twist.angular.y = r[1]
        state.twist.angular.x = r[2]

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

def prepare_data(data_path, target_frame_rate=25):
    print('Preparing data .. ')
    target_frame_rate =  min(target_frame_rate, 25)
    
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

    for pid in np.unique(data[:, 1]):

        ped_frames = data[data[:, 1]==pid, 0]        
        num_intp_points = int(len(ped_frames)*frame_rate_multiplier)

        ped_pos = data[data[:, 1]==pid, 2:4]
        
        intp_ped_pos = interpolate(ped_pos, 'quadratic', num_intp_points)

        intp_ped_vel = np.gradient(intp_ped_pos, 1.0/target_frame_rate, axis=0).round(2)
        # ped_acc = np.gradient(ped_vel, 1.0/target_frame_rate, axis=0).round(2)

        start_idx = intp_data_frames.tolist().index(ped_frames[0])

        
        node = Node(pid, start_idx, node_type='pedestrian', max_len=num_intp_points, frame_rate=target_frame_rate)
        
        for i in range(len(intp_ped_pos)):
            
            p = [intp_ped_pos[i][0], intp_ped_pos[i][1], 0.]
            v = [intp_ped_vel[i][0], intp_ped_vel[i][1], 0.]
            
            theta = math.atan2(v[1], v[0]) # radians
            q = angleToQuaternion(theta)
            
            r = [0., 0., 0.]
            
            node.update_states(p, v, q, r)
            
        # break
            
        ped_nodes.append(node)
        
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
    data_root = '/home/loc/peddrl_ws/src'
    data_path = '/preddrl_tracker/data/crowds_zara01.txt'
    frames, peds_per_frame, ped_nodes = prepare_data(data_root + data_path, target_frame_rate=ros_rate)

    # prepare gazebo plugin
    rospy.init_node("spawn_preddrl_agents", anonymous=True, disable_signals=True)

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
    r = rospy.Rate(ros_rate)
    print('Publishing pedestrian states ...')
    state_pub = rospy.Publisher('/preddrl_tracker/ped_states', AgentStates, queue_size=10)
    t = 0
    
    actors_id_list = []
    while True:
        try:
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
                    rospy.loginfo("[Frame-%d] Spawning model: actor_id = %s"%(t, actor_id))
                    spawn_model(actor_id, xml_string, "", model_pose, "world")
                    actors_id_list.append(actor_id)

                if actor.type==int(4):
                    actors_id_list.remove(actor_id)
                    delete_model(actor_id)

            # if t>=len(frames)-1:
            if t>100:
                rospy.loginfo('[Frame-%d] Resetting frame to 0. '%(t))
                [delete_model(actor_id) for actor_id in actors_id_list]
                t = 0
                actors_id_list = []

            else:
                t += 1

            # rospy.sleep(1/ros_rate) # this doen't work well in python2
            r.sleep() # turn of use_sim_time if r.sleep() doesn't work
            
        except KeyboardInterrupt:
            print('Closing down .. ')
            # delete all model at exit
            # print('deleting existing actor models')
            # [delete_model(actor_id) for actor_id in actors_id_list]  
            break         







