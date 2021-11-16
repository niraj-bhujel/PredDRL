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

from preddrl_data.scripts.prepare_data import prepare_data

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



#%%
if __name__ == '__main__':
    
    ros_rate = 25.
    rospack = RosPack()
    data_root = rospack.get_path('preddrl_data')
    # data_root = rospy.myargv(argv=sys.argv)[0]
    data_path = data_root + '/crowds_zara01.txt'

    # data_root = '/home/loc/peddrl_ws/src'
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