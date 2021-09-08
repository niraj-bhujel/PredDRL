#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import os
import sys
if './' not in sys.path:
    sys.path.insert(0, './')
# if './preddrl_tracker/scripts' not in sys.path:
#     sys.path.insert(0, '/preddrl_tracker/scripts')

import rospy

import math
import numpy as np

from geometry_msgs.msg import Point, Pose, Quaternion
from rospkg import RosPack

from preddrl_tracker.scripts.pedestrian_state_publisher import prepare_data

from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelStates, ModelState


class RespawnPedestrians:
    def __init__(self, ped_data_path, frame_rate=25, num_peds=10):

        rospack1 = RosPack()
        pkg_path = rospack1.get_path('preddrl_gazebo')
        default_actor_model_file = pkg_path + "/models/actor_model.sdf"
    
        actor_model_file = rospy.get_param('~actor_model_file', default_actor_model_file)
        with open(actor_model_file) as file_xml:
            self.xml_string = file_xml.read()
    
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

        rospy.wait_for_service("gazebo/delete_model")
        self.delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.frames, self.peds_per_frame, self.pedestrians = prepare_data(ped_data_path, frame_rate, num_peds)
        print("Total pedestrians:{}, Total frames:{}".format(len(self.pedestrians), len(self.frames)))
        
        self.spawnned_ped_list = []
        
    def respawn(self, t, model_states=None, verbose=0):

        if t>len(self.frames):
            t = t%len(self.frames)

        if model_states is None:
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)

        # print(model_states.name)
        # print(self.spawnned_ped_list)
        for ped in self.pedestrians:
            
            model_name = 'pedestrian_' + str(ped.id)
            
            if t>=ped.first_timestep and t<=ped.last_timestep:            
                p, v, q, r = ped.states_at(t)
    
                model_pose = Pose(Point(x=p[0], y=p[1], z=p[2]), 
                                  Quaternion(w=q[0], x=q[1], y=q[2], z=q[3]))
                    
                if model_name not in model_states.name:
                    if verbose>1:
                        rospy.loginfo("[Frame-%d] Spawning %s"%(t, model_name))

                    self.spawn_model(model_name, self.xml_string, "", model_pose, "world")
                    
                    self.spawnned_ped_list.append(model_name)
                    
                else:
                    if verbose>1:
                        rospy.loginfo("[Frame-%d] Updating %s"%(t, model_name))
                    
                    tmp_state = ModelState()
                    tmp_state.model_name = model_name
                    tmp_state.pose = model_pose
                    tmp_state.reference_frame ="world"
                    
                    self.set_model_state(tmp_state)

            elif model_name in model_states.name:
                if verbose>1:
                    rospy.loginfo("[Frame-%d] Deleting %s"%(t, model_name))
                self.delete_model(model_name)
                if model_name in self.spawnned_ped_list:
                    self.spawnned_ped_list.remove(model_name)

if __name__=='__main__':

    rospy.init_node('ped_spawnner', disable_signals=True)
    ped_spawnner = RespawnPedestrians('./preddrl_tracker/data/crowds_zara01.txt')
    try:
        for step in range(20000):
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)

            ped_spawnner.respawn(step, model_states)
            rospy.sleep(0.5)

    except KeyboardInterrupt:
        print("Clearing models .. ")
        for model_name in ped_spawnner.spawnned_ped_list:
            if model_name in model_states.name:
                ped_spawnner.delete_model(model_name)