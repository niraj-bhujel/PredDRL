#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import os

import rospy

import math
import numpy as np

from geometry_msgs.msg import Point, Pose, Quaternion
from rospkg import RosPack


from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelStates, ModelState

from utils.env_utils import euler_to_quaternion

class RespawnPedestrians:
    def __init__(self, ):

        rospack1 = RosPack()
        pkg_path = rospack1.get_path('preddrl_gazebo')
        default_actor_model_file = pkg_path + "/models/actor_model.sdf"
    
        actor_model_file = rospy.get_param('~actor_model_file', default_actor_model_file)
        with open(actor_model_file) as file_xml:
            self.ped_model = file_xml.read()
    
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        self.spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

        rospy.wait_for_service("gazebo/delete_model")
        self.delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        
        self.spawnned_models = []
        
        model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        for model_name in model_states.name:
            if 'pedestrian' in model_name:
                self.delete_model(model_name)

    def respawn(self, ped_states, model_states=None, verbose=0, t=0, reference_frame='world'):


        if model_states is None:
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)

        # print(model_states.name)
        # print(self.spawnned_ped_list)
        current_models = []
        for pid, ped_state in ped_states.items():
            
            model_name = 'pedestrian_' + str(pid)

            px, py, vx, vy, _, _, theta = ped_state

            p = Point(x=px, y=py, z=0.)
            q = euler_to_quaternion([0., 0., theta])

            model_pose = Pose(p, q)
                
            if model_name not in self.spawnned_models:
                if verbose>0:
                    rospy.loginfo("[Frame-%d] Spawning %s"%(t, model_name))

                self.spawn_model(model_name, self.ped_model, "", model_pose, reference_frame)
                
                self.spawnned_models.append(model_name)
                
            else:
                if verbose>0:
                    rospy.loginfo("[Frame-%d] Updating %s"%(t, model_name))
                
                tmp_state = ModelState()
                tmp_state.model_name = model_name
                tmp_state.pose = model_pose
                tmp_state.reference_frame = reference_frame
                
                self.set_model_state(tmp_state)

            current_models.append(model_name)

        # print("spawnned_models:", self.spawnned_models)
        # remove model that are not in the current pedestrian list
        models_to_remove = [model for model in self.spawnned_models if model not in current_models]
        # print("models to remove:", models_to_remove)
        for model_name in models_to_remove:
            if verbose>0:
                rospy.loginfo("[Frame-%d] Deleting %s"%(t, model_name))
            self.delete_model(model_name)
            self.spawnned_models.remove(model_name)

