#!/usr/bin/env python
"""
Created on Mon Dec  2 17:03:34 2019

@author: mahmoud
"""

import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import *
from rospkg import RosPack
from preddrl_msgs.msg  import AgentStates


actors_id_list = []

def actor_poses_callback(actors):
    global actors_id_list 
    for actor in actors.agent_states:
        actor_id = str( actor.id )
        actor_pose = actor.pose
        
        model_pose = Pose(Point(x= actor_pose.position.x,
                               y= actor_pose.position.y,
                               z= actor_pose.position.z),
                         Quaternion(actor_pose.orientation.x,
                                    actor_pose.orientation.y,
                                    actor_pose.orientation.z,
                                    actor_pose.orientation.w) )

        # not working 
        if actor_id not in actors_id_list:
            print("Spawning model: actor_id = %s", actor_id)
            rospy.loginfo("Spawning model: actor_id = %s", actor_id)
            spawn_model(actor_id, xml_string, "", model_pose, "world")
            actors_id_list.append(actor_id)
        else:
            pass

        # if actor.type==int(4):
        #     # delete model
            

    rospy.signal_shutdown("all agents have been spawned !")

if __name__ == '__main__':

    rospy.init_node("spawn_preddrl_agents")

    rospack1 = RosPack()
    pkg_path = rospack1.get_path('preddrl_gazebo_plugin')
    default_actor_model_file = pkg_path + "/models/actor_model.sdf"

    actor_model_file = rospy.get_param('~actor_model_file', default_actor_model_file)
    file_xml = open(actor_model_file)
    xml_string = file_xml.read()

    print("Waiting for gazebo services...")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
    print("service: spawn_sdf_model is available ....")
    
    rospy.Subscriber("/preddrl_tracker/ped_states", AgentStates, actor_poses_callback)

    rospy.spin()
