import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose


class RespawnGoal():
    def __init__(self, gx=0, gy=0):

        self.name = 'goal'
        self.pose = Pose()
        self.pose.position.x = gx
        self.pose.position. y = gy

        # load goal model
        modelPath = './preddrl_turtlebot/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf'
        with open(modelPath, 'r') as f:
            self.model = f.read()

        # clear existing goal if any, this wil create time out
        model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        if self.name in model_states.name:
            self.deleteGoal()

    def spawnGoal(self,):
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox(self.name, self.model, 'robotos_name_space', self.pose, "world")
        rospy.loginfo("Spawnning Goal ( %.1f, %.1f) ", self.pose.position.x, self.pose.position.y)

    def deleteGoal(self):
        rospy.loginfo("Deleting Goal ( %.1f, %.1f) ", self.pose.position.x, self.pose.position.y)
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        del_model_prox(self.name)

    def setPosition(self, position_check=False): 
        # social context
        data_stat = {'x_min': 1, 'x_max': 14, 'y_min': 3, 'y_max': 9}
        x = random.uniform(data_stat['x_min'], data_stat['x_max'])
        y = random.uniform(data_stat['y_min'], data_stat['y_max'])
        self.pose.position.x = x
        self.pose.position.y = y
        return x, y