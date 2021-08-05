import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose


class Respawn():
    def __init__(self):
        # self.modelPath = os.path.dirname(os.path.realpath(__file__))
        # self.modelPath = '/home/ros_admin/tf2rl_turtlebot3-master/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf'
        # self.f = open(self.modelPath, 'r')
        # self.model = self.f.read()

        # self.stage = rospy.get_param('/stage_number')
        # self.stage = 2
        self.stage = 6 # added by niraj
        self.goal_position = Pose()
        # self.init_goal_x = -0.586480
        # self.init_goal_y = 4.857300
        self.init_goal_x = 0#1.5#0.5 1.5 0
        self.init_goal_y = 1#0.2#-1.5 0 -1
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position. y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.test_index = 0
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False, # nb-> repeating flag  
        self.index = 0

        # load goal model
        modelPath = './preddrl_turtlebot/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf'
        with open(modelPath, 'r') as f:
            self.model = f.read()

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self): # nb-> this function should be respawnGoalModel ??
        # print(self.check_model)
        # while True:
        if not self.check_model:
            # rospy.loginfo('Waiting for service spawn_sdf_model')
            rospy.wait_for_service('gazebo/spawn_sdf_model')
            # rospy.loginfo('gazebo/spawn_sdf_model available')

            spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
            rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                          self.goal_position.position.y)
            #     break
            # else:
            #     pass

    def deleteModel(self):
        # while True:
        if self.check_model:
            rospy.wait_for_service('gazebo/delete_model')
            del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
            del_model_prox(self.modelName)
            #     break
            # else:
            #     pass

    def getPosition(self, position_check=False, delete=False, test=False):
        # print(self.stage, position_check, delete, test)
        # goal_xy_list = {
        #         [1.5,2.5],[2.5,-0.5]
        #     }
        if delete:
            self.deleteModel()

        if test or self.stage == -1:
            #第一幅地图
            # goal_x = [0., -1.0, 0., 1.]
            # goal_y = [-1.5, 0., 1.5, 0]
            #第二幅地图静态
            # goal_x = [3., 1.5, -1., 1.5]
            # goal_y = [0, 4, 3, 0]
            #第二幅地图动态
            # goal_x = [0.0,-2.0,-4.5,-5.0,-6.5,-8.0,-9.0,-9.0,-5.0,-5.0,-2.0,-2.0, 0.0]#-5
            # goal_y = [3.0, 4.0, 3.0, 1.0, 4.3, 6.0, 8.5, 7.0, 7.0, 5.0, 3.5, 1.0, 0.0]#8.3

            # goal_x = [3.5, 0 ]#-5
            # goal_y = [4.5, 0]

            goal_x = [3.5, 0  , 4, 0]#-5
            goal_y = [4.5, 4.5, 0, 0]
            # goal_x = [3, -3, 0, 3., -3., 0]
            # goal_y = [-3., -3., 0, 3., 3., 0]
            # goal_x = [3.5, 0  , 4, 0,  0, 4, 3.5,0]#-5
            # goal_y = [4.5, 4.5, 0, 0,4.5, 0, 4.5,0]

            # goal_x = [0 ]#-5 
            # goal_y = [5]
            # goal_x = [1.5]#-5 
            # goal_y = [0]
            # goal_x = [3, -3, 0, 3., -3., 0]
            # goal_y = [-3., -3., 0, 3., 3., 0]
            # goal_x = [-5.523510, -6.815110, -4]
            # goal_y = [4.138770, 5.614350, 7]
            if self.test_index == len(goal_x):
                print("end:", time.time())
            self.goal_position.position.x = goal_x[self.test_index]
            self.goal_position.position.y = goal_y[self.test_index]
            # self.goal_position.position.x, self.goal_position.position.y = random.choice(goal_xy_list)
            self.test_index += 1

        elif self.stage == 0:
            while position_check:
                goal_x_list = [0,0]
                goal_y_list = [5,0]

                # goal_x_list = [0,3,4,3,1]
                # goal_y_list = [5,4,0,1.5,2]
                # goal_x_list = [2., 1., 2.5, -2., -3., 2., -2., 0., 1., -1., -3.5, -1., 3.5]
                # goal_y_list = [0., -1., 2.5, 0., 2., -3.5, -2., -1., 1., 2.5, -3.5, 1.3, 1.5]

                self.index = random.randrange(0, len(goal_x_list))
                # print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        elif self.stage == 2 or self.stage == 3:
            while position_check:
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        elif self.stage == 1:
            while position_check:
                goal_x_list = [1.5, 2.5, -1.5, -0.5, 3.7, 3.5, 1.5, 0., 0.5, 0.5, 3.5, 2.5, 3.5]
                goal_y_list = [2.5, -0.5, -0.5, 2.5, 3, 1., 4., 4.5, 5., 2.5, -0.5, 1.3, 3.5]

                self.index = random.randrange(0, 13)
                # print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        elif self.stage == 4:
             while position_check:
                goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 0.5, 0, -0.1, -2]
                goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.8, -1, 1.6, -0.8]

                self.index = random.randrange(0, 12)
                # print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]
        elif self.stage == 5:
             while position_check:
                
                goal_xy_list = [
                [-1.5, 0.5], [-1.5, 1.5], [-0.5, 0.5], [-0.5, 1.5],
                [0.5, -0.5], [0.5, -1.5], [2.5, -0.5], [2.5, 0.5],
                [5.5,-1.5], [5.5,-0.5], [5.5,0.5], [5.5,1.5]
                ]
                self.index = random.randrange(0, 12)
                # print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_xy_list[self.index][0]
                self.goal_position.position.y = goal_xy_list[self.index][1]
        else:
            while position_check:
                # train_env_1
                # goal_x_list = [0, 1, 1, -1, -1, -1, -2.5, 0., 2.5, 2.5, -1.5, 2., 0.5, 1.0, -1.0, 1.5, -1.5]
                # goal_y_list = [2., 1, -1, -1, 1, 1, -1.5, 3.5, 3.5, -1.5, 2, 2., 0.5, 3.5,  3.5, 4.5, 4.5]

                # train_env_2
                goal_x_list = [2., 1., 2.5, -2., -3., 2., -2., 0., 1., -1., -3.5, -1., 3.5]
                goal_y_list = [0., -1., 2.5, 0., 2., -3.5, -2., -1., 1., 2.5, -3.5, 1.3, 1.5]
                self.index = random.randrange(0, len(goal_x_list))
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        # time.sleep(0.5)
        # rospy.loginfo('Respawnning model')
        if not self.check_model:
            self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
    # def getPosition(self, position_check=False, delete=False, test=False):
    #     if delete:
    #         self.deleteModel()
    #
    #
    #
    #     # time.sleep(0.5)
    #     self.respawnModel()
    #
    #     self.goal_position.position.x = self.init_goal_x
    #     self.goal_position.position.y = self.init_goal_y
    #
    #     return self.goal_position.position.x, self.goal_position.position.y