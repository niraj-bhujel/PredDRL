import rospy
import random
import time
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose


class Respawn():
    def __init__(self, stage=0):

        self.stage = stage
        self.goal_position = Pose()
        self.init_goal_x = 0#1.5#0.5 1.5 0
        self.init_goal_y = 1#0.2#-1.5 0 -1
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position. y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1_ = -0.6, -0.6
        self.obstacle_2_ = -0.6, 0.6
        self.obstacle_3_ = 0.6, -0.6
        self.obstacle_4_ = 0.6, 0.6

        self.obstacle_1 = -1, -1
        self.obstacle_2 = -1, 1
        self.obstacle_3 = 1, -1
        self.obstacle_4 = 1, 1
        self.obstacle_5 = -2.5, -2.5
        self.obstacle_6 = -2.5, 2.5
        self.obstacle_7 = 2.5, -2.5
        self.obstacle_8 = 2.5, 2.5
        self.obstacle_9 = -2.5, 0
        self.obstacle_10 = 2.5, 0
        self.obstacle_11 = 0, -2.5
        self.obstacle_12 = 0, 2.5
        self.obstacle_13 = -5, -5
        self.obstacle_14 = -5, 5
        self.obstacle_15 = 5, -5
        self.obstacle_16 = 5, 5
        self.obstacle_17 = 5, 0
        self.obstacle_18 = -5, 0
        self.obstacle_19 = 0, -5
        self.obstacle_20 = 0, 5
        

        self.test_index = 0
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False, # nb-> repeating flag , flag used to show if model goal already exists in the env
        self.index = 0

        # load goal model
        modelPath = './preddrl_turtlebot/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf'
        with open(modelPath, 'r') as f:
            self.model = f.read()

    def checkModel(self, model):
        self.check_model = False
        self.num_existing_model = 0
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True
                self.num_existing_model+=1

    def respawnModel(self): # nb-> this function should be respawnGoalModel ??

        rospy.wait_for_service('gazebo/spawn_sdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
        # rospy.loginfo("New goal ( %.1f, %.1f) respawnned ", self.goal_position.position.x, self.goal_position.position.y)


    def deleteModel(self):

        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        del_model_prox(self.modelName)
        # rospy.loginfo("Goal ( %.1f, %.1f) deleted ", self.goal_position.position.x, self.goal_position.position.y)

    # def getPosition(self, position_check=False, delete=False, test=False):
    def getPosition(self, position_check=False, test=False): # niraj-> removed delete flag
        # print(self.stage, position_check, delete, test)

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

            goal_x = [3.5, 0  , 4, 0]
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

        elif self.stage == 2 or self.stage == 3:
            while position_check:
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0
                if abs(goal_x - self.obstacle_1_[0]) <= 0.4 and abs(goal_y - self.obstacle_1_[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2_[0]) <= 0.4 and abs(goal_y - self.obstacle_2_[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3_[0]) <= 0.4 and abs(goal_y - self.obstacle_3_[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4_[0]) <= 0.4 and abs(goal_y - self.obstacle_4_[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

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
                goal_x = random.randrange(-35, 35) / 10.0
                goal_y = random.randrange(-35, 35) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                
                elif abs(goal_x - self.obstacle_5[0]) <= 0.4 and abs(goal_y - self.obstacle_5[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_6[0]) <= 0.4 and abs(goal_y - self.obstacle_6[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_7[0]) <= 0.4 and abs(goal_y - self.obstacle_7[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_8[0]) <= 0.4 and abs(goal_y - self.obstacle_8[1]) <= 0.4:
                    position_check = True
            
                # elif abs(goal_x - self.obstacle_9[0]) <= 0.4 and abs(goal_y - self.obstacle_9[1]) <= 0.4:
                #     position_check = True
                # elif abs(goal_x - self.obstacle_10[0]) <= 0.4 and abs(goal_y - self.obstacle_10[1]) <= 0.4:
                #     position_check = True
                # elif abs(goal_x - self.obstacle_11[0]) <= 0.4 and abs(goal_y - self.obstacle_11[1]) <= 0.4:
                #     position_check = True
                # elif abs(goal_x - self.obstacle_12[0]) <= 0.4 and abs(goal_y - self.obstacle_12[1]) <= 0.4:
                #     position_check = True

                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        # elif self.stage == 5:
        #      while position_check:
                
        #         goal_xy_list = [
        #         [-1.5, 0.5], [-1.5, 1.5], [-0.5, 0.5], [-0.5, 1.5],
        #         [0.5, -0.5], [0.5, -1.5], [2.5, -0.5], [2.5, 0.5],
        #         [5.5,-1.5], [5.5,-0.5], [5.5,0.5], [5.5,1.5]
        #         ]
        #         self.index = random.randrange(0, 12)
        #         # print(self.index, self.last_index)
        #         if self.last_index == self.index:
        #             position_check = True
        #         else:
        #             self.last_index = self.index
        #             position_check = False

        #         self.goal_position.position.x = goal_xy_list[self.index][0]
        #         self.goal_position.position.y = goal_xy_list[self.index][1]

        elif self.stage==6:
            while position_check:
                # train_env_1
                # goal_x_list = [0, 1, 1, -1, -1, -1, -2.5, 0., 2.5, 2.5, -1.5, 2., 0.5, 1.0, -1.0, 1.5, -1.5]
                # goal_y_list = [2., 1, -1, -1, 1, 1, -1.5, 3.5, 3.5, -1.5, 2, 2., 0.5, 3.5,  3.5, 4.5, 4.5]

                # train_env_2
                # goal_x_list = [2., 1., 2.5, -2., -3., 2., -2., 0., 1., -1., -3.5, -1., 3.5]
                # goal_y_list = [0., -1., 2.5, 0., 2., -3.5, -2., -1., 1., 2.5, -3.5, 1.3, 1.5]

                # social_context, data
                goal_x_list = [12.0, 5,0, 8.0, 10.0, 12.0, 14.0]
                goal_y_list = [11.5, 8.3, 4.0, 13.1, 3.2, 8.6]
                self.index = random.randrange(0, len(goal_x_list))
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        elif self.stage == 7:
            # social context
            data_stat = {'x_min': -1, 'x_max': 10, 'y_min': -1, 'y_max': 10}
            x = random.uniform(data_stat['x_min'], data_stat['x_max'])
            y = random.uniform(data_stat['y_min'], data_stat['y_max'])
            self.goal_position.position.x = x
            self.goal_position.position.y = y

        elif self.stage == 10:
            while position_check:
                goal_x = random.randrange(-60, 60) / 10.0
                goal_y = random.randrange(-60, 60) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                
                elif abs(goal_x - self.obstacle_5[0]) <= 0.4 and abs(goal_y - self.obstacle_5[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_6[0]) <= 0.4 and abs(goal_y - self.obstacle_6[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_7[0]) <= 0.4 and abs(goal_y - self.obstacle_7[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_8[0]) <= 0.4 and abs(goal_y - self.obstacle_8[1]) <= 0.4:
                    position_check = True
            
                # elif abs(goal_x - self.obstacle_9[0]) <= 0.4 and abs(goal_y - self.obstacle_9[1]) <= 0.4:
                #     position_check = True
                # elif abs(goal_x - self.obstacle_10[0]) <= 0.4 and abs(goal_y - self.obstacle_10[1]) <= 0.4:
                #     position_check = True
                # elif abs(goal_x - self.obstacle_11[0]) <= 0.4 and abs(goal_y - self.obstacle_11[1]) <= 0.4:
                #     position_check = True
                # elif abs(goal_x - self.obstacle_12[0]) <= 0.4 and abs(goal_y - self.obstacle_12[1]) <= 0.4:
                #     position_check = True

                elif abs(goal_x - self.obstacle_13[0]) <= 0.4 and abs(goal_y - self.obstacle_13[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_14[0]) <= 0.4 and abs(goal_y - self.obstacle_14[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_15[0]) <= 0.4 and abs(goal_y - self.obstacle_15[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_16[0]) <= 0.4 and abs(goal_y - self.obstacle_16[1]) <= 0.4:
                    position_check = True

                elif abs(goal_x - self.obstacle_17[0]) <= 0.4 and abs(goal_y - self.obstacle_17[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_18[0]) <= 0.4 and abs(goal_y - self.obstacle_18[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_19[0]) <= 0.4 and abs(goal_y - self.obstacle_19[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_20[0]) <= 0.4 and abs(goal_y - self.obstacle_20[1]) <= 0.4:
                    position_check = True

                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y


        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y