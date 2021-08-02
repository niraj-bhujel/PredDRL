import rospy
import numpy as np
import math
from gym import spaces
from gym.utils import seeding
from math import pi
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
from .respawnGoal import Respawn


class Env:
    def __init__(self):
        self.goal_x = 1.0
        self.goal_y = 0
        self.inflation_rad = 0.37  # 包含0.17的自身半径
        self.heading = 0
        self.pre_heading = 0
        self.max_v = 0.2
        self.max_w = 1.5
        self.goal_threshold = 0.3#0.4
        self.collision_threshold = 0.15
        self.vel_cmd = [0., 0.]
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.test = False
        self.num_beams = 20  # 激光数
        low = np.array([-2., -2.])
        high = np.array([2., 2.])
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        low = [0.0] * (self.num_beams)
        low.extend([0., -2., -2*pi, 0])  #极坐标
        # low.extend([0., -1.5, -2.0, -2.0,-2.0, -2.0]) #笛卡尔坐标
        high = [3.5] * (self.num_beams)
        high.extend([0.2, 2., 2*pi, 4])
        # high.extend([0.2, 1.5, 2.0, 2.0, 2.0, 2.0])
        self.observation_space = spaces.Box(np.array(low), np.array(high), dtype=np.float32)
#
        self.input_shape = 20
        self.window_size = 3
#        
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        # self.point_goal = rospy.Subscriber('/move_base_simple/goal', Odometry, self.getOdometry)

        self.respawn_goal = Respawn()
        self.past_distance = 0.
#
        # self.preprocessor = HistoryPreprocessor(self.input_shape, history_length=self.window_size)
#        
    def euler_from_quaternion(self, orientation_list):
        x, y, z, w = orientation_list
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

        return r, p, y


    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # goal_distance = round(math.hypot(self.goal_x - self.position.position.x, self.goal_y - self.position.position.y), 2)
        self.past_distance = goal_distance
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = self.euler_from_quaternion(orientation_list)

        # goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        # goal_angle = math.atan2(self.goal_y - self.position.position.y, self.goal_x - self.position.position.x)
        
        inc_y = self.goal_y - self.position.y
        inc_x = self.goal_x - self.position.x
        goal_angle = math.atan2(inc_y, inc_x)

        # if inc_y>0 and inc_x<0:
        #     goal_angle += pi
        # elif inc_y<0 and inc_x<0:
        #     goal_angle -= pi
        
        heading = goal_angle - yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)
        # print(0)
        # print(goal_angle)

    def getState(self, scan):
        scan_range = []
        scan_range_collision = []
        heading = self.heading
        done = False

        # for i in range(0,20):
        #     if scan.ranges[i] == float('Inf'):
        #         scan_range.append(3.5)
        #     elif np.isnan(scan.ranges[i]):
        #         scan_range.append(0)
        #     else:
        #         scan_range.append(scan.ranges[i])

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range_collision.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range_collision.append(0)
            else:
                scan_range_collision.append(scan.ranges[i])

        # obstacle_min_range = round(min(scan_range), 2)
        # obstacle_angle = np.argmin(scan_range)
        if self.collision_threshold > min(scan_range_collision) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # current_distance = round(math.hypot(self.goal_x - self.position.position.x, self.goal_y - self.position.position.y), 2)
        if current_distance < self.goal_threshold:
            self.get_goalbox = True
            
        # print(scan_range_collision)
#        
        state = scan_range_collision+ self.vel_cmd + [heading, current_distance] # 极坐标
#
        # state = scan_range + self.vel_cmd + [self.position.x, self.position.y, self.goal_x, self.goal_y] #笛卡尔坐标
        
        return state, done, self.get_goalbox
       
    def setReward(self, state, done):

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # current_distance = round(math.hypot(self.goal_x - self.position.position.x, self.goal_y - self.position.position.y), 2)
        distance_rate = (self.past_distance - current_distance)
        self.past_distance = current_distance

        if done:
            rospy.loginfo("Collision!!")
            reward = -150
            self.pub_cmd_vel.publish(Twist())

        elif self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True, test=self.test)
            # send_goal=None
            # while send_goal is None:
            #     try:
            #         send_goal = rospy.wait_for_message('/move_base_simple/goal', PoseStamped,timeout=5)
            #     except:
            #         pass
            # self.goal_x, self.goal_y = send_goal.pose.position.x, send_goal.pose.position.y
            # print(send_goal)
            # rospy.loginfo("Goal position : %.1f, %.1f", self.goal_x,
            #                   self.goal_y)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
        else:
            reward = (self.goal_threshold-state[-1]) * 0.1 #- 0.25*abs(state[-3])15*distance_rate
            # reward = 15*distance_rate
        # 增加一层膨胀区域，越靠近障碍物负分越多
        obstacle_min_range = round(min(state[:self.num_beams]), 2)
        if obstacle_min_range < self.inflation_rad:
            # reward += 100.0*(obstacle_min_range - self.inflation_rad)/self.inflation_rad
            reward -= 5.0*(1 - obstacle_min_range/self.inflation_rad)

        return reward

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.pre_heading = self.heading
        vel_cmd = Twist()
        # vel_cmd.linear.x = (action[0]*2.5 + 5.0) / 20.0
        # vel_cmd.linear.x = (action[0]*1.75 + 3.5) / 20.0

        vel_cmd.linear.x = (action[0] + 2.0) / 20.0
        vel_cmd.angular.z = action[1]
        self.vel_cmd = [vel_cmd.linear.x, vel_cmd.angular.z]
        self.pub_cmd_vel.publish(vel_cmd)
        # print(self.vel_cmd)

        # data = None
        # while data is None:
        #     try:
        #         data = rospy.wait_for_message('scan', LaserScan, timeout=5)
        #     except Exception as e:
        #         print(e)

        data = rospy.wait_for_message('scan', LaserScan, timeout=5)

#
        state, done, success = self.getState(data)
        # state, done, success = self.getState(laser)
#
        reward = self.setReward(state, done)

        # 到达目标或者碰撞到障碍物都reset
        # print(state[-2])
        return np.array(state), reward, done, success, {}

    def render(self):
        pass

    def reset(self):
        print('Waiting for gazebo reset simulation service ... ')
        rospy.wait_for_service('gazebo/reset_simulation')
        print('gazebo reset simulation service available')
        try:
            print('Resetting environment ... ')
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except Exception as e:
                print(e)
        # send_goal=None

        if self.initGoal:
            # while send_goal is None:
        #         try:
        #             send_goal = rospy.wait_for_message('/move_base_simple/goal', PoseStamped, timeout=5)
        #         except:
        #             pass
            
        #     self.goal_x, self.goal_y = send_goal.pose.position.x, send_goal.pose.position.y
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
            # print(send_goal)
            # rospy.loginfo("Goal position : %.1f, %.1f", self.goal_x,
            #                   self.goal_y)
        self.vel_cmd = [0., 0.]
        self.goal_distance = self.getGoalDistace()

      
        state, done, success = self.getState(data)
        # print(np.array(state).shape)

        return np.array(state)



