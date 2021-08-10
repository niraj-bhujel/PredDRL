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
from gazebo_msgs.msg import ModelStates

from .respawnGoal import Respawn
from preddrl_tracker.scripts.node import Node

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
        # self.initGoal = True # coment out by niraj
        # self.get_goalbox = False # removed by niraj
        self.position = Point()
        self.test = False
        self.num_beams = 20  # 激光数

        self.action_space = spaces.Box(low=np.array([-2., -2.]), 
                                       high=np.array([2., 2.]), 
                                       dtype=np.float32)
        
        self.observation_space = spaces.Box(low=np.array([0.0]*self.num_beams + [0., -2., -2*pi, 0]), 
                                            high=np.array([3.5]*self.num_beams + [0.2, 2., 2*pi, 4]), 
                                            dtype=np.float32)

        self.input_shape = 20
        self.window_size = 3  
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.respawn_goal = Respawn()
        self.past_distance = 0.
        self.nodes = {}

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

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

    def getTracks(self, ):
        try:
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        except rospy.ROSException:
            rospy.logerr('ModelStates timeout')
            raise ValueError 

        node_states = []
        # keep track of model states
        for i in range(model_states.name):
            m_name = model_states.name[i]
            
            if 'square' in m_name:
                continue
            if 'goal' in m_name:
                continue
            if 'gound' in m_name:
                continue
            
            if m_name=='obstacle':
                node_type = 'obstacle'
            elif m_name == 'turtlebot3_burger':
                node_type = 'robot'
            else:
                node_type = 'pedestrian'

            pose = model_states.pose[i]
            twist = model_states.twist[i]

            m_pos = point_to_numpy(pose.position)
            m_quat = quat_to_numpy(pose.orientation)
            m_vel = vector3_to_numpy(twist.linear)
            m_rot = vector3_to_numpy(twist.angular)

            if m_name in self.nodes.keys():
                node = self.nodes[m_name]
                node.update(m_pos, m_vel, m_quat, m_rot)
                
            else:
                node = Node(node_id=str(m_name), node_type=node_type)
                node.update(m_pos, m_vel, m_quat, m_rot)

            node_states.append(node)
            
        return node_states

            

    def getState(self,):

        done = False
        success = False # added by niraj

        try:
            scan = rospy.wait_for_message('scan', LaserScan, timeout=100)
        except rospy.ROSException:
            rospy.logerr('LaserScan timeout')
            raise ValueError

        scan_ranges = np.array(scan.ranges)
        scan_ranges[scan_ranges==np.inf]==3.5
        scan_ranges[scan_ranges==np.nan] == 1e-6

        if self.collision_threshold > min(scan_ranges):
            rospy.loginfo("Collision!!")
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # current_distance = round(math.hypot(self.goal_x - self.position.position.x, self.goal_y - self.position.position.y), 2)
        if current_distance < self.goal_threshold:
            rospy.loginfo("Success!!, Goal (%.2f, %.2f) reached.", self.goal_x, self.goal_y)
            success = True # modified by niraj
            
        state = scan_ranges.tolist() + self.vel_cmd + [self.heading, current_distance] # 极坐标
        # state = scan_range + self.vel_cmd + [self.position.x, self.position.y, self.goal_x, self.goal_y] #笛卡尔坐标
        
        return state, done, success
       
    def setReward(self, state, done, success):

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # current_distance = round(math.hypot(self.goal_x - self.position.position.x, self.goal_y - self.position.position.y), 2)
        # distance_rate = (self.past_distance - current_distance)
        self.past_distance = current_distance

        if done:
            reward = -150

        elif success:
            reward = 200

        else:
            reward = (self.goal_threshold-state[-1]) * 0.1 #- 0.25*abs(state[-3])15*distance_rate
            # reward = 15*distance_rate
        # 增加一层膨胀区域，越靠近障碍物负分越多
        obstacle_min_range = round(min(state[:self.num_beams]), 2)
        if obstacle_min_range < self.inflation_rad:
            # reward += 100.0*(obstacle_min_range - self.inflation_rad)/self.inflation_rad
            reward -= 5.0*(1 - obstacle_min_range/self.inflation_rad)

        return reward

    def step(self, action):
        self.pre_heading = self.heading
        vel_cmd = Twist()
        # vel_cmd.linear.x = (action[0]*2.5 + 5.0) / 20.0
        # vel_cmd.linear.x = (action[0]*1.75 + 3.5) / 20.0

        vel_cmd.linear.x = (action[0] + 2.0) / 20.0
        vel_cmd.angular.z = action[1]
        self.vel_cmd = [vel_cmd.linear.x, vel_cmd.angular.z]
        self.pub_cmd_vel.publish(vel_cmd)
    
        state, done, success = self.getState()
        
        reward = self.setReward(state, done, success)
        # added by niraj
        if done:
            self.pub_cmd_vel.publish(Twist())
        # added by niraj
        if success:
            self.pub_cmd_vel.publish(Twist())
            self.init_goal(position_check=True, test=self.test)

        return np.array(state), reward, done, success, {}



    # add a separate function to initialize goal, delete old goal if exist and respawn new goal
    def init_goal(self, position_check=False, test=False):
        
        self.goal_x, self.goal_y = self.respawn_goal.getPosition(position_check, test)
        
        if self.respawn_goal.check_model: self.respawn_goal.deleteModel()
        self.respawn_goal.respawnModel()

        rospy.loginfo("Init New Goal : (%.1f, %.1f)", self.goal_x, self.goal_y)
        self.goal_distance = self.getGoalDistace()


    def reset(self, initGoal=False):

        try:
            rospy.wait_for_service('gazebo/reset_simulation')

            # print('Resetting environment ... ')
            self.reset_proxy()
            # print('Environment is reset.')
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        if initGoal:
            self.init_goal()

        self.vel_cmd = [0., 0.]
        self.goal_distance = self.getGoalDistace()

        state, done, success = self.getState()

        return np.array(state)








