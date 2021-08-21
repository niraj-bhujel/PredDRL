import rospy
import numpy as np
import math
from copy import deepcopy

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
from utils.graph_utils import create_graph, node_type_list

def vector3_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])

def quat_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])

def point_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])

class Env:
    def __init__(self):
        self.goal_x = 1.0
        self.goal_y = 0
        self.inflation_rad = 0.37  # 包含0.17的自身半径
        self.heading = 0
        self.pre_heading = 0
        self.max_v = 0.2
        self.max_w = 1.5
        self.goal_threshold = 0.3
        self.collision_threshold = 0.15
        # self.vel_cmd = [0., 0.]

        self.position = Point()
        self.test = False
        self.num_beams = 20  # 激光数

        self.action_space = spaces.Box(low=np.array([-0.001, -2.]), 
                                       high=np.array([0.5, 2.]), 
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
        

        # keep track of nodes and their id, added by niraj
        self.nodes = dict()
        self.nid = 0
        self.max_goal_distance = 3

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
        self.orientation = odom.pose.pose.orientation

        orientation_list = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        _, _, yaw = self.euler_from_quaternion(orientation_list)
        
        inc_y = self.goal_y - self.position.y
        inc_x = self.goal_x - self.position.x
        goal_angle = math.atan2(inc_y, inc_x)
        
        heading = goal_angle - yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getGraphState(self, v=0, w=0):
        try:
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        except rospy.ROSException:
            rospy.logerr('ModelStates timeout')
            raise ValueError 

        nodes = []
        # keep track of model states
        for i in range(len(model_states.name)):
            m_name = model_states.name[i]
            
            if 'square' in m_name:
                continue

            elif 'ground' in m_name:
                continue

            elif 'goal' in m_name:
                continue
                # node_type = 'goal'

            elif 'obstacle' in m_name:
                node_type = 'obstacle'

            elif 'burger' in m_name:
                node_type = 'robot'

            else:
                node_type = 'pedestrian'

            # get the corresponding node
            if m_name in self.nodes.keys():
                node = self.nodes[m_name]
            else:
                self.nid +=1 
                node = Node(node_id=self.nid, node_type=node_type)
                self.nodes[m_name] = node

            # preprare data to update
            pose = model_states.pose[i]
            twist = model_states.twist[i]

            m_pos = point_to_numpy(pose.position)
            m_vel = vector3_to_numpy(twist.linear)

            m_quat = quat_to_numpy(pose.orientation)
            m_rot = vector3_to_numpy(twist.angular)

            if node_type =='robot':
                m_pos = point_to_numpy(self.position)
                m_quat = quat_to_numpy(self.orientation)
                # m_pos = [0.0, 0.0]
                action = [v, w]

            else: # relative pos and vel
                # m_pos -= point_to_numpy(self.position) # measured from robot pos

                # if len(node)>0:
                #     # last_pos = node._pos[-1]
                #     m_vel = (m_pos[:2] - np.array(node._pos[-1]))/node.time_step
                # else:
                #     m_vel = np.zeros_like(m_pos)
                action = [0., 0.]

            node.update_states(m_pos[:2], m_vel[:2], m_quat, m_rot, action=action)

            # udpate goal, only after updating states
            if node_type=='robot':
                node._goal = [self.goal_x, self.goal_y]
                
            elif node_type=='obstacle':
                node._goal = m_pos[:2]

            else:
                node._goal = node.cv_prediction(node.last_timestep)[-1]

            nodes.append(node)
        
        g = create_graph(nodes)

        done=False
        success = False
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        # current_distance = g.ndata['gdist'][g.ndata['cid']==node_type_list.index('robot')]

        if self.collision_threshold > g.edata['dist'].min():
            rospy.loginfo("Collision!!")
            done = True
            reward = -150

        elif current_distance < self.goal_threshold:
            rospy.loginfo("Success!!, Goal (%.2f, %.2f) reached.", self.goal_x, self.goal_y)
            success = True
            reward = 200
            
        elif current_distance > self.max_goal_distance: # added by niraj
            rospy.loginfo("Robot too far away from goal!!")
            done = True
            reward = -150

        # elif abs(v)>3:
        #     reward =  -np.log(v**2)

        # elif abs(w)>np.pi:
        #     reward = -np.log(w**2)

        else:
            # reward = (self.goal_threshold-current_distance) * 0.1
            reward = (current_distance - self.goal_threshold) * 0.1

        # # 增加一层膨胀区域，越靠近障碍物负分越多
        # obstacle_min_range = round(min(scan_range_collision), 2)
        # if obstacle_min_range < self.inflation_rad:
        #     # reward += 100.0*(obstacle_min_range - self.inflation_rad)/self.inflation_rad
        #     reward -= 5.0*(1 - obstacle_min_range/self.inflation_rad)

        return g, reward, done, success

    def getState(self, v=0, w=0):
        # scan_range = [] # commented by niraj
        scan_range_collision = []
        heading = self.heading

        done = False
        success = False # added by niraj

        # moved from step to getState - niraj
        try:
            scan = rospy.wait_for_message('scan', LaserScan, timeout=100)
            # if data is not None: break
        except rospy.ROSException:
            rospy.logerr('LaserScan timeout during env step')
            raise ValueError

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range_collision.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range_collision.append(0)
            else:
                scan_range_collision.append(scan.ranges[i])

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # obstacle_min_range = round(min(scan_range), 2)
        # obstacle_angle = np.argmin(scan_range)
        if self.collision_threshold > min(scan_range_collision) + 1e-6:
            rospy.loginfo("Collision!!")
            done = True
            reward = -150

        elif current_distance < self.goal_threshold:
            rospy.loginfo("Success!!, Goal (%.2f, %.2f) reached.", self.goal_x, self.goal_y)
            success = True
            reward = 200
            

        elif current_distance > self.max_goal_distance: # added by niraj
            rospy.loginfo("Robot too far away from goal!!")
            done = True
            reward = -150

        elif abs(v)>3:
            reward =  -np.log(v**2)

        elif abs(w)>np.pi:
            reward = -np.log(w**2)

        else:
            # reward = (self.goal_threshold-current_distance) * 0.1
            reward = (current_distance - self.goal_threshold) * 0.1

        # # 增加一层膨胀区域，越靠近障碍物负分越多
        # obstacle_min_range = round(min(scan_range_collision), 2)
        # if obstacle_min_range < self.inflation_rad:
        #     # reward += 100.0*(obstacle_min_range - self.inflation_rad)/self.inflation_rad
        #     reward -= 5.0*(1 - obstacle_min_range/self.inflation_rad)

        # print(scan_range_collision)
        state = scan_range_collision + [v, w] + [heading, current_distance] # by niraj
        # state = scan_range_collision + self.vel_cmd + [heading, current_distance] # 极坐标
        # state = scan_range_collision + self.vel_cmd + [self.position.x, self.position.y, self.goal_x, self.goal_y] #笛卡尔坐标
        
        return state, reward, done, success
       
    def _setReward(self, state, done, success):

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # current_distance = round(math.hypot(self.goal_x - self.position.position.x, self.goal_y - self.position.position.y), 2)
        # distance_rate = (self.past_distance - current_distance)
        # self.past_distance = current_distance

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
        v, w = action[0], action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = v
        vel_cmd.angular.z = w

        self.pub_cmd_vel.publish(vel_cmd)
        # self.vel_cmd = [vel_cmd.linear.x, vel_cmd.angular.z]

        # state, reward, done, success = self.getState(v, w)
        state, reward, done, success = self.getGraphState(v, w)

        # NOTE! if goal node is included in the graph, goal must be respawned before calling graph state, otherwise graph create fails. 
        if success:
            self.init_goal(position_check=True, test=self.test)

        if done or success:
            self.pub_cmd_vel.publish(Twist()) 

        # # stop agent
        # vel_cmd.linear.x = 0
        # vel_cmd.angular.z = 0
        # self.pub_cmd_vel.publish(vel_cmd)

        return state, reward, done, success

    # add a separate function to initialize goal, delete old goal if exist and respawn new goal
    def init_goal(self, position_check=False, test=False):
        
        self.goal_x, self.goal_y = self.respawn_goal.getPosition(position_check, test)
        
        if self.respawn_goal.check_model: self.respawn_goal.deleteModel()
        self.respawn_goal.respawnModel()

        rospy.loginfo("Init New Goal : (%.1f, %.1f)", self.goal_x, self.goal_y)
        self.goal_distance = self.getGoalDistace()
        # self.vel_cmd = [0., 0.] # comment by niraj

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

        # set initial velocity and goal distance
        # self.vel_cmd = [0., 0.]
        # self.goal_distance = self.getGoalDistace()

        # state, _, _, _ = self.getState()
        state, _, _, _ = self.getGraphState()

        return state