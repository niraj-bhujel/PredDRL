import rospy
import numpy as np
import math
from copy import deepcopy

from gym import spaces
from gym.utils import seeding
from math import pi, cos, sin
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates

from .respawnGoal import Respawn
from utils.agent import Agent
from utils.graph_utils import create_graph, neighbor_eids, node_type_list

def vector3_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])

def quat_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])

def point_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])

SelfD=0.175
SelfL=0.23

class Env:
    def __init__(self):
        self.goal_x = 1.0
        self.goal_y = 0
        self.inflation_rad = 0.37  # 包含0.17的自身半径

        self.maxLinearSpeed = 0.67
        self.maxAngularSpeed = 2.0

        self.goal_threshold = 0.3
        self.collision_threshold = 0.15

        self.position = Point()
        self.test = False
        self.num_beams = 20  # 激光数

        self.action_type='vw'

        if self.action_type=='xy':
            self.action_space = spaces.Box(low=np.array([-self.maxLinearSpeed, -self.maxLinearSpeed]), 
                                           high=np.array([self.maxLinearSpeed, self.maxLinearSpeed]), 
                                           dtype=np.float32)
        elif self.action_type=='vw':
             self.action_space = spaces.Box(low=np.array([0.0, -self.maxAngularSpeed]), 
                                           high=np.array([self.maxLinearSpeed, self.maxAngularSpeed]), 
                                           dtype=np.float32)           
        
        self.observation_space = spaces.Box(low=np.array([0.0]*self.num_beams + [0., -2., -2*pi, 0]), 
                                            high=np.array([3.5]*self.num_beams + [0.2, 2., 2*pi, 4]), 
                                            dtype=np.float32)

        self.input_shape = 20
        self.window_size = 3 

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.setOdometry)

        # self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        # self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        # self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.respawn_goal = Respawn(stage=2) # stage argument added by niraj

        # self.past_distance = 0. # comment by niraj
        

        # keep track of nodes and their id, added by niraj
        self.nodes = dict()
        self.nid = 0
        self.max_goal_distance = 7

        self.robot = Agent(node_id=self.nid, node_type='robot')
        self.nid+=1

        try:
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        except rospy.ROSException:
            rospy.logerr('ModelStates timeout')
            raise ValueError 

        self.static_obstacles = self.getStaticObstacles(model_states)
        self.pedestirans = self.getPedestrians(model_states)

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

        return goal_distance

    def setOdometry(self, odom):

        self.position = odom.pose.pose.position
        self.orientation = odom.pose.pose.orientation

        orientation_list = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(orientation_list)
        
        inc_y = self.goal_y - self.position.y
        inc_x = self.goal_x - self.position.x
        goal_angle = math.atan2(inc_y, inc_x)
        
        heading = goal_angle - self.yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def preferred_vel(self, pref_speed=0.7):
        d = np.array((self.goal_x - self.position.x, self.goal_y - self.position.y))
        v_pref = pref_speed * d/np.linalg.norm(d)

        return v_pref

    def action_to_vel_cmd(self, action, action_type='xy'):
        # adopted from https://github.com/dongfangliu/NH-ORCA-python/
        vel_msg = Twist()

        vel_msg.linear.y = 0
        vel_msg.linear.z = 0

        vel_msg.angular.x = 0 
        vel_msg.angular.y = 0

        if action_type=='xy':
            vx, vy = action[0], action[1]

            A = 0.5*cos(self.yaw)+SelfD*sin(self.yaw)/SelfL
            B = 0.5*cos(self.yaw)-SelfD*sin(self.yaw)/SelfL
            C = 0.5*sin(self.yaw)-SelfD*cos(self.yaw)/SelfL
            D = 0.5*sin(self.yaw)+SelfD*cos(self.yaw)/SelfL
            
            vr = (vy-C/A*vx)/(D-B*C/A)
            vl = (vx-B*vr)/A

            vel_msg.linear.x = 0.5*(vl+vr)
            # hand edition constraint
            # vel_msg.linear.x = np.clip(-self.maxLinearSpeed, vel_msg.linear.x, self.maxLinearSpeed)
            vel_msg.linear.x = np.clip(0.0, vel_msg.linear.x, self.maxLinearSpeed)

            vel_msg.angular.z = (vr-vl)/SelfL
            # hand edition constraint
            vel_msg.angular.z = np.clip(-self.maxAngularSpeed, vel_msg.angular.z, self.maxAngularSpeed)

        else:
            v, w = action[0], action[1]  
            vel_msg.linear.x = v
            vel_msg.angular.z = w

        return vel_msg


    def getStaticObstacles(self, model_states, obstacle_dict=None):
        # call only once

        if not obstacle_dict:
            obstacle_dict = {}
            
        for i, m_name in enumerate(model_states.name):

            if not 'obstacle' in m_name:
                continue

            # preprare data to update
            pose = model_states.pose[i]
            twist = model_states.twist[i]

            m_pos = point_to_numpy(pose.position)
            # m_vel = vector3_to_numpy(twist.linear)

            m_quat = quat_to_numpy(pose.orientation)
            m_rot = vector3_to_numpy(twist.angular)

            if m_name in self.obstacle_dict:
                node = self.obstacle_dict[node]
            else:
                node = Agent(node_id=self.nid, node_type='obstacle')
                self.nid += 1

            node.update_states(m_pos[:2], q=m_quat, r=m_rot)
            node.update_action(action=[0.0, 0.0])
            node.update_goal(goal=m_pos[:2])

            obstacle_dict[m_name] = node

        return obstacle_dict

    def getPedestrians(self, model_states, ped_dict=None):

        if not ped_dict:
            ped_dict = {}

        for i, m_name in enumerate(model_states.name):
            if not 'pedestrian' in m_name:
                continue

            # preprare data to update
            pose = model_states.pose[i]
            twist = model_states.twist[i]

            m_pos = point_to_numpy(pose.position)
            # m_vel = vector3_to_numpy(twist.linear)

            m_quat = quat_to_numpy(pose.orientation)
            m_rot = vector3_to_numpy(twist.angular)

            if m_name in ped_dict:
                node = ped_dict[m_name]
            else:
                node = Agent(node_id=self.nid, node_type='pedestrian')
                self.nid += 1

            node.update_states(m_pos[:2], q=m_quat, r=m_rot)
            node.update_action(action=[0.0, 0.0])

            cv_pred = node.cv_prediction(time_step=node.time_step)

            node.update_goal(goal=cv_pred[-1])

            ped_dict[m_name] = node
            
        return ped_dict


    def getGraphState(self, action=[0.0, 0.0]):

        try:
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        except rospy.ROSException:
            rospy.logerr('ModelStates timeout')
            raise ValueError 

        self.robot.update_states(p = point_to_numpy(self.position)[:2],
                                 # p=[0.0, 0.0],
                                 q=quat_to_numpy(self.orientation),
                                 r = self.yaw,
                                 )
        self.robot.update_action(action) 
        self.robot.update_goal([self.goal_x, self.goal_y]) 
        
        self.static_obstacles = self.getStaticObstacles(model_states, self.static_obstacles)
        self.pedestirans = self.getPedestrians(model_states, self.pedestirans)

        graph_nodes = [self.robot] + list(self.static_obstacles.values()) + list(self.pedestirans.values())

        g = create_graph(graph_nodes)

        return g

    def getState(self, action):
        scan_range_collision = []

        try:
            scan = rospy.wait_for_message('scan', LaserScan, timeout=100)

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

        state = scan_range_collision + [action, self.heading, self.getGoalDistace] # by niraj
        # state = scan_range_collision + self.vel_cmd + [heading, current_distance] # 极坐标
        # state = scan_range_collision + self.vel_cmd + [self.position.x, self.position.y, self.goal_x, self.goal_y] #笛卡尔坐标
        
        return state


    def step(self, action):

        vel_cmd = self.action_to_vel_cmd(action, self.action_type)
        self.pub_cmd_vel.publish(vel_cmd)

        # state = self.getState(v, w)
        state = self.getGraphState(action)

        done=False
        success = False
        collision = False

        current_distance = self.getGoalDistace()
        reaching_goal = current_distance < self.goal_threshold
        too_far = current_distance > self.max_goal_distance
        
        robot_node = state.nodes()[state.ndata['cid']==node_type_list.index('robot')]
        robot_neighbor_eids = neighbor_eids(state, robot_node)
        robot_neighbor_dist = state.edata['dist'][robot_neighbor_eids]
        
        if robot_neighbor_dist.size(0)>0: 
            if self.collision_threshold > robot_neighbor_dist.min().item()-0.2:
                collision=True

        if collision:
        # if self.collision_threshold > min(scan_range_collision) + 1e-6:
            rospy.loginfo("Collision!!")
            done = True
            reward = -100

        elif reaching_goal:
            rospy.loginfo("Success!!")
            success = True
            reward = 100
            
        elif too_far: # added by niraj
            rospy.loginfo("Robot too far away from goal!!")
            done = True
            reward = -100

        else:
            # reward = -0.5
            reward = (self.goal_threshold-current_distance)*0.1

        # NOTE! if goal node is included in the graph, goal must be respawned before calling graph state, otherwise graph create fails. 
        if success:
            self.init_goal(position_check=True, test=self.test)

        if done:
            # self.pub_cmd_vel.publish(Twist())
            self.reset()

        # stop agent, while policy update
        # self.pub_cmd_vel.publish(Twist())

        return state, reward, done, success

    # add a separate function to initialize goal, delete old goal if exist and respawn new goal
    def init_goal(self, position_check=False, test=False):
        
        self.goal_x, self.goal_y = self.respawn_goal.getPosition(position_check, test)
        
        if self.respawn_goal.check_model: self.respawn_goal.deleteModel()
        self.respawn_goal.respawnModel()

        rospy.loginfo("Init New Goal : (%.1f, %.1f)", self.goal_x, self.goal_y)

    def reset(self, initGoal=False):
        self.pub_cmd_vel.publish(Twist())

        try:
            rospy.wait_for_service('gazebo/reset_simulation')
            reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
            reset_proxy()
            
            rospy.loginfo('Env Reset')

        except (rospy.ServiceException) as e:
            rospy.loginfo("gazebo/reset_simulation service call failed")

        if initGoal:
            self.init_goal()

        # state = self.getState()
        state = self.getGraphState()

        return state