import rospy
import math
import random

import numpy as np
from copy import deepcopy

from gym import spaces
from gym.utils import seeding
from math import pi, cos, sin
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped, Quaternion
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState

from .respawnGoal import Respawn
from .respawnPeds import RespawnPedestrians
from .env_utils import *

from utils.agent import Agent
from utils.graph_utils import create_graph, min_neighbor_distance, node_type_list
from utils.timer import Timer
from policy.orca import ORCA
from preddrl_data.scripts.prepare_data import prepare_data

SelfD=0.175
SelfL=0.23


class Env:
    def __init__(self, test=False, stage=0, graph_state=False, dataset='zara1'):

        self.test = test
        self.graph_state = graph_state
        self.stage = stage
        self.dataset = dataset
        self.robot_name = 'turtlebot3_burger'

        self.gx = 0
        self.gy = 1

        self.inflation_rad = 0.37  # 包含0.17的自身半径

        self.goal_threshold = 0.3
        self.collision_threshold = 0.15

        self.inital_pos = Point(7.5, 6.5, 0.)
        # self.inital_pos = Point(0., 0., 0.)
        
        self.num_beams = 20  # 激光数

        self.maxLinearSpeed = 0.43
        self.maxAngularSpeed = 2.0

        self.action_type='xy'

        if self.action_type=='xy':
            self.action_space = spaces.Box(low=np.array([-self.maxLinearSpeed, -self.maxLinearSpeed]), 
                                           high=np.array([self.maxLinearSpeed, self.maxLinearSpeed]), 
                                           dtype=np.float32)
        elif self.action_type=='vw':
             self.action_space = spaces.Box(low=np.array([0., -self.maxAngularSpeed]), 
                                           high=np.array([self.maxLinearSpeed, self.maxAngularSpeed]), 
                                           dtype=np.float32)            
        
        self.observation_space = spaces.Box(low=np.array([0.0]*self.num_beams + [0., -2., -2*pi, 0]), 
                                            high=np.array([3.5]*self.num_beams + [0.2, 2., 2*pi, 4]), 
                                            dtype=np.float32)

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.setOdometry)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.setScan)

        self.respawn_goal = Respawn(stage) # stage argument added by niraj        

        self.time_step = 0.2
        self.timer = Timer() # set by trainer

        self.max_goal_distance = 20.
        self.last_goal_distance = 0.

        # robot policy
        self.orca = ORCA(self.time_step)

        self.global_step = 0

        self.future_steps = 4
        
        self.collision_times = 0

        self.initialize_agents()

    def initialize_agents(self, ):
        self.vel_cmd = Twist()
        # keep track of nodes and their id, added by niraj
        self.nid = 0
        self.robot = Agent(node_id=self.nid, node_type='robot', time_step=self.time_step)
        self.nid+=1

        self.robot_goal = Agent(node_id=self.nid, node_type='robot_goal', time_step=self.time_step)
        self.nid += 1

        self.obstacles = dict()

        if self.stage==7:
            self.pedestrians, self.ped_frames, _  = prepare_data('./preddrl_data/{}.txt'.format(self.dataset), 
                                                            target_frame_rate=2*int(1/self.time_step),
                                                            max_peds=40)
            # self.ped_frames = self.ped_frames[:50]# use only first 50 frames
            print("Total pedestrians:{}, Total frames:{}".format(len(self.pedestrians), len(self.ped_frames)))
            self.respawn_pedestrian = RespawnPedestrians()

        self.update_agents(robot_action=(0., 0.))

    def setScan(self, scan):
        self.scan = scan

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getGoalDistance(self):
        goal_distance = round(math.hypot(self.gx - self.position.x, self.gy - self.position.y), 2)

        return goal_distance

    def setOdometry(self, odom):

        self.position = odom.pose.pose.position
        self.orientation = odom.pose.pose.orientation
        self.linear = odom.twist.twist.linear

        orientation_list = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)
        
        inc_y = self.gy - self.position.y
        inc_x = self.gx - self.position.x
        goal_angle = math.atan2(inc_y, inc_x)
        
        heading = goal_angle - self.yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def sample_robot_action(self, policy='uniform'):
        if policy =='uniform':
            action = self.action_space.sample()
            print("uniform Vel:", action)

        elif policy == 'vpref':
            action = self.robot.preferred_vel() #+ np.random.normal(0, 0.1, size=(2,))
            print("V_pref:", action)
            if self.action_type=='vw':
                action = self.xy_to_vw(action)
        else:
            obstacle_pos = [tuple(o.pos) for _, o in self.obstacles.items()]
            action, _ = self.orca.predict(self.robot, humans=self.pedestrians_list,
                                                obstacles=obstacle_pos)
            print('ORCA Vel:', action)
            if self.action_type=='vw':
                action = self.xy_to_vw(action)

        return action

    def vw_to_xy(self, action):
        theta = self.yaw + action[1]
        vx = action[0] * np.cos(theta)
        vy = action[0] * np.sin(theta)
        return vx, vy

    def xy_to_vw(self, v):
        
        A = 0.5*cos(self.yaw)+SelfD*sin(self.yaw)/SelfL
        B = 0.5*cos(self.yaw)-SelfD*sin(self.yaw)/SelfL
        C = 0.5*sin(self.yaw)-SelfD*cos(self.yaw)/SelfL
        D = 0.5*sin(self.yaw)+SelfD*cos(self.yaw)/SelfL
        
        vx, vy = v[0], v[1]

        vr = (vy-C/A*vx)/(D-B*C/A)
        vl = (vx-B*vr)/A

        v = 0.5*(vl+vr)

        w = (vr-vl)/SelfL

        v = np.clip(0, v, self.maxLinearSpeed)
        w = np.clip(-self.maxAngularSpeed, w, self.maxAngularSpeed)

        return v, w

    def action_to_vel_cmd(self, action, action_type='xy'):
        # adopted from https://github.com/dongfangliu/NH-ORCA-python/
        vel_msg = Twist()

        vel_msg.linear.y = 0
        vel_msg.linear.z = 0

        vel_msg.angular.x = 0 
        vel_msg.angular.y = 0

        if action_type=='xy':
            v, w = self.xy_to_vw(action)

        else:
            v, w = action[0], action[1]

            # v = (v+2)/10.

        vel_msg.linear.x = v
        vel_msg.angular.z = w

        return vel_msg


    def updateObstaclesStates(self, model_states, obstacle_dict=None):
        # call only once

        if not obstacle_dict:
            obstacle_dict = {}
        
        for i, m_name in enumerate(model_states.name):

            if not 'obstacle' in m_name:
                continue

            # preprare data to update
            pose = model_states.pose[i]
            twist = model_states.twist[i]


            if m_name in obstacle_dict:
                node = obstacle_dict[m_name]
            else:
                node = Agent(node_id=self.nid, node_type='obstacle', time_step=self.time_step)
                self.nid += 1

            px, py = pose.position.x, pose.position.y
            node.update_states(px, py, px, py, theta=0)
            obstacle_dict[m_name] = node

        return obstacle_dict

    def updatePedestrians(self, model_states=None):
        '''
        Get pedestrians at current step, update and return pedestrian states, 
        Meanwhile spawn the current pedestrians to gazebo
        '''
        if self.global_step>len(self.ped_frames):
            t = self.global_step%len(self.ped_frames)
        else:
            t = self.global_step

        curr_peds = [ped for ped in self.pedestrians if t>=ped.first_timestep and t<ped.last_timestep]

        ped_states = {}
        # update action of the current peds
        for ped in curr_peds:
            # ground truth state
            state = ped.get_state_at(t) 
            ped.set_state(state.px, state.py, state.vx, state.vy, state.gx, state.gy, state.theta)
            
            # ground truth action
            if self.action_type=='xy':
                action = (state.vx, state.vy)
            else:
                action = (math.hypot(state.vx, state.vy), state.theta)
            ped.set_action(action)

            ped_futures = np.zeros((self.future_steps, 2))
            for i, ts in enumerate(range(t, min(t+self.future_steps, ped.last_timestep))):
                _s = ped.get_state_at(ts)

                if self.action_type=='xy':
                    future = (_s.vx, _s.vy)
                else:
                    future = (math.hypot(_s.vx, _s.vy), _s.theta)
                ped_futures[i] = future

            ped.set_futures(ped_futures)
            # print('future:', ped.futures)

            ped_states[ped.id] = ped.deserialize_state(state)

        self.respawn_pedestrian.respawn(ped_states, model_states)
            
        return curr_peds

    def update_agents(self, action=None, robot_action=None):

        try:
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        except rospy.ROSException:
            rospy.logerr('ModelStates timeout')
            raise ValueError 

        # for i, m_name in enumerate(model_states.name):

        #     if not m_name==self.robot_name:
        #         continue

        #     # preprare data to update
        #     robot_pose = model_states.pose[i]
        #     robot_twist = model_states.twist[i]

        # print('robot position (odom)', self.position)
        # print('robot position (model_states)', robot_pose.position)
        # print('robot linear(odom)', self.linear)
        # print('robot linear (model_states)', robot_twist.linear)

        if self.robot.px is not None:
            vx = (self.position.x - self.robot.px)/self.time_step
            vy = (self.position.y - self.robot.py)/self.time_step
        else:
            vx, vy = (0., 0.)

        # print('robot linear(manual)', vx, vy)
        self.robot.set_state(self.position.x, self.position.y, vx, vy, self.gx, self.gy, theta=self.yaw)
        self.robot.set_action(robot_action) 

         # future vel
        self.robot.set_futures([self.robot.action for _ in range(self.future_steps)])

        # goal as a node
        self.robot_goal.set_state(self.gx, self.gy, 0., 0., self.gx, self.gy, theta=0.0)
        self.robot_goal.set_futures([(0., 0.) for _ in range(self.future_steps)]) # future vel
        
        # update obstacle and pedestrians
        # self.obstacles = self.updateObstaclesStates(model_states, self.obstacles)
        self.pedestrians_list = self.updatePedestrians(model_states)

        
    def getGraphState(self, action=None, last_state=None, done=False, success=False):

        graph_nodes = [self.robot, self.robot_goal] + self.pedestrians_list

        state = create_graph(graph_nodes, self.robot.pos)

        if action is None:
            return state


        current_distance = self.getGoalDistance()
        reaching_goal = current_distance < self.goal_threshold
        too_far = current_distance > self.max_goal_distance

        if self.action_type=='vw':
            theta = last_state.ndata['yaw'].numpy().flatten() + action[:, 1]
            px = last_state.ndata['pos'][:, 0].numpy() + np.cos(theta) * action[:, 0] * self.time_step
            py = last_state.ndata['pos'][:, 1].numpy() + np.sin(theta) * action[:, 0] * self.time_step
            pred_pos = np.stack([px, py], axis=-1)
        else:
            pred_pos = last_state.ndata['pos'] + action * self.time_step

        collision_agents = []
        distance_matrix = np.linalg.norm(pred_pos[:, None, :]-pred_pos[None, :, :], axis=-1)
        
        num_collisions = 0
        for i in range(distance_matrix.shape[0]):
            if last_state.ndata['cid'][i]==node_type_list.index('robot_goal'):
                continue
            for j in range(i+1, distance_matrix.shape[1]):
                if last_state.ndata['cid'][j]==node_type_list.index('robot_goal'):
                    continue

                if distance_matrix[i][j] < self.collision_threshold+0.15:
                    # get the corresponding id
                    collision_agents.extend([int(last_state.ndata['tid'][i]), int(last_state.ndata['tid'][j])])

                    # break # one collision is enough
                    num_collisions += 1

        collision = self.robot.id in collision_agents

        # # compute correct action reward for agents
        # ped_mask = (last_state.ndata['cid']==node_type_list.index('pedestrian')).unsqueeze(1).numpy()
        # self.action_error = np.mean(np.linalg.norm(last_state.ndata['action'].numpy() - action, axis=-1, keepdims=True)*ped_mask)

        if collision:
            rospy.loginfo("Collision!!")
            done = True
            reward = -100
            self.collision_times += 1

        elif reaching_goal:
            success = True
            reward = 150#100
            rospy.loginfo('Success!!')
            
        elif too_far:
            done = True
            reward = 100
            rospy.loginfo('Too Far from Goal!!')

        else:
            reward = 0.

        goal_distance = self.getGoalDistance()
        if goal_distance < self.last_goal_distance:
            reward += 1
        else:
            reward -= 1

        self.last_goal_distance = goal_distance

        reward -= num_collisions/last_state.number_of_nodes()
        # reward -= self.action_error

        return state, reward, done, success

    def getState(self, action=None, done=False, success=False):
        
        scan = None
        while scan is None:
            try:
                scan = rospy.wait_for_message('scan', LaserScan, timeout=100)

            except rospy.ROSException:
                rospy.logerr('LaserScan timeout during env step')
        
        scan_range_collision = []
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range_collision.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range_collision.append(0.0)
            else:
                scan_range_collision.append(scan.ranges[i])

        if action==None:
            return scan_range_collision + [0., 0.,self.heading, self.getGoalDistance()]

        collision =  min(scan_range_collision)+1e-6 < self.collision_threshold

        # one step ahead collision detection
        if self.action_type=='vw':
            vx, vy = self.vw_to_xy(action)
        else:
            vx, vy = action

        px = self.position.x + vx * self.time_step
        py = self.position.y + vy * self.time_step

        for _, obstacle in self.obstacles.items():
            if np.linalg.norm((obstacle.px-px, obstacle.py-py))<0.1:
                collision=True
                break

        goal_distance = self.getGoalDistance()
        reaching_goal = goal_distance< self.goal_threshold
        too_far = goal_distance > self.max_goal_distance

        state = scan_range_collision + [action[0], action[1], self.heading, goal_distance]

        if collision:
            rospy.loginfo("Collision!!")
            done = True
            reward = -100
            self.collision_times += 1

        elif reaching_goal:
            success = True
            reward = 150#100
            rospy.loginfo('Success!!')
            
        elif too_far:
            done = True
            reward = -100
            rospy.loginfo('Too Far from Goal!!')

        else:
            reward = 0.
            # reward = self.goal_threshold-self.getGoalDistance()

        goal_distance = self.getGoalDistance()
        if goal_distance < self.last_goal_distance:
            reward += 1
        else:
            reward -= 1

        self.last_goal_distance = goal_distance

        return state, reward, done, success

    def step(self, action, last_state):

        robot_action = action[last_state.ndata['tid']==node_type_list.index('robot')].flatten() if self.graph_state else action

        self.vel_cmd = self.action_to_vel_cmd(robot_action, self.action_type)
        self.pub_cmd_vel.publish(self.vel_cmd)
        rospy.sleep(self.time_step)
        self.pub_cmd_vel.publish(Twist())

        self.update_agents(action, robot_action=robot_action)


        if self.graph_state:
            state, reward, done, success = self.getGraphState(action, last_state)
        else:
            state, reward, done, success = self.getState(action)

        # NOTE! if goal node is included in the graph, goal must be respawned before calling graph state, otherwise graph create fails. 
        if success:
            self.init_goal(position_check=True)


        if done:
            # self.pub_cmd_vel.publish(Twist())
            self.reset()

        self.global_step+=1

        return state, reward, done, success

    def init_goal(self, position_check=False, test=False):
        
        self.gx, self.gy = self.respawn_goal.getPosition(position_check, test)
        
        if self.respawn_goal.check_model: self.respawn_goal.deleteModel()
        self.respawn_goal.respawnModel()


        rospy.loginfo("Init New Goal : (%.1f, %.1f)", self.gx, self.gy)

    def reset(self, initGoal=False):
        # reset robot velocity
        self.pub_cmd_vel.publish(Twist())

        # reset robot pose and randomly set the orientation
        tmp_state = ModelState()
        tmp_state.model_name = "turtlebot3_burger"
        tmp_state.pose = Pose(self.inital_pos, euler_to_quaternion([0.0, 0.0, random.uniform(0, 360)]))
        tmp_state.reference_frame = "world"
        rospy.wait_for_service("/gazebo/set_model_state")
        set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        set_model_state(tmp_state)

        if initGoal:
            self.init_goal()

        if self.graph_state:
            state = self.getGraphState()
        else:
            state = self.getState()

        return state