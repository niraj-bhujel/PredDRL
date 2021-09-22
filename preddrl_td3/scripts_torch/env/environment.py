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
from utils.graph_utils import create_graph, min_neighbor_distance, node_type_list, find_collision_nodes
from utils.timer import Timer
from policy.orca import ORCA
from preddrl_tracker.scripts.pedestrian_state_publisher import prepare_data

SelfD=0.175
SelfL=0.23


class Env:
    def __init__(self, test=False, stage=0, graph_state=False):

        self.test = test
        self.graph_state = graph_state
        self.stage = stage

        self.goal_x = 0
        self.goal_y = 1

        self.inflation_rad = 0.37  # 包含0.17的自身半径

        self.maxLinearSpeed = 0.67
        self.maxAngularSpeed = 2.0

        self.goal_threshold = 0.3
        self.collision_threshold = 0.15

        self.inital_pos = Point(7.5, 6.5, 0.)
        # self.inital_pos = Point(0., 0., 0.)
        
        self.num_beams = 20  # 激光数

        self.action_type='xy'

        if self.action_type=='xy':
            self.action_space = spaces.Box(low=np.array([-self.maxLinearSpeed, -self.maxLinearSpeed]), 
                                           high=np.array([self.maxLinearSpeed, self.maxLinearSpeed]), 
                                           dtype=np.float32)
        else:
             self.action_space = spaces.Box(low=np.array([0., -self.maxAngularSpeed]), 
                                           high=np.array([self.maxLinearSpeed, self.maxAngularSpeed]), 
                                           dtype=np.float32)           
        
        self.observation_space = spaces.Box(low=np.array([0.0]*self.num_beams + [0., -2., -2*pi, 0]), 
                                            high=np.array([3.5]*self.num_beams + [0.2, 2., 2*pi, 4]), 
                                            dtype=np.float32)

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.setOdometry)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.setScan)

        self.goal_spawnner = Respawn(stage) # stage argument added by niraj        

        self.time_step = 0.2
        self.timer = Timer() # set by trainer

        self.max_goal_distance = 20.
        self.last_goal_distance = 0.

        # robot policy
        self.orca = ORCA(self.time_step)

        self.global_step = 0

        self.future_steps = 4

        self.collision_times = 0

        self.initialize_agents() # initialize agents

    def initialize_agents(self, ):

        self.vel_cmd = Twist()
        # keep track of nodes and their id, added by niraj
        # self.nid = 0
        self.robot = Agent(node_id=99, node_type='robot', time_step=self.time_step)
        # self.nid+=1

        self.robot_goal = Agent(node_id=98, node_type='robot_goal', time_step=self.time_step)
        # self.nid += 1

        self.obstacles = dict()

        if self.stage==7:
            self.pedestrians, self.ped_frames, _  = prepare_data('./preddrl_tracker/data/students003.txt', 
                                                            target_frame_rate=2*int(1/self.time_step), 
                                                            max_peds=40)
            # self.ped_frames = self.ped_frames[:50]# use only first 50 frames
            print("Total pedestrians:{}, Total frames:{}".format(len(self.pedestrians), len(self.ped_frames)))
            self.ped_spawnner = RespawnPedestrians()

        self.update_agents()

    def setScan(self, scan):
        self.scan = scan

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass


    def getGoalDistance(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def setOdometry(self, odom):

        self.position = odom.pose.pose.position
        self.orientation = odom.pose.pose.orientation

        orientation_list = [self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)
        
        inc_y = self.goal_y - self.position.y
        inc_x = self.goal_x - self.position.x
        goal_angle = math.atan2(inc_y, inc_x)
        
        heading = goal_angle - self.yaw

        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def sample_robot_action(self, method='uniform'):
        if method =='uniform':
            action = self.action_space.sample()

        elif method == 'vpref':
            action = self.robot.preferred_vel(self.robot.vpref) + np.random.normal(0, 0.1, size=(2,))
            print('Pref Vel:', action)
            if self.action_type=='vw':
                action = self.xy_to_vw(action)
            
        else:
            obstacle_pos = [tuple(o.pos) for _, o in self.obstacles.items()]
            action, _ = self.orca.predict(self.robot, humans=self.pedestrians_list, obstacle_pos=obstacle_pos)
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
            node.set_states(px, py, px, py, theta=0)
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
            state = ped.get_states_at(t) 

            ped.set_states(state[0], state[1], state[4], state[5], state[6])

            # ground truth action
            if self.action_type=='xy':
                action = (state[2], state[3])
            else:
                action = (math.hypot(state[2], state[3]), state[6])

            ped.set_action(action)

            # ground truth futures
            ped_futures = np.zeros((self.future_steps, 2))
            for i, _t in enumerate(range(t, min(t+self.future_steps, ped.timesteps))):
                ped_futures[i] = ped.get_states_at(_t)[:2]

            ped.set_futures(ped_futures)

            ped_states[ped.id] = state

        self.ped_spawnner.respawn(ped_states, model_states)
            
        return curr_peds


    def update_agents(self):

        try:
            model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        except rospy.ROSException:
            rospy.logerr('ModelStates timeout')
            raise ValueError 

        self.robot.set_states(self.position.x, self.position.y, self.goal_x, self.goal_y, theta=self.yaw)
        self.robot.set_action((self.vel_cmd.linear.x, self.vel_cmd.angular.z)) 

        robot_futures = self.robot.cv_prediction(self.future_steps)
        self.robot.set_futures(robot_futures)

        # goal as a node
        self.robot_goal.set_states(self.goal_x, self.goal_y, self.goal_x, self.goal_y, theta=0.0)
        self.robot_goal.set_futures([self.robot_goal.pos for _ in range(self.future_steps)])
        
        # update obstacle and pedestrians
        # self.obstacles = self.updateObstaclesStates(model_states, self.obstacles)
        self.pedestrians_list = self.updatePedestrians(model_states)

    def getGraphState(self, action=None, last_state=None):

        graph_nodes = [self.robot, self.robot_goal] + self.pedestrians_list

        curr_state = create_graph(graph_nodes, self.robot.pos)

        if action is None:
            return curr_state

        # compute  rewards
        current_distance = self.getGoalDistance()
        reaching_goal = current_distance < self.goal_threshold
        too_far = current_distance > self.max_goal_distance

        # make curr_state and last_state similar
        # last_state, curr_state, last_nodes, curr_nodes = remove_uncommon_nodes(deepcopy(last_state), deepcopy(curr_state))

        #corresponding node betwn last_state and current state. NOTE! mask out robot nodes in the last state
        comm_tid = np.intersect1d(last_state.ndata['tid'].numpy(), curr_state.ndata['tid'].numpy())
        last_node_idx =  [last_state.nodes()[last_state.ndata['tid']==tid] for tid in comm_tid]
        curr_node_idx = [curr_state.nodes()[curr_state.ndata['tid']==tid] for tid in comm_tid]

        if self.action_type=='vw':
            theta = last_state.ndata['yaw'].numpy().flatten() + action[:, 1]
            px = last_state.ndata['pos'][:, 0].numpy() + np.cos(theta) * action[:, 0] * self.time_step
            py = last_state.ndata['pos'][:, 1].numpy() + np.sin(theta) * action[:, 0] * self.time_step
            pred_pos = np.stack([px, py], axis=-1)
        else:
            pred_pos = last_state.ndata['pos'] + action * self.time_step

        collision_agents = []
        distance_matrix = np.linalg.norm(pred_pos[:, None, :]-pred_pos[None, :, :], axis=-1)
        
        for i in range(distance_matrix.shape[0]):
            if last_state.ndata['cid'][i]==node_type_list.index('robot_goal'):
                continue
            for j in range(i+1, distance_matrix.shape[1]):
                if last_state.ndata['cid'][j]==node_type_list.index('robot_goal'):
                    continue

                if distance_matrix[i][j] < self.collision_threshold+0.15:
                    # get the corresponding id

                    collision_agents.append(last_state.ndata['tid'][i])
                    collision_agents.append(last_state.ndata['tid'][j])

                    break # one collision is enough

        robot_tid = curr_state.ndata['tid'][curr_state.ndata['cid']==node_type_list.index('robot')]
        collision = True if robot_tid in collision_agents else False
        
        # get the corresponding idx of last_state colliding node in curr_state
        collision_agents_idx = [curr_state.nodes()[curr_state.ndata['tid']==tid] for tid in collision_agents]

        # reward for each node
        total_rewards = np.zeros([curr_state.number_of_nodes(), 1])        
        total_rewards[collision_agents_idx] = -150.

        # compute goal reaching rewards with current action, exclude robot goal 
        goal_mask = (last_state.ndata['cid']!=node_type_list.index('robot_goal')).unsqueeze(1).numpy()
        # robot_mask = (last_state.ndata['cid']==node_type_list.index('robot')).unsqueeze(1).numpy()
        goal_dist = np.linalg.norm(pred_pos - last_state.ndata['goal'].numpy(), axis=-1, keepdims=True)
        goal_reaching_rewards = (goal_dist<self.goal_threshold) * 150. * goal_mask
        total_rewards[curr_node_idx] += goal_reaching_rewards[last_node_idx]

        # goal distance rewards
        goal_dist_rewards = (self.goal_threshold - goal_dist) * 0.1 * goal_mask
        total_rewards[curr_node_idx] += goal_dist_rewards[last_node_idx]

        # compute correct action reward for agents
        ped_mask = (last_state.ndata['cid']==node_type_list.index('pedestrian')).unsqueeze(1).numpy()
        vel_error = np.linalg.norm(last_state.ndata['action'].numpy() - action, axis=-1, keepdims=True)*ped_mask
        # total_rewards -= np.linalg.norm(last_state.ndata['action'].numpy() - action, axis=-1, keepdims=True)
        total_rewards[curr_node_idx] -= vel_error[last_node_idx]
        print('Action Error:', np.mean(vel_error))

        return curr_state, collision, reaching_goal, too_far, total_rewards

    def getState(self, action=None):

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

        collision =  min(scan_range_collision)+1e-6 < self.collision_threshold

        if action==None:
            return scan_range_collision + [0., 0.,self.heading, self.getGoalDistance()]
        
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

        if collision:
            reward = -100

        elif reaching_goal:
            reward = 150
            
        elif too_far:
            reward = -100

        else:
            reward = (self.goal_threshold - goal_distance)*0.1

        if goal_distance < self.last_goal_distance:
            reward += 1#50
        else:
            reward -= 1

        # if abs(self.heading)>1.5:
        #     reward -= 1#20
        self.last_goal_distance = goal_distance

        state = scan_range_collision + [action[0], action[1], self.heading, self.getGoalDistance()]

        return state, collision, reaching_goal, too_far, reward


    def step(self, action, last_state):
        '''
        action: action of the nodes (N, 2) if graph_state else action of the robot (2,) only
        '''
        if self.graph_state:
            robot_action = action[last_state.ndata['tid']==99].flatten() 
        else:
            robot_action = action

        self.vel_cmd = self.action_to_vel_cmd(robot_action, self.action_type)
        self.pub_cmd_vel.publish(self.vel_cmd)
        rospy.sleep(self.time_step)
        self.pub_cmd_vel.publish(Twist())

        # increment step before updating agents
        self.global_step+=1

        self.update_agents()

        if self.graph_state:
            state, collision, reaching_goal, too_far, reward = self.getGraphState(action, last_state)
        else:
            state, collision, reaching_goal, too_far, reward = self.getState(action)

        done=False
        success = False

        if collision:
            rospy.loginfo("Collision!!")
            done = True
            # reward = -100
            self.collision_times += 1

        elif reaching_goal:
            success = True
            # reward = 150#100
            rospy.loginfo('Success!!')
            
        elif too_far:
            done = True
            # reward = -100
            rospy.loginfo('Too Far from Goal!!')


        # NOTE! if goal node is included in the graph, goal must be respawned before calling graph state, otherwise graph create fails. 
        if success:
            self.init_goal(position_check=True, test=self.test)
            self.robot.set_goal(self.goal_x, self.goal_y)

        if done:
            # self.pub_cmd_vel.publish(Twist())
            self.reset()

        return state, reward, done, success

    # add a separate function to initialize goal, delete old goal if exist and respawn new goal
    def init_goal(self, position_check=False, test=False):
        
        self.goal_x, self.goal_y = self.goal_spawnner.getPosition(position_check, test)
        
        if self.goal_spawnner.check_model: self.goal_spawnner.deleteModel()
        self.goal_spawnner.respawnModel()


        rospy.loginfo("Init New Goal : (%.1f, %.1f)", self.goal_x, self.goal_y)

    def reset(self, initGoal=False):
        # reset robot velocity
        self.pub_cmd_vel.publish(Twist())
        # reset scan as well
        # self.scan = None

        # try:
        #     rospy.wait_for_service('gazebo/reset_simulation')
        #     reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        #     reset_proxy()
            
        #     rospy.loginfo('Env Reset')

        # except (rospy.ServiceException) as e:
        #     rospy.loginfo("gazebo/reset_simulation service call failed")

        # reset robot pose and randomly set the orientation
        tmp_state = ModelState()
        tmp_state.model_name = "turtlebot3_burger"
        tmp_state.pose = Pose(self.inital_pos, 
                              euler_to_quaternion([0.0, 0.0, random.uniform(0, 360)])
                              )
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