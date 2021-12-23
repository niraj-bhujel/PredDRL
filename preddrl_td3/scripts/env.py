import rospy
import math
import random
import torch
import numpy as np
from copy import deepcopy, copy

from gym import spaces
from gym.utils import seeding
from math import pi, cos, sin
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped, Quaternion, Vector3
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState

from spawnner.respawnGoal import RespawnGoal
from spawnner.respawnPeds import RespawnPedestrians
from utils.env_utils import *

from utils.agent import Agent
from utils.graph_utils import create_graph, min_neighbor_distance, node_type_list
from utils.timer import Timer
from policy.orca import ORCA
from preddrl_data.scripts.prepare_data import prepare_data

# # turtlebot3_burger, https://emanual.robotis.com/docs/en/platform/turtlebot3/features/#features
SelfD=0.1 # Robot effective distance/radius
SelfL=0.16 # Length of wheel axels in meters
# turtlebot2
# SelfD = 0.175
# SelfL = 0.23
class Env:
    def __init__(self, args, graph_state=False, robot_name='turtlebot3_burger'):

        self.graph_state = graph_state
        self.robot_name = robot_name

        self.stage = args.stage
        self.dataset = args.dataset
        self.verbose = args.verbose
        self.spawn_pedestrians = args.spawn_pedestrians

        self.pred_steps = args.pred_steps
        self.future_steps = args.future_steps
        self.history_steps = args.history_steps

        self.state_dims = args.state_dims
        self.pred_states = args.pred_states

        self.gx = 0
        self.gy = 0

        self.goal_threshold = 0.3
        self.collision_threshold = 0.2

        self.robot_radius = 0.2
        self.discomfort_zone = 0.4

        self.inital_pos = Point(10.0, 10.0, 0.)

        self.minLinearSpeed = 0.
        self.maxLinearSpeed = 0.7
        self.maxAngularSpeed = 2.

        self.action_space = spaces.Box(low=np.array([-self.minLinearSpeed, -self.minLinearSpeed]), 
                                       high=np.array([self.maxLinearSpeed, self.maxLinearSpeed]), 
                                       dtype=np.float32)          

        self.observation_space = spaces.Box(low=np.array([0.0]*2), 
                                            high=np.array([self.maxLinearSpeed]*2), 
                                            dtype=np.float32)

        self.time_step = 0.25
        self.timer = Timer() # set by trainer

        self.max_goal_dist = 15.
        self.last_goal_dist = 100.

        self.global_step = 0
        self.frame_num = 0

        self.collision_times = 0
        self.discomforts = 0

        self.goal_spawnner = RespawnGoal(self.stage, self.gx, self.gy)

        self.pedestrian_spawnner = RespawnPedestrians()

        rospy.Subscriber('gazebo/model_states', ModelStates, self.model_states_cbk)

    def model_states_cbk(self, model_states):
        self.model_states = model_states

    def initialize_agents(self, ):

        self.orca = ORCA(self.time_step)

        self.robot = Agent(node_id=99, node_type='robot', time_step=self.time_step)

        self.robot_goal = Agent(node_id=98, node_type='robot_goal', time_step=self.time_step)

        if self.stage==7:
            self.pedestrians, self.ped_frames, _  = prepare_data('./preddrl_data/{}.txt'.format(self.dataset), 
                                                            target_frame_rate=2*1/self.time_step,
                                                            max_peds=40)
            print("Total pedestrians:{}, Total frames:{}".format(len(self.pedestrians), len(self.ped_frames)))
        
        self.init_goal()
            

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getGoalDistance(self):
        goal_distance = round(math.hypot(self.gx - self.position.x, self.gy - self.position.y), 2)

        return goal_distance

    def sample_robot_action(self, policy='uniform'):
        if policy =='uniform':
            action = self.action_space.sample()
            print("uniform vel:", action)

        elif policy == 'vpref':
            action = self.robot.preferred_vel() #+ np.random.normal(0, 0.1, size=(2,))
            print("V_pref(sampled):", action)

        else:
            action, _ = self.orca.predict(self.robot, humans=self.pedestrians_list)
            print('ORCA Vel:', np.round(action, 3))

        return action


    def updatePedestrians(self, ):
        '''
        Get pedestrians at current step, update and return pedestrian states, 
        Meanwhile spawn the current pedestrians to gazebo
        '''
        if self.frame_num>len(self.ped_frames):
            t = self.frame_num%len(self.ped_frames)
        else:
            t = self.frame_num

        self.pedestrians_list = [ped for ped in self.pedestrians if t>=ped.first_timestep and t<ped.last_timestep]

        ped_states = {}
        # update action of the current peds
        for ped in self.pedestrians_list:

            # history
            ped.history = ped.get_history_at(t, self.history_steps)
            ped.futures = ped.get_futures_at(t, self.future_steps)

            # current state
            state = ped.get_state_at(t) 
            ped.set_state(state.px, state.py, state.vx, state.vy, state.gx, state.gy, state.theta)
            
            # next action
            next_state = ped.get_state_at(t+1)

            ped.set_action((next_state.vx, next_state.vy))

            ped_states[ped.id] = ped.deserialize_state(state)

        if self.spawn_pedestrians:
                self.pedestrian_spawnner.respawn(ped_states)
            

    def update_robot_goal(self, ):
        # update robot goal node
        self.robot_goal.update_history(self.gx, self.gy, 0., 0., self.gx, self.gy, theta=0.0)
        self.robot_goal.set_state(self.gx, self.gy, 0., 0., self.gx, self.gy, theta=0.0)
        self.robot_goal.futures = np.array([(self.gx, self.gy, 0., 0., self.gx, self.gy, 0.) for _ in range(self.future_steps)])
        self.robot_goal.history = np.array([(self.gx, self.gy, 0., 0., self.gx, self.gy, 0.) for _ in range(self.history_steps)])

    def update_robot(self, robot_action):

        # update robot state
        self.robot.update_history(self.position.x, self.position.y, self.linear.x, self.linear.y, self.gx, self.gy, theta=self.yaw)
        self.robot.set_state(self.position.x, self.position.y, self.linear.x, self.linear.y, self.gx, self.gy, theta=self.yaw)
        self.robot.set_action(robot_action)
        # update robot history
        self.robot.history = self.robot.get_history(self.history_steps)

        # update robot futures
        self.robot.futures = np.zeros((self.future_steps, 7))
        px, py = self.robot.pos
        vx, vy = self.robot.preferred_vel()
        for t in range(self.future_steps):
            px = px + vx * self.time_step
            py = py + vy * self.time_step
            self.robot.futures[t] = (px, py, vx, vy, self.gx, self.gy, math.atan2(vy, vx))

    def update_agents(self, robot_action=(0., 0.)):

        self.update_robot(robot_action)
        self.update_robot_goal()
        # update pedestrians
        self.updatePedestrians()

        
    def getGraphState(self, ):
        #NOTE: Ensure nodes are updated before creating graph by calling update_agents()
        nodes = [self.robot, self.robot_goal] + self.pedestrians_list
        state = create_graph(nodes, self.state_dims)

        return state


    def step(self, action, last_state):

        reward = 0

        # # compute correct action reward for agents
        gt_action = last_state.ndata['future_vel'].view(-1, self.future_steps, 2)[:, :self.pred_steps, :].numpy()
        action_error = np.mean(np.abs(gt_action.reshape(-1, self.pred_steps*2) - action), axis=-1, keepdims=True)
        self.writer.add_scalar("Common/action_error", np.mean(action_error), self.global_step)
        reward -= action_error

        robot_idx = last_state.ndata['cid']==node_type_list.index('robot')
        goal_mask = last_state.ndata['cid']!=node_type_list.index('robot_goal')
        goal_mask = goal_mask.numpy().reshape(-1, 1)

        action = action.reshape(-1, self.pred_steps, 2)
        curr_pos = copy(last_state.ndata['pos'].numpy())
        last_goal_dist = np.linalg.norm((curr_pos - last_state.ndata['goal'].numpy()), axis=-1, keepdims=True) # (num_nodes, 1)
        for t in range(self.pred_steps):

            curr_action = action[:, t, :]

            # predicted pos of all nodes
            curr_pos = curr_pos + curr_action * last_state.ndata['dt'].numpy()
            
            robot_action = curr_action[robot_idx].flatten()
            robot_pos = curr_pos[robot_idx].flatten()

            self.position.x = robot_pos[0]
            self.position.y = robot_pos[1]
            self.linear.x = robot_action[0]
            self.linear.y = robot_action[1]
            self.yaw = math.atan2(self.linear.y, self.linear.x)

            if self.verbose>0:
                print('Action:', np.round(robot_action, 3))
                print("Vpref:", np.round(self.robot.preferred_vel(), 3))
            
            self.set_robot_pose(self.position, self.yaw)

            # distance between agents
            dmat = np.linalg.norm(curr_pos[:, None, :] - curr_pos[None, :, :], axis=-1)
            min_agent_dist = np.min(dmat + np.diag(np.full(dmat.shape[0], 1e3)), axis=-1, keepdims=True)

            # penalty for collisions for all node
            collisions = min_agent_dist<self.collision_threshold
            reward -= collisions*goal_mask*100

            # agent goal distance
            curr_goal_dist = np.linalg.norm((curr_pos - last_state.ndata['goal'].numpy()), axis=-1, keepdims=True) # (num_nodes, 1)
            
            # reward for reaching goal
            reaching_goal = curr_goal_dist<self.goal_threshold
            reward += reaching_goal*goal_mask*100

            # penalty for moving too far
            too_far = curr_goal_dist>self.max_goal_dist
            reward -= too_far*goal_mask*100

            # reward/penalty for moving toward/away from goal
            reward += np.where(last_goal_dist-curr_goal_dist>0.01, 1, -1)

            if self.verbose>0:
                print('Reward:', round(reward.mean(), 3))

            done = too_far + collisions
            success = reaching_goal

            # init goal if success
            if reaching_goal[robot_idx]: 
                self.init_goal()
            self.update_agents(robot_action=robot_action)

            self.frame_num += 1

            # rewards for robot
            if reaching_goal[robot_idx]:
                rospy.loginfo('SUCCESS !!')
                break

            elif collisions[robot_idx]:
                self.collision_times += 1
                rospy.loginfo("COLLISION !!")
                break

            elif too_far[robot_idx]:
                rospy.loginfo('TOO FAR !!')
                break

        state = self.getGraphState()

        self.global_step+=1

        return state, reward, done, success

    def init_goal(self, position_check=False, test=False):
        self.robot_goal.reset()
        self.gx, self.gy = self.goal_spawnner.getPosition(position_check, test)

        if self.goal_spawnner.modelName in self.model_states.name:
        # if self.goal_spawnner.check_model:
            self.goal_spawnner.deleteGoal()
        
        self.goal_spawnner.spawnGoal()

        rospy.loginfo("New Goal : (%.1f, %.1f)", self.gx, self.gy)

    def set_robot_pose(self, position, yaw):

        tmp_state = ModelState()
        tmp_state.model_name = "turtlebot3_burger"
        tmp_state.pose = Pose(position, euler_to_quaternion([0.0, 0.0, yaw]))
        tmp_state.reference_frame = "world"
        rospy.wait_for_service("/gazebo/set_model_state")
        set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        set_model_state(tmp_state)

        if self.verbose>0:
            print('Robot pos set to:', round(position.x, 2), round(position.y, 2))

    def reset(self, initGoal=False):
        print('Resetting env ... ')

        # reset robot history
        self.robot.reset()

        # reset current state
        self.position = deepcopy(self.inital_pos)
        self.yaw = random.uniform(0, 3.14)
        self.linear = Vector3(0., 0., 0.)
        # NOTE: Goal must be updated before computing robot heading     
        # self.heading = compute_heading(self.position.x, self.position.y, self.yaw, self.gx, self.gy) 
        self.set_robot_pose(self.position, self.yaw)

        # update agent before getting state
        self.update_agents()

        state = self.getGraphState()

        return state