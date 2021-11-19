import os
import sys
import time
import logging
import shutil
import random
import numpy as np
from collections import deque
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

import dgl
from dgl.heterograph import DGLHeteroGraph

import rospy

from gym.spaces import Box

if './' not in sys.path: 
    sys.path.insert(0, './')

tracker_path = './preddrl_tracker/scripts/'
if tracker_path not in sys.path:
    sys.path.insert(0, tracker_path)
        
td3_path = './preddrl_td3/scripts_torch'
if not td3_path in sys.path:
    sys.path.insert(0, td3_path)
    
from misc.prepare_output_dir import prepare_output_dir
from misc.initialize_logger import initialize_logger
from utils.normalizer import EmpiricalNormalizer
from utils.utils import save_path, frames_to_gif, save_ckpt, load_ckpt, copy_src, create_new_dir
from utils.graph_utils import node_type_list
from utils.vis_graph import network_draw, data_stats
from utils.timer import Timer
from replay_buffer.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class Trainer:
    def __init__(self, policy, env, args, net_params=None, test_env=None, **kwargs):

        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = args.episode_max_steps if args.episode_max_steps is not None else args.max_steps
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir

        # replay buffer
        self._buffer_size = args.buffer_size
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step

        # test settings
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images
        self._evaluate = args.evaluate
        self._resume_training = args.resume_training

        self._policy = policy
        self._sampling_method = args.sampling_method

        self._device = policy.device

        self._env = env
        self._test_env = self._env if test_env is None else test_env
        self._verbose = args.verbose
        self._dataset = args.dataset

        # self.timer = Timer()
        # self.r = rospy.Rate(1/self._env.time_step)

        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)

        # prepare log directory
        suffix = '_'.join(['%s'%policy.policy_name,
                        'warmup_%d'%policy.n_warmup,
                        'bs%d'%policy.batch_size,
                        # 'seed_%d'%args.seed,
                        # 'stage_%d'%args.stage,
                        # 'episode_step%d'%args.episode_max_steps,
                        'sampling_%s'%args.sampling_method,
                        'input_%s'%'_'.join(args.input_states),
                        'h%d'%net_params['net']['hidden_dim'],
                        'l%d'%net_params['net']['num_layers'],
                        ])

        if self._use_prioritized_rb:
            suffix += '_use_prioritized_rb'
        if self._use_nstep_rb:
            suffix += '_use_nstep_rb'

        if args.prefix is not None:
            suffix += '_%s'%args.prefix

        self._output_dir = prepare_output_dir(args=args, 
                                              user_specified_dir=self._logdir, 
                                              # time_format='%Y_%m_%d_%H-%M-%S',
                                              time_format='%Y_%m_%d',
                                              suffix=suffix
                                              )

        # backup scripts
        copy_src('./preddrl_td3/scripts_torch', self._output_dir + '/scripts')
        self.logger = initialize_logger(logging_level=logging.getLevelName(args.logging_level), 
                                        output_dir=self._output_dir)

        # prepare buffer
        if self._use_prioritized_rb:
            self.replay_buffer = PrioritizedReplayBuffer(size=self._buffer_size,
                                                         use_nstep=self._use_nstep_rb,
                                                         n_step = self._n_step,
                                                         beta_frames=self._max_steps)
        else:
            self.replay_buffer = ReplayBuffer(size=self._buffer_size)

        self._load_memory = args.load_memory
        self._memory_path = create_new_dir('./preddrl_td3/memory')
        self._memory_path += '/{}_stage{}_nwarmup{}_sampling_{}'.format(type(self.replay_buffer).__name__, 
                                                                         args.stage, 
                                                                         policy.n_warmup, 
                                                                         args.sampling_method)

        # graph visualization
        self._vis_graph = args.vis_graph
        self._vis_graph_dir = self._output_dir + '/graphs/'
        shutil.rmtree(self._vis_graph_dir, ignore_errors=True)
        create_new_dir(self._vis_graph_dir)


        # prepare TensorBoard output
        summary_dir = self._output_dir + '/summary'
        shutil.rmtree(summary_dir, ignore_errors=True)
        create_new_dir(summary_dir)

        self.writer = SummaryWriter(summary_dir)
        self._env.writer = self.writer
        self._policy.writer = self.writer
        self._policy._verbose = self._verbose

    def set_seed(self, seed):
        #setup seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __call__(self):
        print('Begin training ...')
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        
        n_episode = 1
        episode_success = 0
        success_rate = 0

        actor_loss = 0
        critic_loss = 0

        episode_start_time = time.perf_counter()
        

        if self._load_memory:
            print('Loading memory ...')
            try:
                with open(self._memory_path + '.pkl', 'rb') as f:
                    self.replay_buffer = pickle.load(f)
                total_steps=self._policy.n_warmup + 1
            except Exception as e:
                print('Unable to load memory!', e)

        if self._resume_training:
            total_steps = self._policy.iteration

        obs = self._env.reset(initGoal=True) # add initGoal arg by niraj

        ##### Begin Training ###########
        while total_steps < self._max_steps:
            self._env.timer.tic()
            
            if self._verbose>0:
                print('Step - {}/{}'.format(total_steps, self._max_steps))    

            if total_steps < self._policy.n_warmup:
                action  = self._env.sample_robot_action(self._sampling_method)

                if isinstance(obs, DGLHeteroGraph):
                    robot_action = action
                    action = obs.ndata['action'].numpy()
                    action[obs.ndata['tid']==node_type_list.index('robot')] = robot_action

            else:
                action = self._policy.get_action(obs)

            next_obs, reward, done, success = self._env.step(action, obs)          
            
            if isinstance(next_obs, DGLHeteroGraph):
                robot_action = action[obs.ndata['tid']==node_type_list.index('robot')].flatten()
            else:
                robot_action = action

            if self._verbose>0:
                # print(obs.ndata['tid'], next_obs.ndata['tid'])
                # print('Agent Action:', np.round(action, 2))
                print('Robot Action:', np.round(robot_action, 2))
                print('Reward:', np.round(reward, 2))
                # print('Vpref', obs.ndata['vpref'])
                print('Vel cmd:[{:3.3f}, {:3.3f}]'.format(self._env.vel_cmd.linear.x, self._env.vel_cmd.angular.z))
                print('Robot position:', self._env.robot.pos)
            if self._verbose>1:
                print("Pos:{}, Vel:{}, Goal:{}, Goal Distance:{:.2f}".format(np.round(self._env.robot.pos, 2).tolist(),
                                                    np.round(self._env.robot.vel, 2).tolist(), 
                                                    np.round(self._env.robot.goal, 2).tolist(),
                                                    self._env.robot.distance_to_goal()))  

                print('Reward:{:3.3f}'.format(reward))
                # print("Position:[{:2.2f}, {:2.2f}], Goal:[{:.2f}, {:.2f}], Goal Distance:{:.2f}".format(self._env.position.x, self._env.position.y, 
                #                                                                                           self._env.goal_x, self._env.goal_y,
                #                                                                                           self._env.getGoalDistance()))

                print("Pos:{}, Vel:{}, Goal:{}, Goal Distance:{:.2f}".format(np.round(self._env.robot.pos, 2).tolist(),
                                                    np.round(self._env.robot.vel, 2).tolist(), 
                                                    np.round(self._env.robot.goal, 2).tolist(),
                                                    self._env.robot.distance_to_goal()))              
            
            # plot graph, 
            if self._vis_graph and total_steps<100:
                network_draw(obs,
                             show_node_label=True, node_labels=['tid'],
                             show_edge_labels=False, edge_labels=['dist'],
                             show_legend=False,
                             fsuffix = 'episode_step%d'%episode_steps,
                             counter=total_steps,
                             save_dir=self._vis_graph_dir, 
                             show_dirction=True,
                             extent = data_stats[self._dataset]
                             )
                # pickle.dump(obs, open(self._vis_graph_dir + 'step{}_episode_step{}.pkl'.format(total_steps, episode_steps), "wb"))

            # update buffer/memory
            if not episode_steps==0:
                self.replay_buffer.add([obs, action, reward, next_obs, done])

                if total_steps==self._policy.n_warmup:                
                    pickle.dump(self.replay_buffer, open(self._memory_path + '.pkl', 'wb'))

            episode_steps += 1
            episode_return += reward
            total_steps += 1
            fps = episode_steps / (time.perf_counter() - episode_start_time)

            self.writer.add_scalar(self._policy.policy_name + "/act_0", robot_action[0], total_steps)
            self.writer.add_scalar(self._policy.policy_name + "/act_1", robot_action[1], total_steps)
            self.writer.add_scalar(self._policy.policy_name + "/reward", reward, total_steps)

            # update
            obs = next_obs

            if done or episode_steps == self._episode_max_steps:
                obs = self._env.reset()

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            if total_steps > self._policy.n_warmup-1:
                # count success rate only after warmup
                if success:
                    episode_success += 1

                if done or success or episode_steps==0: # episode_steps 0 means episode_steps == self._episode_max_steps see line 271
                    n_episode += 1

                    self.logger.info("Total Steps: {}, Episode: {}, Sucess Rate:{:.2f}".format(
                            total_steps, n_episode, success_rate))


                success_rate = episode_success/n_episode

                self.writer.add_scalar("Common/training_return", episode_return, total_steps)
                self.writer.add_scalar("Common/success_rate", success_rate, total_steps)
                self.writer.add_scalar("Common/collisions_rate", self._env.collision_times/total_steps, total_steps)


                if total_steps % self._policy.update_interval==0 and len(self.replay_buffer)>self._policy.batch_size:

                    samples = self.sample_data(batch_size=self._policy.batch_size,
                                               device=self._policy.device)

                    self._policy.train(samples["obs"], 
                                        samples["act"], 
                                        samples["next_obs"],
                                        samples["rew"], 
                                        samples["done"],
                                        samples["weights"])

                    if self._use_prioritized_rb:
                        priorities = self._policy.compute_td_error(samples["obs"], 
                                                                   samples["act"], 
                                                                   samples["next_obs"],
                                                                   samples["rew"], 
                                                                   samples["done"])

                        self.replay_buffer.update_priorities(samples['idxes'], np.abs(priorities))


                if total_steps % self._test_interval == 0:
                    
                    avg_test_return = self.evaluate_policy(total_steps)

                    self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                        total_steps, avg_test_return, self._test_episodes))

                    self.writer.add_scalar("Common/average_test_return", avg_test_return, total_steps)
                    
                if total_steps % self._save_model_interval == 0:
                    save_ckpt(self._policy, self._output_dir, total_steps)

            # self.r.sleep()
            self._env.timer.toc()
            self.writer.add_scalar("Common/fps", self._env.timer.fps, total_steps)
            if self._verbose>1:
                print('Time per step:', self._env.timer.diff)

        # self.writer.close()
        save_ckpt(self._policy, self._output_dir, total_steps)

    def sample_data(self, batch_size, device):

        sampled_data, idxes, weights = self.replay_buffer.sample(batch_size)

        obs, act, rew, next_obs, done = map(list, zip(*sampled_data))
        # laser and graph state
        if isinstance(obs[0], tuple):
            obs_scan, obs_graph = zip(*obs)

            obs_scan = torch.Tensor(np.stack(obs_scan, 0)).to(device)
            obs_graph = dgl.batch(obs_graph).to(device)
            obs = (obs_scan, obs_graph)

            next_scan, next_graph = zip(*next_obs)
            next_scan = torch.Tensor(np.stack(next_scan, 0)).to(device)
            next_graph = dgl.batch(next_graph).to(device)
            next_obs = (next_scan, next_graph)
        # graph state 
        elif isinstance(obs[0], DGLHeteroGraph):
            obs = dgl.batch(obs).to(device)
            next_obs = dgl.batch(next_obs).to(device)

        # laser state
        elif isinstance(obs[0], list):
            obs = torch.Tensor(np.stack(obs, 0)).to(device)
            next_obs = torch.Tensor(np.stack(next_obs, 0)).to(device)
        
        else:
            raise Exception("Data type not known!")

        act = torch.Tensor(np.concatenate(act)).view(-1, 2).to(device)
        rew = torch.Tensor(np.array(rew)).view(-1, 1).to(device)
        done = torch.Tensor(np.array(done)).view(-1, 1).to(device)

        if self._use_prioritized_rb:
            weights = torch.Tensor(np.stack(weights, 0)).to(device)
        else:
            weights = torch.Tensor(np.ones((batch_size,))).to(device)

        return {'obs':obs, 'act':act, 'next_obs':next_obs, 'rew':rew, 'done':done, 'weights':weights, 'idxes':idxes}
        
    def evaluate_policy(self, total_steps):
        
        if self._normalize_obs:
            self._test_env.normalizer.set_params(*self._env.normalizer.get_params())

        avg_test_return = 0.

        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []

            obs = self._test_env.reset(initGoal=True)

            for j in range(self._episode_max_steps):

                action = self._policy.get_action(obs)

                next_obs, reward, done, success = self._test_env.step(action, obs)

                if isinstance(obs, DGLHeteroGraph):
                    robot_action = action[obs.ndata['tid']==node_type_list.index('robot')].flatten()
                else:
                    robot_action = action
                print('STEP-[{}/{}]'.format(j, self._episode_max_steps))
                print('Robot Action:', np.round(robot_action, 2))
                print('Reward:', np.round(reward, 2))

                # print('Agent Action (P):', np.round(action, 2))
                # print('Agent Action (GT):', np.round(obs.ndata['action'].numpy(), 2))
                # print('Agent Vel (GT):', np.round(obs.ndata['vel'].numpy(), 2))

                episode_return += reward

                obs = next_obs

                if done:
                    break

            avg_test_return += episode_return

        return avg_test_return / self._test_episodes

#%%
if __name__ == '__main__':

    import args
    import yaml

    from utils.utils import *

    from policy.td3 import TD3
    from policy.ddpg import DDPG
    from policy.ddpg_graph import GraphDDPG
    from policy.td3_graph import GraphTD3
    from policy.gcn import GatedGCN

    from env.environment import Env

    from gazebo_msgs.srv import DeleteModel
    from gazebo_msgs.msg import ModelStates

    policies = {'td3': TD3, 'ddpg':DDPG, 'ddpg_graph': GraphDDPG, 'td3_graph': GraphTD3, 'gcn':GatedGCN}

    parser = args.get_argument()
    args = parser.parse_args()
    # print({val[0]:val[1] for val in sorted(vars(args).items())})

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.debug)

    rospy.init_node('pred_drl', disable_signals=True)

    graph_state=True if 'graph' in args.policy else False

    env = Env(test=False, stage=args.stage, graph_state=graph_state, dataset=args.dataset)
    # test_env = Env(test=True, stage=args.stage, graph_state=graph_state, dataset=args.dataset)

    # args.seed = _s._int_list_from_bigint(_s.hash_seed(_s.create_seed()))[0]
    with open("./preddrl_td3/scripts_torch/net_params.yaml", 'r') as f:
        net_params = yaml.load(f, Loader = yaml.FullLoader)

    print('Creating policy ... ')
    policy = policies[args.policy](state_shape=env.observation_space.shape,
                                    action_dim=env.action_space.high.size,
                                    max_action=env.action_space.high,
                                    gpu=args.gpu,
                                    memory_capacity=args.memory_capacity,
                                    batch_size=args.batch_size,
                                    n_warmup=args.n_warmup,
                                    net_params=net_params,
                                    args=args,)

    policy = policy.to(policy.device)
    print(repr(policy))
    print('Total Paramaters:', model_parameters(policy))
    print('Actor Paramaters:', model_parameters(policy.actor))
    print('Critic Paramaters:', model_parameters(policy.critic))
    print({k:v for k, v in policy.__dict__.items() if k[0]!='_'})

    trainer = Trainer(policy, env, args, net_params)
    trainer.set_seed(args.seed)

    if args.resume_training or args.evaluate:
        # eval_path = './preddrl_td3/results/'
        # model_path = '2021_11_11_GraphDDPG_warmup_1000_bs100_stage_7_episode_step1000_sampling_orca_node_prediction'
        policy = load_ckpt(policy, trainer._output_dir)

    try:
        if args.evaluate:
            print('-' * 89)
            print('Evaluating ...', trainer._output_dir)
            trainer.evaluate_policy(1000)  # 每次测试都会在生成临时文件，要定期处理

        else:

            print('-' * 89)
            print('Training %s'%trainer._output_dir)
            trainer()

    # except Exception as e:
    #     print(e)
        # continue

    except KeyboardInterrupt: # this is to prevent from accidental ctrl + c
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)
        
        print("Waiting for gazebo delete_model services...")
        rospy.wait_for_service("gazebo/delete_model")
        delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        print('Clearning existing pedestrians models from', model_states.name)
        [delete_model(model_name) for model_name in model_states.name if 'pedestrian' in model_name]  

    if args.clean:
        print('Cleaning output dir ... ')
        shutil.rmtree(trainer._output_dir)