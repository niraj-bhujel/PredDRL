import os
import sys
import time
import logging
import shutil
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

import dgl
from dgl.heterograph import DGLHeteroGraph

import rospy

from gym.spaces import Box

if './' not in sys.path: 
    sys.path.insert(0, './')
    
from misc.prepare_output_dir import prepare_output_dir
from misc.initialize_logger import initialize_logger
from utils.normalizer import EmpiricalNormalizer
from utils.utils import save_ckpt, load_ckpt, copy_src, create_new_dir
from utils.graph_utils import node_type_list
from vis.vis_graph import network_draw, data_stats
from vis.vis_traj import plot_traj, data_stats
from utils.timer import Timer
from buffer.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class Trainer:
    def __init__(self, policy, env, args, output_dir, phase='train', test_env=None, **kwargs):

        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = args.episode_max_steps if args.episode_max_steps is not None else args.max_steps
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._future_steps = args.future_steps
        self._history_steps = args.history_steps
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

        self._vis_graph = args.vis_graph
        self._vis_traj = args.vis_traj
        self._vis_traj_interval = args.vis_traj_interval

        self._output_dir = output_dir

        # self.timer = Timer()
        # self.r = rospy.Rate(1/self._env.time_step)

        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)



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
        if self._vis_graph:
            self._vis_graph_dir = create_new_dir(self._output_dir + '/vis_graphs/{}/'.format(phase), clean=True)

        if self._vis_traj:
            self._plot_dir = create_new_dir(self._output_dir + '/vis_traj/{}/'.format(phase), clean=True)
        

        # prepare TensorBoard writer
        self._summary_dir = create_new_dir(self._output_dir + '/summary/{}'.format(phase), clean=True)
        self.writer = SummaryWriter(self._summary_dir)
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
        n_success = 0
        n_timeout = 0
        success_rate = 0

        actor_loss = 0
        critic_loss = 0

        episode_start_time = time.perf_counter()
        

        if self._load_memory:
            print('Loading memory ...', self._memory_path)
            try:
                with open(self._memory_path + '.pkl', 'rb') as f:
                    self.replay_buffer = pickle.load(f)
                total_steps=self._policy.n_warmup
            except Exception as e:
                print('Unable to load memory!', e)

        if self._resume_training:
            total_steps = self._policy.iteration

        obs = self._env.reset()

        ##### Begin Training ###########
        while total_steps < self._max_steps:
            self._env.timer.tic()
            
            if self._verbose>0:
                print('Step - {}/{}'.format(total_steps, self._max_steps))    

            if total_steps < self._policy.n_warmup:
                robot_action = self._env.sample_robot_action(self._sampling_method)
                action = obs.ndata['future_vel'].numpy()
                action[obs.ndata['cid']==node_type_list.index('robot')] = np.tile(robot_action, self._future_steps)

            else:
                action = self._policy.get_action(obs) #(num_nodes, future*2)

            next_obs, reward, done, success = self._env.step(action, obs)          

            if isinstance(next_obs, DGLHeteroGraph):
                robot_action = action[obs.ndata['cid']==node_type_list.index('robot')].flatten()
            else:
                robot_action = action

            if self._vis_traj and total_steps%self._vis_traj_interval==0:
                obs_traj = obs.ndata['history_pos'].view(-1, self._history_steps, 2).numpy()
                gt_traj = obs.ndata['future_pos'].view(-1, self._future_steps, 2).numpy()
                pred_traj = obs.ndata['pos'].unsqueeze(1).numpy() + action.reshape(-1, self._future_steps, 2).cumsum(axis=1)*obs.ndata['dt'].unsqueeze(-1).numpy()
                plot_traj(obs_traj, gt_traj, pred_traj[:, None, :, :], 
                          ped_ids=obs.ndata['tid'].numpy(), 
                          extent={'x_min': -1., 'x_max': 15., 'y_min': -1., 'y_max': 15.}, 
                          limit_axes=True, legend=True, counter=total_steps, 
                          save_dir=self._plot_dir)


            if self._vis_graph and total_steps<100:
                network_draw(next_obs,
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
            self.replay_buffer.add([obs, action, reward, next_obs, done])

            if total_steps==self._policy.n_warmup-1:                
                pickle.dump(self.replay_buffer, open(self._memory_path + '.pkl', 'wb'))

            self.writer.add_scalar(self._policy.policy_name + "/act_0", robot_action[0], total_steps)
            self.writer.add_scalar(self._policy.policy_name + "/act_1", robot_action[1], total_steps)
            self.writer.add_scalar(self._policy.policy_name + "/reward", reward, total_steps)

            episode_steps += 1
            episode_return += reward
            total_steps += 1
            fps = episode_steps / (time.perf_counter() - episode_start_time)

            # update
            obs = next_obs

            time_out = episode_steps == self._episode_max_steps
            if done or time_out:
                obs = self._env.reset()

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            if total_steps > self._policy.n_warmup-1:
                # count success rate only after warmup
                if success:
                    n_success += 1

                if time_out:
                    n_timeout += 1

                if done or success or time_out:
                    n_episode += 1

                    self.logger.info("Total Steps: {}, Episode: {}, Success Rate:{:.2f}".format(total_steps, n_episode, n_success/n_episode))

                self.writer.add_scalar("Common/episode_return", episode_return, total_steps)
                self.writer.add_scalar("Common/success_rate", n_success/n_episode, total_steps)
                self.writer.add_scalar("Common/timeout_rate", n_timeout/n_episode, total_steps)
                self.writer.add_scalar("Common/collisions_rate", self._env.collision_times/n_episode, total_steps)
                self.writer.add_scalar("Common/discomforts", self._env.discomforts, total_steps)
                self.writer.add_scalar("Common/dones", int(done), total_steps)


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

            self._env.timer.toc()
            self.writer.add_scalar("Common/fps", self._env.timer.fps, total_steps)
            if self._verbose>1:
                print('Time per step:', self._env.timer.diff)

        save_ckpt(self._policy, self._output_dir, total_steps)
        self.writer.close()

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

        # graph state  only
        elif isinstance(obs[0], DGLHeteroGraph):
            obs = dgl.batch(obs).to(device)
            next_obs = dgl.batch(next_obs).to(device)

        # laser state only
        elif isinstance(obs[0], list):
            obs = torch.Tensor(np.stack(obs, 0)).to(device)
            next_obs = torch.Tensor(np.stack(next_obs, 0)).to(device)
        
        else:
            raise Exception("Data type not known!")

        act = torch.Tensor(np.concatenate(act)).to(device)
        rew = torch.Tensor(np.array(rew)).view(-1, 1).to(device)
        done = torch.Tensor(np.array(done)).view(-1, 1).to(device)

        if self._use_prioritized_rb:
            weights = torch.Tensor(np.stack(weights, 0)).to(device)
        else:
            weights = torch.Tensor(np.ones((batch_size,))).to(device)

        return {'obs':obs, 'act':act, 'next_obs':next_obs, 'rew':rew, 'done':done, 'weights':weights, 'idxes':idxes}
        
    def evaluate_policy(self, ):
        
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

                print('STEP-[{}/{}]'.format(j, self._episode_max_steps))
                print('Robot Action:', np.round(action, 2))
                print('Reward:', np.round(reward, 2))

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
    from utils.state_utils import state_dims

    from policy.td3 import TD3
    from policy.ddpg import DDPG
    from policy.ddpg_graph import GraphDDPG
    from policy.td3_graph import GraphTD3

    from env import Env

    from gazebo_msgs.srv import DeleteModel
    from gazebo_msgs.msg import ModelStates

    policies = {'td3': TD3, 'ddpg':DDPG, 'ddpg_graph': GraphDDPG, 'td3_graph': GraphTD3}

    rospy.init_node('pred_drl', disable_signals=True)

    parser = args.get_argument()
    args = parser.parse_args()


    # update state dim
    state_dims['history_pos'] = args.history_steps * state_dims['pos']
    state_dims['history_vel'] = args.history_steps * state_dims['vel']
    state_dims['history_disp'] = args.history_steps * state_dims['pos']
    state_dims['future_pos'] = args.future_steps * state_dims['pos']
    state_dims['future_vel'] = args.future_steps * state_dims['vel']
    state_dims['future_disp'] = args.future_steps * state_dims['pos']
    args.state_dims = state_dims

    # load net params
    with open("./preddrl_td3/scripts/net_params.yaml", 'r') as f:
        net_params = yaml.load(f, Loader = yaml.FullLoader)

    print(net_params)


    # prepare output dir
    if not args.evaluate:
        # prepare log directory
        suffix = '_'.join(['run%d'%args.run,
                          '%s'%args.policy,
                        'warmup_%d'%args.n_warmup,
                        'bs%d'%args.batch_size,
                        'ht%d'%args.history_steps,
                        'ft%d'%args.future_steps,
                        'input_%s'%'_'.join(args.input_states),
                        'h%d'%net_params['hidden_dim'],
                        'l%d'%net_params['num_layers'],
                        ])

        if args.prefix is not None:
            suffix += '_%s'%args.prefix

        output_dir = prepare_output_dir(args=args, 
                                        user_specified_dir=args.logdir, 
                                        # time_format='%Y_%m_%d_%H-%M-%S',
                                        time_format='%Y_%m_%d',
                                        suffix=suffix
                                        )
        # backup scripts
        copy_src('./preddrl_td3/scripts', output_dir + '/scripts')
        pickle.dump(args, open(output_dir + '/args.pkl', 'wb'))

    elif args.evaluate:
        output_dir = './preddrl_td3/results/'
        output_dir += '2021_12_16_run0_ddpg_graph_warmup_2000_bs100_input_history_disp_vel_vpref_h256_l2_pred_future'
        # output_dir = trainer._output_dir
        

    # create enviornment
    env = Env(args, graph_state=True if 'graph' in args.policy else False)

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

    trainer = Trainer(policy, env, args, output_dir, phase='train' if not args.evaluate else 'test')
    trainer.set_seed(args.seed)


    try:
        if args.evaluate:
            print('-' * 89)
            print('Evaluating ...', output_dir)
            policy = load_ckpt(policy, output_dir)
            trainer.evaluate_policy()  # 每次测试都会在生成临时文件，要定期处理

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