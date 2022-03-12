import os
import sys
import time
import logging
import shutil
import random
import numpy as np
from collections import deque
from copy import deepcopy, copy
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

import dgl
from dgl.heterograph import DGLHeteroGraph


if './' not in sys.path: 
    sys.path.insert(0, './')
    
from misc.prepare_output_dir import prepare_output_dir
from misc.initialize_logger import initialize_logger
from utils.normalizer import EmpiricalNormalizer
from utils.utils import save_ckpt, load_ckpt, copy_src, create_new_dir
from utils.graph_utils import node_type_list, remove_uncommon_nodes
from vis.vis_graph import network_draw, data_stats
from vis.vis_traj import vis_traj_helper
from utils.timer import Timer
from utils.info import *
from buffer.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

try:
    import rospy
    from gym.spaces import Box
except Exception:
    pass

class Trainer:
    def __init__(self, policy, env, args, phase='train', test_env=None, **kwargs):

        # experiment settings
        self._max_steps = args.max_steps
        self._episode_max_steps = args.episode_max_steps if args.episode_max_steps is not None else args.max_steps
        self._save_model_interval = args.save_model_interval
        self._logdir = args.logdir

        self._input_states = args.input_states
        self._input_edges = args.input_edges
        self._pred_states = args.pred_states
        self._pred_edges = args.pred_edges
        self._future_steps = args.future_steps
        self._history_steps = args.history_steps
        self._pred_steps = args.pred_steps
        
        # replay buffer
        self._buffer_size = args.buffer_size
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step

        # test settings
        self._test_interval = args.test_interval
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._evaluate = args.evaluate
        self._resume_training = args.resume_training

        self._policy = policy
        self._policy._verbose = args.verbose

        self._sampling_method = args.sampling_method

        self._load_memory = args.load_memory

        self._device = policy.device

        self._env = env
        self._test_env = self._env if test_env is None else test_env
        self._verbose = args.verbose
        self._dataset = args.dataset

        self._vis_graph = args.vis_graph
        self._save_graph = args.save_graph
        self._vis_traj = args.vis_traj
        self._vis_traj_interval = args.vis_traj_interval

        self._output_dir = args.output_dir


        self.logger = initialize_logger(logging_level=logging.getLevelName(args.logging_level), 
                                        output_dir=args.output_dir)

        # prepare buffer
        if self._use_prioritized_rb:
            self.replay_buffer = PrioritizedReplayBuffer(size=self._buffer_size,
                                                         use_nstep=self._use_nstep_rb,
                                                         n_step = self._n_step,
                                                         beta_frames=self._max_steps)
        else:
            self.replay_buffer = ReplayBuffer(size=self._buffer_size)


        self._memory_path = create_new_dir('./preddrl_td3/memory')
        self._memory_path += '/{}_nwarmup{}_sampling_{}'.format(type(self.replay_buffer).__name__, self._policy.n_warmup, self._sampling_method)

        # tensorboard visualization
        self._summary_dir = create_new_dir(args.output_dir + '/summary/{}/'.format(phase), clean=True)
        # create writer
        self.writer = SummaryWriter(self._summary_dir)
        self._env.writer = self.writer
        self._policy.writer = self.writer

        # graph visualization (for bothe train and eval)
        if args.vis_graph or args.save_graph:
            self._vis_graph_dir = create_new_dir(args.output_dir + '/vis_graphs/{}/'.format(phase), clean=True)

        if args.vis_traj:
            self._vis_traj_dir = create_new_dir(args.output_dir + '/vis_traj/{}/'.format(phase), clean=True)

    def set_seed(self, seed):
        #setup seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __call__(self):

        print('**** Training Begins ****')
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        
        n_episode = 1
        n_success = 0
        n_collision = 0
        n_discomfort = 0
        n_timeout = 0


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

        # initialize 
        self._env.initialize_agents()        
        obs = self._env.reset()

        ##### Begin Training ###########
        while total_steps < self._max_steps:
            self._env.timer.tic()
            
            if self._verbose>0:
                print('Step - {}/{}'.format(total_steps, self._max_steps))    

            robot_idx = obs.ndata['cid']==node_type_list.index('robot')
            if total_steps < self._policy.n_warmup:
                robot_action = self._env.sample_robot_action(self._sampling_method)
                action = obs.ndata['future_vel'].numpy()
                action = action.reshape(-1, self._future_steps, 2)[:, :self._pred_steps, :].reshape(-1, self._pred_steps*2)
                action[robot_idx] = np.tile(robot_action, self._pred_steps)

            else:
                obs, action = self._policy.get_action(obs) #(num_nodes, future*2)

            next_obs, reward, done, success, info = self._env.step(action, deepcopy(obs))

            if self._vis_traj and total_steps%self._vis_traj_interval==0:
                vis_traj_helper(obs, action, self._pred_steps, total_steps, self._vis_traj_dir)


            if self._vis_graph and total_steps<100:
                network_draw(deepcopy(obs),
                             show_node_labels=True, node_labels=['tid'],
                             show_edge_labels=False, edge_labels=['dist'],
                             show_legend=False,
                             fsuffix = 'episode_step%d'%episode_steps,
                             counter=total_steps,
                             save_dir=self._vis_graph_dir, 
                             extent = data_stats[self._dataset],
                             )
            if self._save_graph and total_steps<100:
                pickle.dump(next_obs, open(self._vis_graph_dir + 'step{}_episode_step{}.pkl'.format(total_steps, episode_steps), "wb"))

            self.writer.add_scalar(self._policy.policy_name + "/robot_act0", action[robot_idx].flatten()[0], total_steps)
            self.writer.add_scalar(self._policy.policy_name + "/robot_act1", action[robot_idx].flatten()[1], total_steps)
            self.writer.add_scalar(self._policy.policy_name + "/robot_reward", reward[robot_idx], total_steps)
            self.writer.add_scalar(self._policy.policy_name + "/collision_penalty", self._env.collision_penalty, total_steps)
            self.writer.add_scalar(self._policy.policy_name + "/success_reward", self._env.success_reward, total_steps)

            # remove uncommon nodes
            _obs, _next_obs, _obs_nidx, _next_obs_nidx = remove_uncommon_nodes(deepcopy(obs), deepcopy(next_obs))

            assert _obs.number_of_nodes() == _next_obs.number_of_nodes(), "obs is not similar to next_obs"
            
            self.replay_buffer.add([_obs, action[_obs_nidx], reward[_obs_nidx], _next_obs, done[_obs_nidx]])
            # self.replay_buffer.add([obs, action, reward, next_obs, done])

            if total_steps==self._policy.n_warmup-1:                
                pickle.dump(self.replay_buffer, open(self._memory_path + '.pkl', 'wb'))

            episode_steps += 1
            episode_return += reward[robot_idx]
            total_steps += 1

            # update
            obs = next_obs

            if isinstance(info, Collision) or isinstance(info, Lost) or isinstance(info, Timeout):
                obs = self._env.reset()
                # reset episode
                episode_steps = 0 
                episode_return = 0

            if total_steps > self._policy.n_warmup-1:
                # count success rate only after warmup
                if isinstance(info, ReachGoal):
                    n_success += 1

                if isinstance(info, Collision):
                    n_collision += 1

                if isinstance(info, Timeout):
                    n_timeout += 1

                if isinstance(info, Discomfort):
                    n_discomfort += 1

                if isinstance(info, Collision) or isinstance(info, Lost) or isinstance(info, Timeout) or isinstance(info, ReachGoal):
                    n_episode += 1

                    self.logger.info("Total Steps: {}, Episode: {}, Success Rate:{:.2f}".format(total_steps, n_episode, n_success/n_episode))


                self.writer.add_scalar("Common/episode_return", episode_return, total_steps)
                self.writer.add_scalar("Common/success_rate", n_success/n_episode, total_steps)
                self.writer.add_scalar("Common/timeout_rate", n_timeout/n_episode, total_steps)
                self.writer.add_scalar("Common/collisions_rate", n_collision/n_episode, total_steps)
                self.writer.add_scalar("Common/discomfort_rate", n_discomfort/total_steps, total_steps)

                if total_steps % self._policy.update_interval==0 and len(self.replay_buffer)>self._policy.batch_size:

                    samples = self.sample_data(batch_size=self._policy.batch_size,
                                               device=self._policy.device)

                    self._policy.train_step(samples["obs"], 
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

                if total_steps % self._save_model_interval == 0:
                    save_ckpt(self._policy, self._output_dir, total_steps)

                self._env.schedulers_step(total_steps-self._policy.n_warmup)

            self._env.timer.toc()
            self.writer.add_scalar("Common/fps", self._env.timer.fps, total_steps)
            if self._verbose>1:
                print('Time per step:', self._env.timer.diff)

        save_ckpt(self._policy, self._output_dir, total_steps)
        self.writer.close()

    def sample_data(self, batch_size, device):

        sampled_data, idxes, weights = self.replay_buffer.sample(batch_size)

        obs, act, rew, next_obs, done = map(list, zip(*sampled_data))

        ndata = list(set(self._input_states + self._pred_states + ['max_action', 'history_vel', 'future_vel']))
        edata = list(set(self._input_edges + self._pred_edges + ['history_dist', 'future_dist']))
        
        obs = dgl.batch(obs, ndata=ndata, edata=edata).to(device)
        next_obs = dgl.batch(next_obs, ndata=ndata, edata=edata).to(device)

        # obs = dgl.batch(obs).to(device)
        # next_obs = dgl.batch(next_obs).to(device)

        act = torch.Tensor(np.concatenate(act)).to(device)
        rew = torch.Tensor(np.concatenate(rew)).view(-1, 1).to(device)
        done = torch.Tensor(np.concatenate(done)).view(-1, 1).to(device)

        if self._use_prioritized_rb:
            weights = torch.Tensor(np.stack(weights, 0)).to(device)
        else:
            weights = torch.Tensor(np.ones((batch_size,))).to(device)

        return {'obs':obs, 'act':act, 'next_obs':next_obs, 'rew':rew, 'done':done, 'weights':weights, 'idxes':idxes}
        
    def evaluate_policy(self, ):
        self._env.initialize_agents()
        avg_test_return = 0.

        for i in range(self._test_episodes):
            episode_return = 0.

            obs = self._env.reset()

            for j in range(self._episode_max_steps):

                obs, action = self._policy.get_action(obs)
                robot_idx = obs.ndata['cid']==node_type_list.index('robot')

                # if self._vis_traj and j<100:
                #     vis_traj_helper(obs, action, self._pred_steps, i, self._vis_traj_dir)

                if self._vis_graph and j<100:
                    # obs.edata['sigma'] = obs.edata['e_ij'].sigmoid().mean(-1, keepdims=True)
                    network_draw(deepcopy(obs),
                                 show_node_label=False, node_labels=['tid'],
                                 show_edge_labels=True, edge_labels=['dist'],
                                 show_legend=False,
                                 fsuffix = 'episode_step%d'%i,
                                 counter=j,
                                 save_dir=self._vis_graph_dir, 
                                 extent = data_stats[self._dataset]
                                 )

                    pickle.dump(obs, open(self._vis_graph_dir + 'step{}_episode_step{}.pkl'.format(i, j), "wb"))

                next_obs, reward, done, success, info = self._env.step(action, deepcopy(obs))

                if self._verbose>0:
                    print('STEP-[{}/{}]'.format(j, self._episode_max_steps))
                    print('Action:', np.round(action, 2))
                    print('GT Action:', np.round(obs.ndata['vpref'].cpu().numpy(), 2))
                    print('Reward:', np.round(reward, 2))

                episode_return += reward.mean()

                obs = next_obs

                if done[robot_idx]:
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
    from policy.ddpg_graph_vae import GraphVAE
    from policy.td3_graph import GraphTD3

    from env import Env

    from gazebo_msgs.srv import DeleteModel
    from gazebo_msgs.msg import ModelStates

    rospy.init_node('pred_drl', disable_signals=True)

    policies = {'td3': TD3, 'ddpg':DDPG, 'ddpg_graph': GraphDDPG, 'td3_graph': GraphTD3, 'graph_vae': GraphVAE}

    parser = args.get_argument()
    args = parser.parse_args()


    # update state dim
    state_dims['history_pos'] = args.history_steps * state_dims['pos']
    state_dims['history_vel'] = args.history_steps * state_dims['vel']
    state_dims['history_disp'] = args.history_steps * state_dims['pos']
    state_dims['history_diff'] = args.history_steps * state_dims['diff']
    state_dims['history_dist'] = args.history_steps * state_dims['dist']

    state_dims['future_pos'] = args.future_steps * state_dims['pos']
    state_dims['future_vel'] = args.future_steps * state_dims['vel']
    state_dims['future_disp'] = args.future_steps * state_dims['pos']
    state_dims['future_diff'] = args.future_steps * state_dims['diff']
    state_dims['future_dist'] = args.future_steps * state_dims['dist']
    args.state_dims = state_dims

    # load net params
    with open("./preddrl_td3/scripts/net_params.yaml", 'r') as f:
        net_params = yaml.load(f, Loader = yaml.FullLoader)

    print(net_params)

    if args.evaluate:
        args.output_dir = args.logdir + '/' 
        args.output_dir += '2022_03_07_graph_vae_warm_1000_bs100_ht4_ft4_pt1_in_pos_vpref_history_vel_pred_future_vel'

        with open(args.output_dir + '/args.pkl', 'rb') as f:
            args = pickle.load(f)

    else:
        # prepare log directory
        suffix = '_'.join(['run-%d'%args.run,
                            '%s'%args.policy,
                            'warm%d'%args.n_warmup,
                            'bs%d'%args.batch_size,
                            'ht%d'%args.history_steps,
                            'ft%d'%args.future_steps,
                            'pt%d'%args.pred_steps,
                            'in_%s'%'_'.join(args.input_states),
                            # 'pred_%s'%'_'.join(args.pred_states),
                            # 'h%d'%net_params['hidden_dim'],
                            # 'l%d'%net_params['num_layers'],
                            '%s'%args.layer,
                            ])

        if args.prefix is not None:
            suffix += '_%s'%args.prefix

        args.output_dir = prepare_output_dir(args=args, 
                                            user_specified_dir=args.logdir, 
                                            # time_format='%Y_%m_%d_%H-%M-%S',
                                            time_format='%Y_%m_%d',
                                            suffix=suffix
                                            )

        # backup scripts
        copy_src('./preddrl_td3/scripts', args.output_dir + '/scripts')
        pickle.dump(args, open(args.output_dir + '/args.pkl', 'wb'))



    # create enviornment
    env = Env(args)

    print('Creating policy ... ')
    policy = policies[args.policy](state_shape=env.observation_space.shape,
                                    action_dim=env.action_space.high.size,
                                    max_action=env.action_space.high[1],
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
    # print({k:v for k, v in policy.__dict__.items() if k[0]!='_'})

    trainer = Trainer(policy, env, args, phase='train' if not args.evaluate else 'test')
    trainer.set_seed(args.seed)


    try:
        if args.evaluate:
            print('-' * 89)
            print('Evaluating ...', args.output_dir)
            policy = load_ckpt(policy, args.output_dir)
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