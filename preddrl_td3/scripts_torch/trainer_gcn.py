import os
import time
import logging
import shutil
import random
import numpy as np
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from gym.spaces import Box

from misc.prepare_output_dir import prepare_output_dir
from misc.initialize_logger import initialize_logger
# from misc.get_replay_buffer import get_replay_buffer
from utils.normalizer import EmpiricalNormalizer
from utils.utils import save_path, frames_to_gif, save_ckpt, load_ckpt, copy_src
from utils.graph_utils import node_type_list
from utils.vis_graph import network_draw

from replay_buffer.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class Trainer:
    def __init__(self, policy, env, args, test_env=None):

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
        self._restore_checkpoint = args.restore_checkpoint

        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env

        self.nstep_buffer = [] 

        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)

        # prepare log directory
        suffix = '_'.join(['%s'%policy.policy_name,
                        'warmup_%d'%self._policy.n_warmup,
                        'bs%d'%policy.batch_size,
                        'seed_%d'%args.seed,
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

        if self._evaluate or self._restore_checkpoint:
            load_ckpt(self._policy, self._output_dir, last_step=1e4)


        # prepare TensorBoard output
        self.writer = SummaryWriter(self._output_dir)
        self._policy.writer = self.writer

        if self._use_prioritized_rb:
            self.replay_buffer = PrioritizedReplayBuffer(size=self._buffer_size,
                                                    beta_frames=self._max_steps)
        else:
            self.replay_buffer = ReplayBuffer(size=self._buffer_size)

        self.n_step_buffer = deque([], self._n_step)
        self.gamma = 0.995
        # self.set_seed(args.seed)
        self._vis_graph = args.vis_graph
        self._plot_dir = self._output_dir + '/graphs/'
        if os.path.exists(self._plot_dir):
            shutil.rmtree(self._plot_dir)
        os.makedirs(self._plot_dir)

    def set_seed(self, seed):
        #setup seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def append_to_replay(self, s, a, r, s_, d):
        # https://github.com/cocolico14/N-step-Dueling-DDQN-PER-Pacman
        self.n_step_buffer.append((s, a, r, s_, d))

        if(len(self.n_step_buffer)<self._n_step):
            return
        
        l_reward, l_next_state, l_done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            l_reward = r + self.gamma * l_reward * (1 - d)
            l_next_state, l_done = (n_s, d) if d else (l_next_state, l_done)
        
        l_state, l_action = self.n_step_buffer[0][:2]

        transition = (l_state, l_action, l_reward, l_next_state, l_done)
        # transitions = tuple(map(lambda x:np.array(x, copy=False, ndmin=1), transition))

        self.replay_buffer.add(transition)

    def __call__(self):

        total_steps = 0
        episode_steps = 0
        episode_return = 0
        
        n_episode = 1
        episode_success = 0
        success_rate = 0

        episode_start_time = time.perf_counter()
        obs = self._env.reset(initGoal=True) # add initGoal arg by niraj

        while total_steps < self._max_steps:
            # print('Step - {}/{}'.format(total_steps, self._max_steps))

            if total_steps < self._policy.n_warmup:
                action = np.stack([self._env.action_space.sample() for _ in range(obs.number_of_nodes())], axis=0)
            else:
                action = self._policy.get_action(obs)

            # only robot action
            robot_action = action[obs.ndata['cid']==node_type_list.index('robot')].flatten()
            # print(robot_action)
            self.writer.add_histogram(self._policy.policy_name + "/robot_actions", robot_action, total_steps)

            next_obs, reward, done, success = self._env.step(robot_action)

            episode_steps += 1
            episode_return += reward
            total_steps += 1

            fps = episode_steps / (time.perf_counter() - episode_start_time)

            if self._vis_graph and total_steps<(self._policy.n_warmup +500):
                network_draw(obs,
                             show_node_label=True, node_label='cid',
                             show_edge_labels=True, edge_label='dist',
                             fsuffix = 'episode_step%d'%episode_steps,
                             counter=total_steps,
                             save_dir=self._plot_dir, 
                             )
            
            # update states
            self.append_to_replay(obs, action, reward, next_obs, done)

            obs = next_obs

            #for success rate
            if done or episode_steps == self._episode_max_steps or success:

                if success and total_steps > self._policy.n_warmup: 
                    episode_success += 1

                if total_steps > self._policy.n_warmup:    
                    n_episode += 1

                success_rate = episode_success/n_episode

                if done or episode_steps == self._episode_max_steps:
                    obs = self._env.reset()

                self.logger.info("Total Epi:{:5} Steps:{:7} Episode Steps:{:5} Return:{:3.4f} SucessRate:{:2.4f} FPS:{:3.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return, success_rate, fps))

                self.writer.add_scalar("Common/training_return", episode_return, total_steps)
                self.writer.add_scalar("Common/success_rate", success_rate, total_steps) 

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter() 


            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._policy.update_interval==0:
                sampled_data, idxes, weights = self.replay_buffer.sample(self._policy.batch_size)

                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*sampled_data)

                self._policy.train(obs_batch)



            if total_steps % self._test_interval == 0:
                
                avg_test_return = self.evaluate_policy(total_steps)

                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))

                self.writer.add_scalar("Common/average_test_return", avg_test_return, total_steps)
                self.writer.add_scalar("Common/fps", fps, total_steps)


            if total_steps % self._save_model_interval == 0:
                save_ckpt(self._policy, self._output_dir, total_steps)

        self.writer.close()

    def evaluate_policy_continuously(self):
        """
        Periodically search the latest checkpoint, and keep evaluating with the latest model until user kills process.
        """
        if self._model_dir is None:
            self.logger.error("Please specify model directory by passing command line argument `--model-dir`")
            exit(-1)

        self.evaluate_policy(total_steps=0)
        while True:
            latest_path_ckpt = tf.train.latest_checkpoint(self._model_dir)

            if self._latest_path_ckpt != latest_path_ckpt:
                self._latest_path_ckpt = latest_path_ckpt

                self._checkpoint.restore(self._latest_path_ckpt)

                self.logger.info("Restored {}".format(self._latest_path_ckpt))

            self.evaluate_policy(total_steps=0)

    def evaluate_policy(self, total_steps):
        
        tf.summary.experimental.set_step(total_steps)

        if self._normalize_obs:
            self._test_env.normalizer.set_params(*self._env.normalizer.get_params())

        avg_test_return = 0.
        if self._save_test_path:
            replay_buffer = get_replay_buffer(self._policy, 
                                              self._test_env, 
                                              size=self._episode_max_steps)

        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []

            obs = self._test_env.reset(initGoal=True)

            for _ in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)

                next_obs, reward, done, success, _ = self._test_env.step(action)

                if self._save_test_path:
                    replay_buffer.add(obs=obs, 
                                      act=action,
                                      next_obs=next_obs, 
                                      rew=reward, 
                                      done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))

                elif self._show_test_progress:
                    self._test_env.render()

                episode_return += reward

                obs = next_obs

                if done:
                    break

            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(total_steps, i, episode_return)

            if self._save_test_path:

                save_path(replay_buffer._encode_sample(np.arange(self._episode_max_steps)), 
                          os.path.join(self._output_dir, prefix + ".pkl"))

                replay_buffer.clear()

            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)

            avg_test_return += episode_return

        if self._show_test_images:
            images = tf.cast(tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3), tf.uint8)

            tf.summary.image('train/input_img', images,)

        return avg_test_return / self._test_episodes


if __name__ == '__main__':
    import sys
    sys.path.insert(0, './')
    sys.path.insert(0, './preddrl_td3/scripts_torch')
    import json
    import rospy
    import args
    from utils.utils import model_attributes

    from policy.td3 import TD3
    from policy.ddpg import DDPG
    from policy.ddpg_graph import GraphDDPG
    from policy.gcn import GatedGCN
    from policy.mlp import SimpleMLP

    from env.environment_stage_3_bk import Env

    policies = {'td3': TD3, 'ddpg':DDPG, 'graph_ddpg': GraphDDPG, 'gcn':GatedGCN, 'mlp': SimpleMLP}

    # from gym.utils import seeding as _s 

    parser = args.get_argument()

    # parser = DDPG.get_argument(parser)

    parser.set_defaults(batch_size=256)
    parser.set_defaults(n_warmup=3000) # 重新训练的话要改回 10000
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(restore_checkpoint=False)
    parser.set_defaults(prefix='gcn_only')
    parser.set_defaults(use_prioritized_rb=False)
    parser.set_defaults(use_nstep_rb=False)

    args = parser.parse_args()
    print({val[0]:val[1] for val in sorted(vars(args).items())})

    # test param, modified by niraj
    if args.evaluate:
        args.test_episodes=50
        args.episode_max_steps = int(1e4)
        args.model_dir = './preddrl_td3/results/compare_network/1conv_2dnn_3input_dropout_1'
        args.show_test_progress=False
        args.save_model_interval = int(1e10)
        args.restore_checkpoint = True

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.debug)

    rospy.init_node('turtlebot3_td3_stage_3', disable_signals=True)

    env = Env()
    test_env = Env()

    # args.seed = _s._int_list_from_bigint(_s.hash_seed(_s.create_seed()))[0]
    with open("./preddrl_td3/scripts_torch/config.json", 'r') as f:
        config = json.load(f)

    policy = policies[args.policy](
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=[400, 300],
        n_warmup=args.n_warmup,
        config=config,
        args=args)

    policy = policy.to(policy.device)
    print(repr(policy))
    for m_name, module in policy.named_children():
        print(m_name, model_attributes(module, verbose=0), '\n')

    # print('offpolicy:', issubclass(type(policy), OffPolicyAgent))
    trainer = Trainer(policy, env, args, test_env=test_env)

    trainer.set_seed(args.seed)

    
    try:
        if args.evaluate:
            print('-' * 89)
            print('Evaluating %s'%trainer._output_dir)
            trainer.evaluate_policy(10000)  # 每次测试都会在生成临时文件，要定期处理

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

    if args.clean:
        print('Cleaning output dir ... ')
        shutil.rmtree(trainer._output_dir)