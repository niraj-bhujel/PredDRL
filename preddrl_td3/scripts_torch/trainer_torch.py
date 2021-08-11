import os
import time
import logging
import argparse
import shutil
import random
import torch
import numpy as np
import tensorflow as tf

from gym.spaces import Box

from misc.prepare_output_dir import prepare_output_dir
from misc.initialize_logger import initialize_logger
from misc.get_replay_buffer import get_replay_buffer
from utils.normalizer import EmpiricalNormalizer
from utils.utils import save_path, frames_to_gif, save_ckpt, load_ckpt, copy_src

# from gym_replay.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from deeprl_replay.ReplayMemory import ReplayBuffer, PrioritizedReplayBuffer

if tf.config.experimental.list_physical_devices('GPU'):
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        print(cur_device)
        tf.config.experimental.set_memory_growth(cur_device, enable=True)


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

        if self._normalize_obs:
            assert isinstance(env.observation_space, Box)
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)

        # prepare log directory
        suffix = '_'.join(['%s'%self._policy.policy_name,
                        # 'warmup_%d'%self._policy.n_warmup,
                        # 'n_step_%d'%self._n_step,
                        # 'max_steps_%d'%self._max_steps,
                        # 'episode_max_steps_%d'%self._episode_max_steps,
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
                                              time_format='%Y_%m_%d_%H-%M-%S',
                                              suffix=suffix
                                              )
        # backup scripts
        copy_src('./preddrl_td3/scripts_torch', self._output_dir + '/scripts')
        self.logger = initialize_logger(logging_level=logging.getLevelName(args.logging_level), 
                                        output_dir=self._output_dir)

        if self._evaluate or self._restore_checkpoint:
            load_ckpt(self._policy, self._output_dir, last_step=1e4)


        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self._output_dir)
        self.writer.set_as_default()

        # self.set_seed(args.seed)

    def set_seed(self, seed):
        #setup seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 1
        #for success rate
        episode_success = 0
        
        # replay_buffer = get_replay_buffer(self._policy, 
        #                                   self._env, 
        #                                   self._use_prioritized_rb, 
        #                                   self._use_nstep_rb, 
        #                                   self._n_step)

        # if self._use_prioritized_rb:
        #     replay_buffer = PrioritizedReplayBuffer(size=self._n_step, 
        #                                             state_dim=self._env.observation_space.shape[0],
        #                                             act_dim=self._env.action_space.shape[0])

        # else:
        #     replay_buffer = ReplayBuffer(size=self._n_step, 
        #                                 state_dim=self._env.observation_space.shape[0],
        #                                 act_dim=self._env.action_space.shape[0])

        if self._use_prioritized_rb:
            replay_buffer = PrioritizedReplayBuffer(size=self._buffer_size,
                                                    beta_frames=self._max_steps)
        else:
            replay_buffer = ReplayBuffer(size=self._buffer_size)

        # separate input (laser scan, vel, polor)
        obs = self._env.reset(initGoal=True) # add initGoal arg by niraj

        while total_steps < self._max_steps:
            # print('Step - {}/{}'.format(total_steps, self._max_steps))

            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                action = self._policy.get_action(obs)

            next_obs, reward, done, success, _ = self._env.step(action)

            if self._show_progress:
                self._env.render()

            episode_steps += 1
            episode_return += reward
            total_steps += 1

            tf.summary.experimental.set_step(total_steps)

            # replay_buffer.add(obs=obs, 
            #                   act=action, 
            #                   next_obs=next_obs, 
            #                   rew=reward, 
            #                   done=done)

            replay_buffer.add([obs, action, reward, next_obs, done])
            obs = next_obs
            #for success rate
            if done or episode_steps == self._episode_max_steps or success:

                if success and total_steps > self._policy.n_warmup: 
                    episode_success += 1

                if total_steps > self._policy.n_warmup:    
                    n_episode += 1

                fps = episode_steps / (time.perf_counter() - episode_start_time)

                self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                    n_episode, total_steps, episode_steps, episode_return, fps))

                tf.summary.scalar(name="Common/training_return", data=episode_return)

                success_rate = episode_success/n_episode
                tf.summary.scalar(name="Common/success rate", data=success_rate)

                if done or episode_steps == self._episode_max_steps:
                    obs = self._env.reset()

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()   

            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._policy.update_interval == 0:
                sampled_data, idxes, weights = replay_buffer.sample(self._policy.batch_size)
                obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*sampled_data)
                samples = {"obs": np.asarray(obs_batch),
                            "act": np.asarray(act_batch),
                            "next_obs":np.asarray(next_obs_batch),
                            "rew": np.expand_dims(rew_batch, 1),
                            "done":np.expand_dims(done_batch, 1),
                            "weights": np.asarray(weights),
                            "indexes": np.asarray(idxes),
                }
                # print({k:v.shape for k,v in samples.items()})

                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    actor_loss, critic_loss, td_errors = self._policy.train(samples["obs"], 
                                                                           samples["act"], 
                                                                           samples["next_obs"],
                                                                           samples["rew"], 
                                                                           samples["done"],
                                                                           samples["weights"] if self._use_prioritized_rb \
                                                                           else np.ones(self._policy.batch_size, dtype=np.float32))

                    tf.summary.scalar(name=self._policy.policy_name+"/actor_loss",
                                      data=actor_loss)
                    
                    tf.summary.scalar(name=self._policy.policy_name+"/critic_loss",
                                      data=critic_loss)

                if self._use_prioritized_rb:
                    # td_error = np.ravel(td_errors) # use previous td_error ->niraj

                    td_error = self._policy.compute_td_error(samples["obs"], 
                                                             samples["act"], 
                                                             samples["next_obs"],
                                                             samples["rew"], 
                                                             samples["done"])

                    replay_buffer.update_priorities(samples["indexes"], np.abs(td_error))



            if total_steps % self._test_interval == 0:
                
                avg_test_return = self.evaluate_policy(total_steps)

                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))

                tf.summary.scalar(name="Common/average_test_return", 
                                  data=avg_test_return)

                tf.summary.scalar(name="Common/fps", 
                                  data=fps)

                self.writer.flush()

            if total_steps % self._save_model_interval == 0:
                # self.checkpoint_manager.save()
                save_ckpt(self._policy, self._output_dir, total_steps)

        tf.summary.flush()

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

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(1e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        parser.add_argument('--normalize-obs', action='store_true',
                            help='Normalize observation')
        parser.add_argument('--logdir', type=str, default='preddrl_td3/results',
                            help='Output directory')

        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(1e6),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results')

        # replay buffer
        parser.add_argument('--buffer_size', type=int, default=100000,
                            help='Size of buffer')
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')

        # added by niraj
        parser.add_argument('--batch_size', default=100, type=int,
                            help='batch size')
        parser.add_argument('--n_warmup', default=3000, type=int, 
                            help='Number of warmup steps') # 重新训练的话要改回 10000
        parser.add_argument('--restore_checkpoint', action='store_true',
                            help='If begin from pretrained model')
        parser.add_argument('--last_step', default=1e4, type=int, 
                            help='Last step to restore.')
        parser.add_argument('--prefix', type=str, default=None,
                            help='Add prefix to log dir')
        parser.add_argument('--seed', default=1901858486, type=int, 
                            help='Seed value.')
        parser.add_argument('--debug', default=0, type=int, 
                            help='Seed value.')
        parser.add_argument('--clean', action='store_true', 
                            help='Remove outputs when exit.')
        return parser

if __name__ == '__main__':
    import sys
    sys.path.insert(0, './')

    import rospy
    from policy.td3_torch import TD3
    from policy.ddpg_torch import DDPG

    from preddrl_td3.env.environment_stage_3_bk import Env

    # from gym.utils import seeding as _s 
    
    parser = Trainer.get_argument()

    parser = DDPG.get_argument(parser)

    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=3000) # 重新训练的话要改回 10000
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(restore_checkpoint=False)
    parser.set_defaults(prefix='torch')
    parser.set_defaults(use_prioritized_rb=True)
    parser.set_defaults(use_nstep_rb=True)

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

    policy = TD3(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=0,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=[400, 300],
        n_warmup=args.n_warmup)

    policy = policy.to(policy.device)

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
        sys.exit()