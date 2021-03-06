# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import sys
sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print(os.getcwd())

import gym
import rospy
from policy.td3_torch import TD3
from policy.ddpg_torch import DDPG
from trainer_torch import Trainer

from gym.utils import seeding as _s 
from preddrl_env.environment_stage_3_bk import Env

if __name__ == '__main__':

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



    rospy.init_node('turtlebot3_td3_stage_3', disable_signals=True)

    env = Env()
    test_env = Env()

    args.seed = _s._int_list_from_bigint(_s.hash_seed(_s.create_seed()))[0]

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

    except Exception as e:
        print(e)
        # continue

    except KeyboardInterrupt: # this is to prevent from accidental ctrl + c
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        sys.exit()