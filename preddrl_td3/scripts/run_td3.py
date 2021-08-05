# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import sys
sys.path.insert(0, './')
import gym
import rospy

from preddrl_env.environment_stage_3_bk import Env

from td3 import TD3
from trainer import Trainer
print(os.getcwd())

# from tf2rl.algos.policy_base import OffPolicyAgent

if __name__ == '__main__':

    parser = Trainer.get_argument()

    parser = TD3.get_argument(parser)

    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=3000) # 重新训练的话要改回 10000
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(restore_checkpoint=False)
    
    args = parser.parse_args()
    print(vars(args))

    print(args.evaluate)
    # test param, modified by niraj
    if args.evaluate:
        args.test_episodes=50
        args.episode_max_steps = int(1e4)
        args.model_dir = './preddrl_td3/results/compare_network/1conv_2dnn_3input_dropout_1'
        args.show_test_progress=False
        args.save_model_interval = int(1e10)
        args.restore_checkpoint = True

        # parser.set_defaults(test_episodes=50)
        # parser.set_defaults(episode_max_steps=int(1e4))
        # parser.set_defaults(model_dir='./preddrl_td3/results/compare_network/1conv_2dnn_3input_dropout_1')
        # parser.set_defaults(show_test_progress=False)
        # parser.set_defaults(save_model_interval=int(1e10))



    rospy.init_node('turtlebot3_td3_stage_3', disable_signals=True)

    env = Env()
    test_env = Env()

    policy = TD3(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=0,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=[400, 300],
        n_warmup=args.n_warmup)

    # print('offpolicy:', issubclass(type(policy), OffPolicyAgent))

    trainer = Trainer(policy, env, args, test_env=test_env)

    try:
        if args.evaluate:
            print('Evaluating policy ...')
            trainer.evaluate_policy(10000)  # 每次测试都会在生成临时文件，要定期处理

        else:
            print('Training policy ...')
            trainer()

    except KeyboardInterrupt: # this is to prevent from accidental ctrl + c
        sys.exit()
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
