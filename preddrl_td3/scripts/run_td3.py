# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import sys
sys.path.insert(0, './')
import gym
import rospy

from preddrl_gazebo_env.environment_stage_3_bk import Env
from td3 import TD3
from trainer import Trainer
print(os.getcwd())

if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = TD3.get_argument(parser)

    args = parser.parse_args()
    args.model_dir = './results/compare_network/1conv_2dnn_3input_dropout_1'

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    print(vars(args))


    # env = gym.make(args.env_name)
    # test_env = gym.make(args.env_name)
    rospy.init_node('turtlebot3_td3_stage_3')
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
    trainer = Trainer(policy, env, args, test_env=test_env)
    try:
        if args.phase=='test':
            trainer.evaluate_policy(10000)  # 每次测试都会在生成临时文件，要定期处理
        elif args.phase=='train':
            trainer()
        else:
            print('args.phase not unknown'. args.phase)
    except KeyboardInterrupt: # this is to prevent from accidental ctrl + c
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
