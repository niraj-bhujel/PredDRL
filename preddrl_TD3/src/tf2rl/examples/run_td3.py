# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
import rospy
from gazebo_env.environment_stage_3_bk import Env
from tf2rl.algos.td3 import TD3
from tf2rl.experiments.trainer import Trainer


# Load = True
Load = False
if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = TD3.get_argument(parser)
    # parser.add_argument('--env-name', type=str, default="2dCarWorld2-v1")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=3000) # 重新训练的话要改回 10000
    parser.set_defaults(max_steps=50000)
    # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/compare_network/1conv_2dnn_3input_dropout_1')
    
    # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/20210213')

    if Load:
        parser.set_defaults(test_episodes=50)
        parser.set_defaults(episode_max_steps=int(1e4))
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/old_weights_beforerate/compare1/1compare_td3_per_nstep_500k_cnn')
        parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/compare_network/1conv_2dnn_3input_dropout_1')
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/22')
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/pioneer_real_180')
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/pioneer_real')
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/21d_60laser_1')
        #td3-sn
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/compare1/td3only2')
        #itd3-sn
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/compare1/td3per_nstep')
        #td3-pn
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/TD3_PN')
        #itd3-pn
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/compare_network/1conv_2dnn_3input')
        #itd3-pn1
        # parser.set_defaults(model_dir='/home/ros_admin/tf2rl_ws/results/td3-pn1')
        parser.set_defaults(show_test_progress=False)
        # parser.set_defaults(show_test_progress=False)
        parser.set_defaults(save_model_interval=int(1e10))
    args = parser.parse_args()

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
    
    if Load:
        trainer.evaluate_policy(10000)  # 每次测试都会在生成临时文件，要定期处理
    else:
        trainer()
