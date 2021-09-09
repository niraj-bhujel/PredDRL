import argparse

def get_argument(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(conflict_handler='resolve')

    # policy setting
    parser.add_argument('--policy', type=str, default='ddpg_graph',
                        help="Model name one of [td3, ddpg, graph_ddpg, gcn]")
    parser.add_argument('--sampling_method', type=str, default='orca',
                        help="Action sampling method. One of [uniform, prefered_vel, orca]")

    parser.add_argument('--memory_capacity', type=int, default=int(1e6))

    # experiment settings
    parser.add_argument('--max_steps', type=int, default=int(1e6),
                        help='Maximum number steps to interact with env.')
    parser.add_argument('--episode_max_steps', type=int, default=int(1e3),
                        help='Maximum steps in an episode')
    parser.add_argument('--n_experiments', type=int, default=1,
                        help='Number of experiments')
    parser.add_argument('--show_progress', action='store_true',
                        help='Call `render` in training process')
    parser.add_argument('--save_model_interval', type=int, default=int(1e6),
                        help='Interval to save model')
    parser.add_argument('--save_summary_interval', type=int, default=int(1e3),
                        help='Interval to save summary')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory to restore model')
    parser.add_argument('--dir_suffix', type=str, default='',
                        help='Suffix for directory that contains results')
    parser.add_argument('--normalize_obs', action='store_true',
                        help='Normalize observation')
    parser.add_argument('--logdir', type=str, default='preddrl_td3/results',
                        help='Output directory')
    parser.add_argument('--overwrite', action='store_false', default=True,
                        help='Overwrite existing experiments')

    # test settings
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate trained model')
    parser.add_argument('--test_interval', type=int, default=int(1e6),
                        help='Interval to evaluate trained model')
    parser.add_argument('--show_test_progress', action='store_true',
                        help='Call `render` in evaluation process')
    parser.add_argument('--test_episodes', type=int, default=5,
                        help='Number of episodes to evaluate at once')
    parser.add_argument('--save_test_path', action='store_true',
                        help='Save trajectories of evaluation')
    parser.add_argument('--show_test_images', action='store_true',
                        help='Show input images to neural networks when an episode finishes')
    parser.add_argument('--save_test_movie', action='store_true',
                        help='Save rendering results')

    # replay buffer
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='Size of buffer')
    parser.add_argument('--use_prioritized_rb', action='store_true',
                        help='Flag to use prioritized experience replay')
    parser.add_argument('--use_nstep_rb', action='store_true',
                        help='Flag to use nstep experience replay')
    parser.add_argument('--n_step', type=int, default=4,
                        help='Number of steps to look over')
    parser.add_argument('--load_memory', action='store_true', default=False,
                        help='If use previously saved memory to save sampling time')
    # others
    parser.add_argument('--logging_level', choices=['DEBUG', 'INFO', 'WARNING'],
                        default='INFO', help='Logging level')

    # graph
    parser.add_argument('--input_states', nargs='+', default=['current_states', 'future_states'],
                        help='Input states for nodes')
    parser.add_argument('--pred_states', nargs='+', default=['action'],
                        help='Prediction states of the nodes')
    parser.add_argument('--input_edges', nargs='+', default=['diff', 'dist'], 
                        help='Inter node disances, dist (l2norm) or diff (l1norm)')
    parser.add_argument('--pred_edges', nargs='+', default=['dist'], 
                        help='Inter node disances, dist (l2norm) or diff (l1norm)')
    parser.add_argument('--vis_graph', action='store_true', default=False,
                        help='Plot graph during training step. Plot in output_dir/graphs/')
    parser.add_argument('--future_steps', default=4., type=float,
                        help='Apply dropout to input features')
    # gated gcn
    parser.add_argument('--in_feat_dropout', default=0., type=float,
                        help='Apply dropout to input features')
    parser.add_argument('--dropout', default=0., type=float,
                        help='Apply dropout on hidden features')
    parser.add_argument('--batch_norm', action='store_false', default=False,
                        help='Apply batch norm between layer')
    parser.add_argument('--residual', action='store_false', default=False,
                        help='Apply batch norm between layer')
    parser.add_argument('--activation', default='ReLU',
                        help='Activation function')
    parser.add_argument('--layer', default='gated_gcn',
                        help='One of [gcn, edge_gcn, gated_gcn, custom_gcn]')

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
                        help='Debugging level for tensorflow.')
    parser.add_argument('--clean', action='store_true', 
                        help='Remove outputs when exit.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--verbose', type=int, default=0,
                        help='One of [0, 1, 2]')

    # simulation env
    parser.add_argument('--stage', type=int, default=0,
                        help='Value from 0-6')   

    return parser