
1. Create a preddrl_ws/src dir, change to src and clone the branch and build

2. source devel/setup.bash 

3. Change to preddrl_gazebo_plugin director
$ roslaunch preddrl_gazebo social_contexts.launch

4. Change to preddrl_ws/src and run the python script to train the model
$ python3 preddrl_td3/scripts_torch/trainer.py --stage 7 --policy ddpg_graph --verbose 1 --n_warmup 2000 --sampling_method orca


