# preddrl_TD3

**1.Dependencies**  
``` 
conda env create -f environment.yaml  
``` 
**2.Run**  
_You must start the simulation environment before you can run the following commands. The simulation environment and DRL algorithm run in different environments_
``` 
conda activate tf2  #launch virtual environment
python ~/predDRL_TD3/src/tf2rl/examples/run_td3.py   #run the script
``` 

**3.Notice**  
If want to re-train model, change variable ***Load*** to False in examples/run_td3.py, ***self.test*** in examples/gazebo_env/environment_stage_3.py to False

**4.Important Files**  
The interaction with Gazebo environment and data collecting is in:
```
~/preddrl_TD3/src/tf2rl/examples/Gazebo_env/environment_stage_3_bk.py
```
The trainer can be modified in:
```
~/anaconda3/envs/tf2/lib/python3.6/site-packages/tf2rl/experiments/trainer.py
```
The actor network architecture can be modified in: 
```
~/anaconda3/envs/tf2/lib/python3.6/site-packages/tf2rl/algos/ddpg.py
```
Parameters setting and start to train in:
```
~/predDRL_TD3/src/tf2rl/examples/run_td3.py
```


