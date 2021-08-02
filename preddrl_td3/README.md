# preddrl_TD3

**1.Create Virtual Environment**  
``` 
conda env create -f environment.yaml  
``` 
**2. Activate Virtual Environment and Run**  
_You must start the simulation environment before you can run the following commands. The simulation environment and DRL algorithm run in different environments_
``` 
#launch virtual environment 
conda activate tf2 
pip install -r requirements.txt --no-deps
cd preddrl_td3 # change to root dir
python scripts/run_td3.py   #run the script
``` 

**3.Notice**  
If want to re-train model, change variable use  to False in examples/run_td3.py, ***self.test*** in examples/gazebo_env/environment_stage_3.py to False

**4.Important Files**  
The interaction with Gazebo environment and data collecting is in:
```
~/preddrl_env/environment_stage_3_bk.py
```
Parameters setting and start to train in:
```
~/preddrl_td3/src/tf2rl/examples/run_td3.py
```
The trainer can be modified in:
```
~/anaconda3/envs/tf2/lib/python3.6/site-packages/tf2rl/experiments/trainer.py
```
The actor network architecture can be modified in: 
```
~/anaconda3/envs/tf2/lib/python3.6/site-packages/tf2rl/algos/ddpg.py
```


Please copy files in script folder to overwrite above two files (trainer.py and ddpg.py).


