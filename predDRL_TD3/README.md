# predDRL_TD3

**1.Dependencies**  
``` 
conda env create -f environment.yaml  
``` 
**2.Run**  
_You must start the simulation environment before you can run the following commands. The simulation environment and DRL algorithm run in different environments_
``` 
conda activate tf2  #launch virtual environment
python run_td3.py   #run the script
``` 

**3.Notice**  
If want to re-train model，change variable ***Load*** to False in examples/run_td3.py, ***self.test*** in examples/gazebo_env/environment_stage_3.py to False
