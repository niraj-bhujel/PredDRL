import rospy
import sys

from gazebo_msgs.msg import ModelStates

if './' not in sys.path:
    sys.path.insert(0, './')
print(sys.path)

# if './preddrl_tracker/scripts' not in sys.path:
#     sys.path.insert(0, '/preddrl_tracker/scripts'

from preddrl_tracker.scripts.pedestrian_state_publisher import prepare_data
from preddrl_td3.scripts_torch.env.respawnPeds import RespawnPedestrians

pedestrians, ped_frames, _, = prepare_data('./preddrl_tracker/data/crowds_zara01.txt', 2.5, 10)
ped_spawnner = RespawnPedestrians()

# use only firt 50 frames
ped_frames = ped_frames[:50]

rospy.init_node('ped_spawnner', disable_signals=True)

try:
    for global_step in range(20000):

        if global_step>len(ped_frames):
            t = global_step%len(ped_frames)
        else:
            t = global_step

        curr_peds = [ped for ped in pedestrians if t>=ped.first_timestep and t<=ped.last_timestep]

        curr_ped_states = {ped.id:ped.get_states_at(t) for ped in curr_peds}
        # print("Waiting for model_states .. ")
        model_states = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=100)

        ped_spawnner.respawn(curr_ped_states, model_states, verbose=1, t=t)
        
        rospy.sleep(0.5)

except KeyboardInterrupt:
    print("Clearing models .. ")
    for model_name in ped_spawnner.spawnned_models:
        if model_name in model_states.name:
            ped_spawnner.delete_model(model_name)