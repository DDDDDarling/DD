
import numpy as np
import gym
import torch
from gym import spaces
import json
import sys
import UdpComms as U
import warnings
warnings.filterwarnings("ignore")
sys.path.append("..")
sock = U.UdpComms(udpIP="127.0.0.1", portTX=8003, portRX=8004, enableRX=True, suppressWarnings=True)
class UnityMissileControlEnv(gym.Env):

    def __init__(self, testing=False, interactive=False, a_p=0.15, a_r=0.1,
                 e_thres=0.6):
        self.a_r = a_r
        self.a_p = a_p
        self.e_thres = e_thres

        self.obs_vector = None

        # Create environment state
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]),
                                       dtype=np.float32)
        self.observation_space = spaces.Discrete(27)

        # Statistics for testing
        self.epoch_counter = -1
        self.iteration_counter = 0
        self.end_iteration = 0
        self.interactive = interactive
        self.score_thr = 0.3
        self.last_target_distance = None
        self.target_distance = None

    def reset(self):
        # Reset statistics
        sock.SendData("RESET")
        # Update statistics
        self.epoch_counter += 1
        self.iteration_counter = 0
        self.end_iteration = 0
        reward = 0
        state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        return state

    def seed(self, seed):
        np.random.seed(seed)


    def step(self, action):
        # Perform an action
        action = action.astype(float)
        # if action[0] > 0 and action[0] < 0.2:
        #     action[0] = 0.2
        # if action[0] < 0 and action[0] > -0.2:
        #     action[0] = -0.2
        # if action[1] > 0 and action[1] < 0.2:
        #     action[1] = 0.2
        # if action[1] < 0 and action[1] > -0.2:
        #     action[1] = -0.2
        # if action[2] > 0 and action[2] < 0.2:
        #     action[2] = 0.2
        # if action[2] < 0 and action[2] > -0.2:
        #     action[2] = -0.2
        action_dict = {"x": action[0], "y": action[1], "z": action[2]}
        print("Choose action:{}".format(action_dict))
        action_json = json.dumps(action_dict)
        sock.SendData(action_json)
        while 1:
            callback = sock.ReadReceivedData()
            if callback:
                # 尝试解析JSON数据
                state_data = json.loads(callback)
                state = [
                    state_data["missile_position_x"]/1000,
                    state_data["missile_position_y"]/1000,
                    state_data["missile_position_z"]/1000,
                    state_data["missile_ratotion_x"]/360,
                    state_data["missile_ratotion_y"]/360,
                    state_data["missile_ratotion_z"]/360,
                    state_data["target_postion_x"]/1000,
                    state_data["target_postion_y"]/1000,
                    state_data["target_postion_z"]/1000,
                    state_data["threatZone1_postion_x"]/1000,
                    state_data["threatZone1_postion_y"]/1000,
                    state_data["threatZone1_postion_z"]/1000,
                    state_data["threatZone1_r"]/100,
                    state_data["threatZone2_postion_x"]/1000,
                    state_data["threatZone2_postion_y"]/1000,
                    state_data["threatZone2_postion_z"]/1000,
                    state_data["threatZone2_r"]/100,
                    state_data["threatZone3_postion_x"]/1000,
                    state_data["threatZone3_postion_y"]/1000,
                    state_data["threatZone3_postion_z"]/1000,
                    state_data["threatZone3_r"]/100,
                    state_data["threatZone4_postion_x"]/1000,
                    state_data["threatZone4_postion_y"]/1000,
                    state_data["threatZone4_postion_z"]/1000,
                    state_data["threatZone4_r"]/100,
                    state_data["angle"]/np.pi,
                    # state_data["threatZone5_postion_x"],
                    # state_data["threatZone5_postion_y"],
                    # state_data["threatZone5_postion_z"],
                    # state_data["threatZone5_r"],
                    state_data["state"]
                ]
                self.target_distance = state_data["target_distance"]
                break
            # else:
            #     print("数据为空")
        done = False
        reward = 0
        # if self.last_target_distance is not None and self.target_distance is not None:
        #     if self.target_distance < self.last_target_distance:
        #         reward += 1 
        #     else:
        #         reward -= 1 
        # self.last_target_distance = self.target_distance
        # print("distance:{}, angle:{}".format(self.target_distance, state_data["angle"]))
        reward -= 0.0015 * self.target_distance
        reward -= 0.05
        reward -= 0.05 * state_data["angle"] / np.pi
        if state_data["state"] == 3 or state_data["state"] == 4:
            reward -= 5
            done = True
        elif state_data["state"] == 5:
            reward += 5
            done = True

        self.iteration_counter += 1
        if self.iteration_counter >= 20:  # end_iteration = 8
            reward -= 5
            done = True
        # print("state:{},reward:{}".format(state_data["state"], reward))
        return state, reward, done, {}

    def render(self, mode='Car', close=False):

        pass


    def __str__(self):
        return 'UnityMissileControlEnv'

# if __name__=='__main__':
#
#     keyboard = {'w': 119, 's': 115, 'a': 97, 'd': 100, 'q': 113, 'e': 101}
#
#     np.random.seed(1)
#     env = UnityCameraControlEnv(a_p=0, a_r=0, e_thres=0, interactive=True)
#     env.seed(1)
#     env.reset()
#     action = 0
#     for i in range(1000):
#         key = env.interactive_keyboard()
#         # print(key)
#         if key == keyboard['w']:  # 'Up'
#             action =
#         elif key == keyboard['s']:  # 'Down'
#             action =
#         elif key == keyboard['a']:  # 'Left'
#             action =
#         elif key == keyboard['d']:  # 'Right'
#             action =
#         elif key == keyboard['q']:  # 'Zoom in'
#             action =
#         elif key == keyboard['e']:  # 'Zoom out'
#             action =
#         else:
#             action =
#         env.step(action)
#     env.close()
