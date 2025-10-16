# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for DeMoMTasksuite.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default='MoDe-Spot-Curtain-v0', help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import omni.log
import task
from controller.se3_keyboard import MMKeyboard
from omni.isaac.lab_tasks.utils import parse_env_cfg




import os
from datetime import datetime
Data_buff = []
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_path = f'./{args_cli.task}/'
file_path = os.path.join(
             folder_path+f'{timestamp}.npy'
        )
os.makedirs(folder_path, exist_ok=True)
def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg,render_mode="rgb_array" if args_cli.enable_cameras else None)
    # check environment name (for reach , we don't allow the gripper)
    if "Curtain" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    #lower()converts all uppercase characters in a string to their lowercase equivalents
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = MMKeyboard(
            arm_pos_sensitivity = args_cli.sensitivity * 0.05,
            arm_rot_sensitivity = args_cli.sensitivity * 0.8,
            base_com_sensitivity = args_cli.sensitivity*2,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()

    teleop_interface.reset()

    finish_flag = False
    next_obs = None
    actions = torch.zeros_like(env.actions)
    i = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            #
            obs_dict, reward, dones, time_out,_ = env.step(actions)
            obs = obs_dict['rgb']


            arm_delta_pose, gripper_command, base_delta_com,finish_flag = teleop_interface.advance()
            arm_delta_pose = arm_delta_pose.astype("float32")
            base_delta_com = base_delta_com.astype("float32")
            # convert to torch
            arm_delta_pose = torch.tensor(arm_delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            base_delta_com = torch.tensor(base_delta_com, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

            # compute actions
            actions =  torch.concat([base_delta_com, arm_delta_pose], dim=1)

            # get keyboard command
            if actions.any()!=0: # in  ["W", "S", "A", "D", "Z", "X","UP", "DOWN", "LEFT", "RIGHT", "M", "N"]:

                data = {
                    'obs': obs.clone().cpu(),
                    'key': teleop_interface._value,
                    'action':actions
                }
                Data_buff.append(data)
                i = i +1

            if finish_flag:
                np.save(file_path, np.array(Data_buff, dtype=object))
                print(f'Save {len(Data_buff)} trag to {file_path}')
                env.close()

                num = sum(1 for file in os.listdir(folder_path) if file.endswith(".npy"))
                print(f'Already collect {num} demos')
                simulation_app.close()


    # close the simulator
    env.close()


if __name__ == "__main__":
    import numpy as np

    main()
    # close sim app
    simulation_app.close()

