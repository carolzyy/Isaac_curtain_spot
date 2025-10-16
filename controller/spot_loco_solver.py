# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import io
import omni
#import omni.kit.commands
import torch
import os

class LocomotionController:
    """The Spot quadruped"""

    def __init__(
        self,
        num_envs=1,
        load_policy=True,
        device='cpu',

    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            usd_path {str} -- robot usd filepath in the directory
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot

        """
        self._policy = None
        self.device = device
        self.num_env = num_envs
        if load_policy:
            self.load_policy()


        self._base_vel_lin_scale = 1
        self._base_vel_ang_scale = 1
        self._action_scale = 0.2

        #['arm0_sh0', 'fl_hx', 'fr_hx', 'hl_hx','hr_hx',
        # 'arm0_sh1', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy',
        # 'arm0_el0', 'fl_kn', 'fr_kn', 'hl_kn', 'hr_kn',
        # 'arm0_el1', 'arm0_wr0', 'arm0_wr1', 'arm0_f1x']
        self._default_joint_pos = torch.tensor([0.1, -0.1, 0.1, -0.1,
                                            0.9, 0.9, 1.1, 1.1,
                                            -1.5, -1.5, -1.5, -1.5,
                                            ],device=self.device)

        self._previous_action = torch.zeros(self.num_env,12,device=self.device)
        self._policy_counter = 0
        self._decimation = 10 #10

    def _compute_observation(self,lin_vel_b,ang_vel_b, gravity_b,
                             current_joint_pos,current_joint_vel,
                             command):
        """
        Compute the observation vector for the policy

        Argument:
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        obs = torch.cat(
            (
                self._base_vel_lin_scale * lin_vel_b,
                self._base_vel_ang_scale * ang_vel_b,
                gravity_b,
                self._base_vel_lin_scale * command[:, 0].unsqueeze(1),
                self._base_vel_lin_scale * command[:, 1].unsqueeze(1),
                self._base_vel_ang_scale * command[:, 2].unsqueeze(1),
                current_joint_pos - self._default_joint_pos,
                current_joint_vel,
                self._previous_action

            ),dim = -1
        )

        return obs

    def compute_action(self, linear_velocity,angular_velocity, gravity_b,
                       current_joint_pos,current_joint_vel,
                       command): #dt,
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt {float} -- Timestep update in the world.
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        """

        obs = self._compute_observation(linear_velocity,angular_velocity, gravity_b,
                                        current_joint_pos, current_joint_vel,
                                        command)
        with torch.no_grad():
            obs = obs.float()
            self.action = self._policy(obs).detach()
        self._previous_action = self.action.clone()

        joint_positions = self._default_joint_pos + (self.action * self._action_scale)
        self._policy_counter += 1

        return joint_positions


    def load_policy(self):
        # Policy
        base_path = os.getcwd()
        policy_path = base_path + '/result/model/spot_policy.pt'
        file_content = omni.client.read_file(
            policy_path
            # assets_root_path + "/Isaac/Samples/Quadruped/Spot_Policies/spot_policy.pt"
        )[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        self._policy = torch.jit.load(file,map_location=self.device)
        print(f'Loaded policy from {policy_path}')