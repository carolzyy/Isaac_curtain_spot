# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
#from operator import index
#from typing import Optional, Tuple
#import os
#import carb
#import numpy as np
import omni.isaac.lab.utils.string as string_utils
import torch
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.motion_generation")
#from omni.isaac.motion_generation import LulaKinematicsSolver
from controller.spot_kinematics_solver import ArticulationKinematicsSolver
from controller.spot_loco_solver import LocomotionController
class OperationSpaceController:
    """ Operation-space controller.
    """
    def __init__(
            self,
            num_robot,
            device,
            #end_effector_frame_name
    ):
        """Initialize operation-space controller.

        Args:
            num_envs: The number of robots to control.
            end_effector_frame_name: The name of end effector
            device: The device to use for computations.

        Raises:
            ValueError: When invalid control command is provided.
        """

        self.arm_ctrl = ArticulationKinematicsSolver()
        self.base_ctrl = LocomotionController(num_envs=num_robot,device=device)
        self.num_robot = num_robot
        self.device = device
        return

    def init_ctrl(self,ee_names,body_names,joint_names):
        if isinstance(ee_names, str):
            ee_names = [ee_names] * self.num_robot
        self.arm_ctrl.intial_multi_solver(ee_names)
        self.ee_idx = string_utils.resolve_matching_names("arm_ee", body_names, )[0][0]#self.robot.find_bodies("arm_ee")[0][0] #arm_fngr
        self.base_idx = string_utils.resolve_matching_names("body", body_names, )[0][0] #self.robot.find_bodies('body')[0][0]  # arm_sh0
        base_body_name = ['fl_hx', 'fr_hx', 'hl_hx', 'hr_hx', 'fl_hy', 'fr_hy', 'hl_hy', 'hr_hy', 'fl_kn', 'fr_kn',
                     'hl_kn', 'hr_kn']
        self.base_idxs = string_utils.resolve_matching_names(base_body_name, joint_names, True)[0] #[self.robot.find_joints(name)[0][0] for name in body_name]
        self.arm_idxs = string_utils.resolve_matching_names('arm0_.*', joint_names, )[0]#self.robot.find_joints('arm0_.*')[0]




    def compute(self,
                root_lin_vel_b,
                root_ang_vel_b,
                gravity_b,
                current_joint_pos,
                current_joint_vel,
                body_state_w,
                base_command,
                arm_command = None):

        base_joint_act = None
        arm_joint_act = None
        if arm_command is None:
            current_base_joints_pos = current_joint_pos[:,self.base_idxs]

            current_base_joints_vel = current_joint_vel[:, self.base_idxs]
            base_joint_act = self.base_ctrl.compute_action(root_lin_vel_b, root_ang_vel_b,  gravity_b,
                                                             current_base_joints_pos, current_base_joints_vel,
                                                             base_command)
            joint_action = base_joint_act
            joint_index = self.base_idxs
            success = [True] * self.num_robot
        else:
            current_arm_joints_pos = current_joint_pos[:, self.arm_idxs]
            ee_pose_w = body_state_w[:, self.ee_idx, 0:7].cpu().numpy()
            robot_base_pose = body_state_w[:, self.base_idx, 0:7].cpu().numpy()
            self.arm_ctrl.set_robot_base_pose(robot_base_pose[:,:3], robot_base_pose[:,3:])
            current_arm_joints_pos = current_arm_joints_pos.cpu().numpy()

            arm_pose_command = arm_command[:,:3].cpu().numpy()
            arm_rot_command = arm_command[:,3:6].cpu().numpy()
            # not add rot command
            arm_rot_command = [arm_rot_command[i]  if arm_rot_command[i].any() else None for i in range(self.num_robot)]
            #print(f'ee position is {ee_pose_w[:,:3]},command is {arm_pose_command}')
            arm_joint_act, success = self.arm_ctrl.compute_inverse_kinematics(
                                                                        current_arm_joints_pos,
                                                                        ee_pose_w[:,:3] + arm_pose_command, #,y,x
                                                                        ee_pose_w[:,3:],
            )
            '''
            if success:
                print(f'success')
            '''
            joint_action = torch.from_numpy(arm_joint_act).to(self.device, torch.float32)
            joint_index = self.arm_idxs


        return joint_action,joint_index,success

