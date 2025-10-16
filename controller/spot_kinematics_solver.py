# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from typing import Optional, Tuple
import os
import carb
import numpy as np
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.motion_generation")
from omni.isaac.motion_generation.kinematics_interface import KinematicsSolver
from omni.isaac.motion_generation import LulaKinematicsSolver

class ArticulationKinematicsSolver:
    """Wrapper class for computing robot kinematics in a way that is easily transferable to the simulated robot Articulation.  A KinematicsSolver
    computes FK and IK at any frame, possibly only using a subset of joints available on the simulated robot.
    This wrapper simplifies computing the current position of the simulated robot's end effector, as well as wrapping an IK result

    Args:
        robot_articulation (Articulation): Initialized robot Articulation object representing the simulated USD robot
        kinematics_solver (KinematicsSolver): An instance of a class that implements the KinematicsSolver
        end_effector_frame_name (str): The name of the robot's end effector frame.  This frame must appear in kinematics_solver.get_all_frame_names()
    """

    def __init__(
        self, #end_effector_frame_name: str
    ):
        self._kinematics_solver = []
        self._ee_frame = []
        return



    def intial_multi_solver(self,ee_frame_name):
        for ee_name in ee_frame_name:
            base_path = os.getcwd()
            kinematics_solver = LulaKinematicsSolver(
                robot_description_path=base_path + '/asset/spot.yaml',
                urdf_path=base_path + '/asset/spot.urdf'
            )
            if ee_name not in kinematics_solver.get_all_frame_names():
                carb.log_error(
                    "Frame name"
                    + ee_name
                    + " not recognized by KinematicsSolver.  Use KinematicsSolver.get_all_frame_names() to get a list of valid frames"
                )
            else:
                self._ee_frame.append(ee_name)
                self._kinematics_solver.append(kinematics_solver)

    def compute_end_effector_pose(self, joint_positions=None,position_only=False) -> Tuple[np.array, np.array]:
        """Compute the pose of the robot end effector using the simulated robot's current joint positions

        Args:
            position_only (bool): If True, only the frame positions need to be calculated.  The returned rotation may be left undefined.

        Returns:
            Tuple[np.array,np.array]:
            position: Translation vector describing the translation of the robot end effector relative to the USD global frame (in stage units)

            rotation: (3x3) rotation matrix describing the rotation of the frame relative to the USD stage global frame
        """
        if joint_positions is None:
            carb.log_error(
                "Attempted to compute forward kinematics for an uninitialized robot Articulation. Cannot get joint positions"
            )

        return self._kinematics_solver.compute_forward_kinematics(
            self._ee_frame, joint_positions, position_only=position_only
        )

    def compute_inverse_kinematics(
        self,
        warm_start,
        target_position: np.array,
        target_orientation: Optional[np.array] = None,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None,
    ) -> Tuple[np.array, bool]:
        """
        Compute inverse kinematics for the end effector frame using the current robot position as a warm start.  The result is returned
        in an articulation action that can be directly applied to the robot.

        Args:
            target_position (np.array): target translation of the target frame (in stage units) relative to the USD stage origin
            target_orientation (np.array): target orientation of the target frame relative to the USD stage global frame. Defaults to None.
            position_tolerance (float): l-2 norm of acceptable position error (in stage units) between the target and achieved translations. Defaults to None.
            orientation tolerance (float): magnitude of rotation (in radians) separating the target orientation from the achieved orienatation.
            orientation_tolerance is well defined for values between 0 and pi. Defaults to None.

        Returns:
            Tuple[np.array, bool]:
            ik_result: An np.array that can be applied to the robot to move the end effector frame to the desired position.

            success: Solver converged successfully
        """
        # for multi robot, need to adapt to luna.pos3,therefore we adapt here, calculate one once a time
        results,succ_flg = [],[]
        for i,kinematic_sover in enumerate(self._kinematics_solver):
            ik_result, succ = kinematic_sover.compute_inverse_kinematics(
                self._ee_frame[i], target_position[i], target_orientation[i], warm_start[i], position_tolerance, orientation_tolerance
            )
            if succ:
                results.append(ik_result)
            else:
                print('IK fail')
                results.append(warm_start[i])
            succ_flg.append(succ)

        return np.array(results), succ_flg #self._joints_view.make_articulation_action(ik_result, None), succ



    def get_end_effector_frame(self) -> str:
        """Get the end effector frame

        Returns:
            str: Name of the end effector frame
        """
        return self._ee_frame

    def get_kinematics_solver(self) -> KinematicsSolver:
        """Get the underlying KinematicsSolver instance used by this class.

        Returns:
            KinematicsSolver: A class that can solve forward and inverse kinematics for a specified robot.
        """
        return self._kinematics_solver

    def set_robot_base_pose(self,pos,quat):
        """set robot base pose. if miss the behavior would be strange """
        for i,kinematic_sover in enumerate(self._kinematics_solver):
            kinematic_sover.set_robot_base_pose(pos[i],quat[i])
