# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

#from ..device_base import DeviceBase
from omni.isaac.lab.devices.device_base import DeviceBase

class MMKeyboard(DeviceBase):
    """A keyboard controller for sending base and arm commands as delta poses and binary command for the gripper(open/close).

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self,
                 arm_pos_sensitivity: float = 0.4,
                 arm_rot_sensitivity: float = 0.8,
                 base_com_sensitivity: float = 1):
        """Initialize the keyboard layer.

        Args:
            arm_pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            arm_rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
            base_com_sensitivity: Magnitude of scale base commands scaling. Defaults to 1.
        """
        # store inputs
        self.arm_pos_sen = arm_pos_sensitivity
        self.arm_rot_sen = arm_rot_sensitivity
        self.base_com_sen = base_com_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._close_gripper = False
        self._delta_arm_pos = np.zeros(3)  # (x, y, z)
        self._delta_arm_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._delta_base_com = np.zeros(3)  # (x, y, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        self._finish = False
        self._value = None

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): K\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V\n"

        msg += "\tMove base forward and back: up/down\n"
        msg += "\tMove base left and right: left/right\n"
        msg += "\tRotate base along z-axis: M/N"

        return msg

    """
    Operations
    """

    def reset(self):
        # command buffers
        self._gripper_act = False
        self._delta_arm_pos = np.zeros(3)  # (x, y, z)
        self._delta_arm_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._delta_base_com = np.zeros(3)  # (x, y, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the result from keyboard event state.
                self._close_gripper = False
        self._delta_arm_pos = np.zeros(3)  # (x, y, z)
        self._delta_arm_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._delta_base_com = np.zeros(3)  # (x, y, yaw)
        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_arm_rot).as_rotvec()
        arm_pose = np.concatenate([self._delta_arm_pos, rot_vec])
        gripper_act =  self._gripper_act#.copy()
        base_com = self._delta_base_com.copy()
        self._delta_arm_pos = np.zeros(3)  # (x, y, z)
        self._delta_arm_rot = np.zeros(3)  # (roll, pitch, yaw)

        #self._value = None

        # return the command and gripper state
        return arm_pose,gripper_act,base_com,self._finish

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self._finish = True
            if event.input.name == "K":
                self._gripper_act = np.logical_not(self._gripper_act)
                print(f'grasp state: {self._gripper_act}')
                self._value = event.input.name
            elif event.input.name in ["W", "S", "A", "D", "Z", "X"]:
                self._value = event.input.name
                self._delta_arm_pos += self._INPUT_KEY_MAPPING[event.input.name]
            #elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
            #    self._delta_arm_rot += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["UP", "DOWN", "LEFT", "RIGHT", "M", "N"]:
                self._delta_base_com += self._INPUT_KEY_MAPPING[event.input.name]
                self._value = event.input.name

        # remove the command when un-pressed

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["UP", "DOWN", "LEFT", "RIGHT", "M", "N"]:
                self._delta_base_com -= self._INPUT_KEY_MAPPING[event.input.name]
                self._value = event.input.name+'_n'


        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "K": True,
            # x-axis (forward) for arm
            "W": np.asarray([1.0, 0.0, 0.0]) * self.arm_pos_sen,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.arm_pos_sen,
            # y-axis (left-right) for arm
            "A": np.asarray([0.0, 1.0, 0.0]) * self.arm_pos_sen,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.arm_pos_sen,
            # z-axis (up-down) for arm
            "Z": np.asarray([0.0, 0.0, 1.0]) * self.arm_pos_sen,
            "X": np.asarray([0.0, 0.0, -1.0]) * self.arm_pos_sen,

            # forward and back command for base
            "UP": np.asarray([1.0, 0.0, 0.0]) * self.base_com_sen,
            "DOWN": np.asarray([-1.0, 0.0, 0.0]) * self.base_com_sen,
            # right and left command for base
            "LEFT": np.asarray([0.0, -1.0, 0.0]) * self.base_com_sen,
            "RIGHT": np.asarray([0.0, 1.0, 0.0]) * self.base_com_sen,
            # yaw command (around z-axis) for base
            "M": np.asarray([0.0, 0.0, 1.0]) * self.base_com_sen,
            "N": np.asarray([0.0, 0.0, -1.0]) * self.base_com_sen,
        }

        '''
        # roll (around x-axis) for arm
        "Z": np.asarray([1.0, 0.0, 0.0]) * self.arm_rot_sen,
        "X": np.asarray([-1.0, 0.0, 0.0]) * self.arm_rot_sen,
        # pitch (around y-axis) for arm
        "T": np.asarray([0.0, 1.0, 0.0]) * self.arm_rot_sen,
        "G": np.asarray([0.0, -1.0, 0.0]) * self.arm_rot_sen,
        # yaw (around z-axis) for arm
        "C": np.asarray([0.0, 0.0, 1.0]) * self.arm_rot_sen,
        "V": np.asarray([0.0, 0.0, -1.0]) * self.arm_rot_sen,
        '''

