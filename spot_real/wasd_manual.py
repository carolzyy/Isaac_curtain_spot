# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""WASD driving of robot."""
import curses
import io
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import OrderedDict
import traceback
from PIL import Image, ImageEnhance

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.api.power_pb2 as PowerServiceProto
from bosdyn.api import arm_command_pb2,robot_command_pb2

import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.api.spot.robot_command_pb2 as spot_command_pb2
import bosdyn.client.util
from bosdyn.api import geometry_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncGRPCTask, AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.util import duration_str, format_metric, secs_to_hms

from bosdyn.client.frame_helpers import (GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME,HAND_FRAME_NAME,
                                         get_a_tform_b, get_vision_tform_body)
#############
from NatNetClient import NatNetClient
import numpy as np
from agent.eval import load_agent
#########
import pyrealsense2 as rs
import cv2,pickle
#############
from Utils.sim2realcfg import joint_order_sim,arm_low_limit,arm_up_limit

Action_map={
    'w':[2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],#up
    's':[-2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],#down
    'a':[0., -2., 0., 0., 0., 0., 0., 0., 0.],#left
    'd':[0., 2., 0., 0., 0., 0., 0., 0., 0.],#right
    'q':[ 0.,  0., 2.,  0.,  0.,  0.,  0.,  0.,  0.],#M
    'e':[ 0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.],#N
    'u':[0.0000, 0.0000, 0.0000, 0.0500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],#w
    'j':[ 0.0000,  0.0000,  0.0000, -0.0500,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],#s
    'h':[0.0000, 0.0000, 0.0000, 0.0000, 0.0500, 0.0000, 0.0000, 0.0000, 0.0000],#a
    'k':[ 0.0000,  0.0000,  0.0000,  0.0000, -0.0500,  0.0000,  0.0000,  0.0000,
          0.0000],#d
    'n':[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0500, 0.0000, 0.0000, 0.0000],#z
    'm':[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0500,  0.0000,  0.0000,
          0.0000],#x
}

LOGGER = logging.getLogger()

VELOCITY_BASE_SPEED = 0.5  # m/s
VELOCITY_BASE_ANGULAR = 0.8  # rad/sec
VELOCITY_CMD_DURATION = 0.6  # seconds
ARM_CMD_DURATION = 0.6  # seconds
COMMAND_INPUT_RATE = 0.1

# Configuration for opti
SPOT_ID = 48  #with z up
TAR_ID = 50
NATNET_SERVER = "10.0.0.229"   # Motive computer IP
LOCAL_IP = "10.0.0.224"         # Local machine IP
SERVER_PORT = 9000
UPDATE_RATE = 30  # Hz


# Logger 1: For general information
logger_general = logging.getLogger("info_logger")
logger_general.setLevel(logging.INFO)

handler_general = logging.FileHandler("info.log")
formatter_general = logging.Formatter('%(asctime)s - GENERAL - %(message)s')
handler_general.setFormatter(formatter_general)

logger_general.addHandler(handler_general)
from wasd import RealSenseReader

def _grpc_or_log(desc, thunk):
    try:
        return thunk()
    except (ResponseError, RpcError) as err:
        LOGGER.error('Failed %s: %s', desc, err)

class ExitCheck(object):
    """A class to help exiting a loop, also capturing SIGTERM to exit the loop."""

    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _sigterm_handler(self, _signum, _frame):
        self._kill_now = True

    def request_exit(self):
        """Manually trigger an exit (rather than sigterm/sigint)."""
        self._kill_now = True

    @property
    def kill_now(self):
        """Return the status of the exit checker indicating if it should exit."""
        return self._kill_now


class CursesHandler(logging.Handler):
    """logging handler which puts messages into the curses interface"""

    def __init__(self, wasd_interface):
        super(CursesHandler, self).__init__()
        self._wasd_interface = wasd_interface

    def emit(self, record):
        msg = record.getMessage()
        msg = msg.replace('\n', ' ').replace('\r', '')
        self._wasd_interface.add_message(f'{record.levelname:s} {msg:s}')

class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client,
                                              LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_robot_state_async()

class OptiTrackReader:
    def __init__(self,client):
        self.state = {}
        self.lock = threading.Lock()
        self.client = client#
        self.client.rigid_body_listener = self._read_rigid_body
        self.client.labeled_marker_listener = self._read_labeled_marker
        self.client.print_level = 0

    def _read_rigid_body(self, id, pos, rot):
        global SPOT_ID
        with self.lock:
            if id == SPOT_ID:
                x,y,z,w = rot
                q_rot = np.array([w,x,y,z])# align the way in isaac
                self.state['spot'] = {"id": id, "pos": pos, "rot": q_rot}

    def _read_labeled_marker(self, labeled_marker_list):
        with self.lock:
            postion_list=[]
            unlabel_list =[]
            for i, marker in enumerate(labeled_marker_list):
                postion_list.append(marker.pos)
                if marker.model_id ==0:
                    unlabel_list.append(marker.pos)
            self.state['all_marker'] = postion_list
            self.state['unlabel_marker']= unlabel_list

    def start(self):
        threading.Thread(target=self.client.run('d'), args=(...), daemon=True).start()

    def get_latest(self):
        with self.lock:
            return self.state

    def get_state(self,rod_length=2.5, trans=None):
        state = self.get_latest()
        while len(state)<1:
            continue
        robot_pos = np.array(state['spot']['pos'])
        robot_rot = np.array(state['spot']['rot'])
        all_marker = state['unlabel_marker']

        filtered_markers = []
        for marker_pos in all_marker:
            marker_pos = np.array(marker_pos)

            distance = np.linalg.norm(marker_pos - robot_pos)
            if (distance<rod_length):
                filtered_markers.append(marker_pos)
        elem_pos = np.array(filtered_markers)

        if trans is not None:
            robot_pos = robot_pos-trans
            elem_pos = elem_pos-trans
            f_points = elem_pos[elem_pos[:,0]>0]


        # Choose 4 equally spaced indices
        num_samples = 4
        indices = np.linspace(0, len(f_points) - 1, num=num_samples, dtype=int)

        # Select those points
        selected_points = f_points[indices]



        
        return robot_pos,robot_rot,all_marker,selected_points


class WasdInterface(object):
    """A curses interface for driving the robot."""

    def __init__(self, robot):
        self._robot = robot
        # Create clients -- do not use the for communication yet.
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        try:
            self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
            self._estop_endpoint = EstopEndpoint(self._estop_client, 'GNClient', 9.0)
        except:
            # Not the estop.
            self._estop_client = None
            self._estop_endpoint = None
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._natnet_client = NatNetClient()
        self.band_tar = np.array([1.5,0.2,0.6])
        self.robot_tar = np.array([0.6,0.5,0])
        self.table_center= np.array([1.5,1.5,0.83])
        self.robot_pos_origin = None
        self.task_name = 'MoDe-Spot-Curtain-v0'
        algo='manul'
        self.agent_flag = False
        self.agent_step_num = 0
        self.traj = []
        self.logs_path = os.getcwd() + f'/logs/{algo}/{self.task_name}/'+time.strftime("%m%d%H%M")
        os.makedirs(self.logs_path, exist_ok=True)



        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        self._opti_thred = None #OptiTrackReader(self._natnet_client)
        self._realsense_thread = RealSenseReader()
        #self._image_task = AsyncImageCapture(robot)
        self._async_tasks = AsyncTasks([
                                        self._robot_state_task,
                                        #self._image_task
                                        ])
        self._lock = threading.Lock()

        self._command_dictionary = {
            27: self._stop,  # ESC key
            ord('\t'): self._quit_program,
            #ord('T'): self._toggle_time_sync,
            ord(' '): self._toggle_estop,
            ord('r'): self._return_to_origin,
            ord('P'): self._toggle_power,
            ord('p'): self._toggle_power,
            ord('c'): self._sit,
            ord('b'): self._reset_agent_status,
            ord('f'): self._stand,
            ord('w'): self._move_forward,
            ord('s'): self._move_backward,
            ord('a'): self._strafe_left,
            ord('d'): self._strafe_right,
            ord('q'): self._turn_left,
            ord('e'): self._turn_right,
            ord('i'): self._realsense_thread.start_display,
            #ord('O'): self._image_task.toggle_video_mode,
            ord('u'): self._unstow,
            ord('l'): self._stow,
            #ord('l'): self._toggle_lease
        }
        self._locked_messages = ['', '', '']  # string: displayed message for user
        self._estop_keepalive = None
        self._exit_check = None

        # Stuff that is set in start()
        self._robot_id = None
        self._lease_keepalive = None

    def start(self):
        """Begin communication with the robot."""
        # Construct our lease keep-alive object, which begins RetainLease calls in a thread.
        self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                               return_at_exit=True)

        self._robot_id = self._robot.get_id()
        if self._estop_endpoint is not None:
            self._estop_endpoint.force_simple_setup(
            )  # Set this endpoint as the robot's sole estop.
        if self._opti_thred is not None:
            self._opti_thred.start()
        if self._realsense_thread is not None:
            self._realsense_thread.start()

    def shutdown(self):
        """Release control of robot as gracefully as possible."""
        LOGGER.info('Shutting down WasdInterface.')
        if self._estop_keepalive:
            # This stops the check-in thread but does not stop the robot.
            self._estop_keepalive.shutdown()
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
        #stop optitrack thread
        if self._opti_thred is not None:
            self._opti_thred.client.shutdown()


    def flush_and_estop_buffer(self, stdscr):
        """Manually flush the curses input buffer but trigger any estop requests (space)"""
        key = ''
        while key != -1:
            key = stdscr.getch()
            if key == ord(' '):
                self._toggle_estop()

    def add_message(self, msg_text):
        """Display the given message string to the user in the curses interface."""
        with self._lock:
            self._locked_messages = [msg_text] + self._locked_messages[:-1]

    def message(self, idx):
        """Grab one of the 3 last messages added."""
        with self._lock:
            return self._locked_messages[idx]

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_task.proto
 
    @property
    def opti_state(self):
        return self._opti_thred.get_latest()


    def realsense_state(self):
        return self._realsense_thread.get_image()
    
    def _reset_agent_status(self):
        self.agent_flag = not self.agent_flag
        if len(self.traj)>1:
            self.save_trajectory_pickle()
        if self.agent_flag:
            if self._opti_thred is not None:
                self.set_robot_origin_position()
            self.agent_step_num = 0
    
    def set_robot_origin_position(self):
        state = self.opti_state
        self.robot_pos_origin = np.array(state['spot']['pos'])
        self.robot_pos_origin[-1] = 0
        #self.table_tar = np.array(state['table']['pos'])

    def get_visual_feature(self):
        rs_data = self.realsense_state()
        image = rs_data['rgb']
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb, 'RGB')
        img.save(self.logs_path + f'/image_{self.agent_step_num}.png')
        #state = self.encoder.extract_dino_features(img_rgb)
        ob_dict={
            'rs_data':rs_data,
            #'state':state,
        }
        return None,ob_dict
    

    def get_ob_buff(self,):
        robot_pos,robot_rot,elem_pos,filtered_point =self._opti_thred.get_state(trans=self.robot_pos_origin)
        #np.array(dof_position),np.array(dof_velocity),joint_state,robot_rot,spot_state.kinematic_state
        dof_pos,dof_vel,joint_dict,robot_rot_r,spot_state = self.get_dof_state()
        obs_dict={
            'robot_pos':robot_pos,
            'robot_rot':robot_rot,
            'elem_pos':elem_pos,
            'filtered_point':filtered_point,
            'dof_pos':dof_pos,
            'dof_vel':dof_vel,
            'spot_state':spot_state
        }
        task = self.task_name.split('-')[-2]
        if task == 'Place':
            obs = np.concatenate(
                (
                    filtered_point.reshape(-1),
                    robot_pos,
                    robot_rot_r ,# maybe need change
                    self.table_center,#table center
                    dof_pos,
                    dof_vel
                ),
                # axis=-1
            )
        elif task =='Drag':
            obs = np.concatenate(
                (
                    filtered_point.reshape(-1),
                    robot_pos,
                    robot_rot_r,  # maybe need change
                    self.band_tar,#[(1.5,0.2,0.6)],  #band tar
                    self.robot_tar,#[(0.6, 0.5, 0.0)],  # band tar
                    dof_pos,
                    dof_vel
                ), axis=-1
            )
    
        return obs,obs_dict
        '''
        Self.band_tar:(1.5,0.2,0.6)
        self.robot_tar:(0.6,0.5,0)
        self.table_center:(1.5,1.5,0.83)

        '''

    def excuate_act_hand(self,action):
        #action = action.squeeze()
        base_act = action[ :3]*VELOCITY_BASE_SPEED * 0.5 # v_x,v_y,w_z 2*0.5
        arm_hand_comd = action[3:6]
        command = self.body_hand_command(base_act,arm_hand_comd)

        self._start_robot_command('agent_action',command_proto=command)


    def excuate_act_joint(self,action):
        base_act = action[ :3]*0.1  # v_x,v_y,w_z
        arm_joint_comd = action[ 3:]
        arm_joint_act = arm_joint_comd+ 0.06 * arm_joint_comd
        clamped_actions = np.clip(arm_joint_act, arm_low_limit, arm_up_limit)
        command = self.body_arm_command(base_act,clamped_actions)

        self._start_robot_command('agent_action',command_proto=command)



    def compute_rewards_drag(self,):
        robot_pos, robot_rot, elem_pos, filtered_point = self._opti_thred.get_state(trans=self.robot_pos_origin)
        #elem_pos_obs = filtered_point - self.robot_pos_origin
        #robot_pos = robot_pos-self.robot_pos_origin
        bandmid_position = np.mean(elem_pos,axis=0)
        bandmid_tar_dis = np.sum(abs(bandmid_position - self.band_tar)[ :2]) * 2

        robot_tar_dis = np.sum(abs(robot_pos - self.robot_tar)[ :2])

        distance = bandmid_tar_dis + robot_tar_dis
        # x first, then y
        rewards = - distance
        if robot_pos[ 0] < self.band_tar[0]:
            rewards = rewards + 0.5

        ele_flag = ( (elem_pos[:,1]>self.band_tar[1]).sum(-1)>1 )
        robot_tar_dis = np.sum(abs(robot_pos - self.robot_tar)[:2])
        flag = ele_flag  & (robot_pos[0]<self.band_tar[0]) & (robot_tar_dis < 0.3)
        if flag:
            rewards = rewards + 3
        return rewards


    def compute_rewards_place(self,):
        robot_pos, robot_rot, elem_pos, filtered_point = self._opti_thred.get_state(trans = self.robot_pos_origin)
        #all_elem = elem_pos - self.robot_pos_origin
        #robot_pos = robot_pos - self.robot_pos_origin
        #elem_pos_obs = filtered_point - self.robot_pos_origin

        distances = np.linalg.norm(elem_pos - robot_pos, axis=1)
        min_index = np.argmin(distances)
        max_index = np.argmax(distances)

        ee_point = elem_pos[min_index]
        rod_ee_position = elem_pos[max_index]

        robot_table_dis = np.sqrt(np.sum((robot_pos[ 0:2] - self.table_center[ 0:2]) ** 2))
        robotee_table_dis = np.sqrt(np.sum((ee_point - self.table_center) ** 2))
        rod_ee_table_dis = np.sqrt(np.sum((rod_ee_position - self.table_center) ** 2))

        rodmid_position = np.mean(elem_pos,axis=0)
        rodmid_table_dis = np.sqrt(np.sum((rodmid_position - self.table_center) ** 2))

        distance = robot_table_dis + robotee_table_dis + rod_ee_table_dis + rodmid_table_dis

        rewards = - distance
        if rod_ee_position[ -1] > 0.7:
            rewards = rewards + 0.75
        if rodmid_position[-1] > 0.7:
            rewards= rewards + 0.75

        table_pos = self.table_center
        table_limit_x_low = table_pos[ 0] - 0.38
        table_limit_x_upp = table_pos[ 0] + 0.38
        x_flag = (rod_ee_position[ 0] > table_limit_x_low) & (table_limit_x_upp > rod_ee_position[ 0])

        table_limit_y_low = table_pos[ 1] - 0.54
        table_limit_y_upp = table_pos[ 1] + 0.54
        y_flag = (rod_ee_position[ 1] > table_limit_y_low) & (table_limit_y_upp > rod_ee_position[ 1])

        table_limit_z_low = table_pos[ -1]
        table_limit_z_upp = table_pos[ -1] + 0.08
        z_ee_flag = (rod_ee_position[ -1] > table_limit_z_low) & (table_limit_z_upp > rod_ee_position[ -1])
        z_mid_flag = (rodmid_position[ -1] > table_limit_z_low)

        flag = x_flag & y_flag & z_ee_flag #& z_mid_flag
        if flag:
            rewards =  rewards + 10

        return rewards,flag


    def body_hand_command(self,body_vel,hand_pos_delta,seperate=True):
        if not seperate:
            return

        if body_vel.any() !=0:
            mobility_command = RobotCommandBuilder.synchro_velocity_command(v_x=body_vel[0], v_y=body_vel[1], v_rot=body_vel[2])
            command = mobility_command
        elif hand_pos_delta.any() !=0:
            hand_pos, hand_rot = self.get_hand_state()
            target_hand_pos = hand_pos + hand_pos_delta
            x = target_hand_pos[0]
            y = target_hand_pos[1]
            z = target_hand_pos[2]
            arm_command = RobotCommandBuilder.arm_pose_command(
                x, y, z, hand_rot.w, hand_rot.x,
                hand_rot.y, hand_rot.z, 'flat_body', ARM_CMD_DURATION)
            command = arm_command
        return command





    def arm_body_command(self,body_vel,arm_cmd):
        command = robot_command_pb2.RobotCommand()
        sh0 = arm_cmd[0]
        sh1 = arm_cmd[1]
        el0 = arm_cmd[2]
        el1 = arm_cmd[3]
        wr0 = arm_cmd[4]
        wr1 = arm_cmd[5]
        traj_point = RobotCommandBuilder.create_arm_joint_trajectory_point(
            sh0, sh1, el0, el1, wr0, wr1)
        arm_joint_traj = arm_command_pb2.ArmJointTrajectory(points=[traj_point])
        joint_move_command = arm_command_pb2.ArmJointMoveCommand.Request(trajectory=arm_joint_traj)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_joint_move_command=joint_move_command)
        command.synchronized_command.arm_command.CopyFrom(arm_command)
        v_x = body_vel[0]
        v_y = body_vel[1]
        v_rot = body_vel[2]
        command.synchronized_command.mobility_command.se2_velocity_request.velocity.linear.x = v_x
        command.synchronized_command.mobility_command.se2_velocity_request.velocity.linear.y = v_y
        command.synchronized_command.mobility_command.se2_velocity_request.velocity.angular = v_rot
        command.synchronized_command.mobility_command.se2_velocity_request.se2_frame_name = 'flat_body'#BODY_FRAME_NAME

        return command

    def get_dof_state(self):
        spot_state = self.robot_state
        joint_state = {}
        for joint in spot_state.kinematic_state.joint_states:
            joint_state[joint.name] = {
                'position': joint.position.value,
                'velocity': joint.velocity.value
            }

        # Extract values in the given order
        dof_position = []
        dof_velocity = []
        for joint_name in joint_order_sim:
            # Replace _ with . to match the dictionary key format
            key = joint_name.replace('_', '.')
            if key in joint_state:
                dof_position.append(joint_state[key]['position'])
                dof_velocity.append(joint_state[key]['velocity'])
        
        robot = get_a_tform_b(spot_state.kinematic_state.transforms_snapshot,
                                 'vision', 'flat_body')
        #flat_body is body center
        robot_pos = np.array(robot.get_translation())  # same to hand.position, but position return vec3
        robot_rot = np.array([robot.rot.w,robot.rot.x, robot.rot.y, robot.rot.z])

        return np.array(dof_position),np.array(dof_velocity),joint_state,robot_rot,spot_state.kinematic_state

    def get_hand_state(self):
        spot_state = self.robot_state

        hand = get_a_tform_b(spot_state.kinematic_state.transforms_snapshot,
                              'flat_body', 'hand')

        '''
        hand = get_a_tform_b(state.kinematic_state.transforms_snapshot,
                             BODY_FRAME_NAME, HAND_FRAME_NAME)
        hand_pos = np.array(hand.get_translation())  # same to hand.position, but position return vec3
        hand_rot = np.array([hand.rot.x, hand.rot.y, hand.rot.z, hand.rot.w])
        hand_7 = np.concatenate([hand_pos, hand_rot])
        '''
        # flat_body is body center
        hand_pos = np.array(hand.get_translation())
        hand_rot =hand.rot
        # same to hand.position, but position return vec3
        #hand_rot = np.array([hand.rot.w, hand.rot.x, hand.rot.y, hand.rot.z])

        return hand_pos,hand_rot
    

    def drive(self, stdscr):
        """User interface to control the robot via the passed-in curses screen interface object."""
        with ExitCheck() as self._exit_check:
            curses_handler = CursesHandler(self)
            curses_handler.setLevel(logging.INFO)
            #LOGGER.addHandler(curses_handler)

            stdscr.nodelay(True)  # Don't block for user input.
            stdscr.resize(26, 140)
            stdscr.refresh()

            # for debug
            curses.echo()

            try:
                while not self._exit_check.kill_now:
                    self._async_tasks.update()
                    self._drive_draw(stdscr, self._lease_keepalive)

                    try:
                        cmd = stdscr.getch()
                        # Do not queue up commands on client
                        self.flush_and_estop_buffer(stdscr)
                        if cmd in [ord('w'),ord('a'),ord('s'),ord('d'),
                                 ord('u'),ord('j'),ord('h'),ord('k'),ord('m'),ord('n')] :
                            
                            obs,obs_dict = self.get_visual_feature()
                            cmd_chr = chr(cmd)
                            action = Action_map[cmd_chr][:6]    
                            action = np.array(action)
                            self.excuate_act_hand(action)
                            time.sleep(0.5) #COMMAND_INPUT_RATE

                            sample = {
                                'ob':obs_dict,
                                'action':action,
                                'time_step':self.agent_step_num
                            }

                            self.traj.append(sample)
                            self.agent_step_num = self.agent_step_num +1
                            if self.agent_flag:
                                self.save_sample_pickle(sample)
                        else:
                            self._drive_cmd(cmd)
                            time.sleep(COMMAND_INPUT_RATE)
                    except Exception:
                        # On robot command fault, sit down safely before killing the program.
                        logging.error("WASD crashed:\n%s", traceback.format_exc())
                        self._safe_power_off()
                        time.sleep(2.0)
                        
                        raise

            finally:
                LOGGER.removeHandler(curses_handler)

    def _drive_draw(self, stdscr, lease_keep_alive):
        """Draw the interface screen at each update."""
        stdscr.clear()  # clear screen
        stdscr.resize(26, 140)
        stdscr.addstr(0, 0, f'{self._robot_id.nickname:20s} {self._robot_id.serial_number}')
        stdscr.addstr(1, 0, self._lease_str(lease_keep_alive))
        stdscr.addstr(2, 0, self._battery_str())
        stdscr.addstr(3, 0, self._estop_str())
        stdscr.addstr(4, 0, self._power_state_str())
        stdscr.addstr(5, 0, self._time_sync_str())
        for i in range(3):
            stdscr.addstr(7 + i, 2, self.message(i))
        stdscr.addstr(10, 0, 'Commands: [TAB]: quit                               ')
        stdscr.addstr(11, 0, '          [T]: Time-sync, [SPACE]: Estop, [P]: Power')
        stdscr.addstr(12, 0, '          [I]: Take image, [O]: Video mode          ')
        stdscr.addstr(13, 0, '          [f]: Stand, [r]: Self-right               ')
        stdscr.addstr(14, 0, '          [v]: Sit,                                 ')
        stdscr.addstr(15, 0, '          [wasd]: Directional strafing              ')
        stdscr.addstr(16, 0, '          [qe]: Turning, [ESC]: Stop                ')
        stdscr.addstr(17, 0, '          [l]: Return/Acquire lease                 ')
        stdscr.addstr(18, 0, self._rs_state_str())

        stdscr.refresh()

    def _drive_cmd(self, key):
        """Run user commands at each update."""
        try:
            cmd_function = self._command_dictionary[key]
            cmd_function()

        except KeyError:
            if key and key != -1 and key < 256:
                self.add_message(f'Unrecognized keyboard command: \'{chr(key)}\'')

    def _try_grpc(self, desc, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            self.add_message(f'Failed {desc}: {err}')
            return None

    def _try_grpc_async(self, desc, thunk):

        def on_future_done(fut):
            try:
                fut.result()
            except (ResponseError, RpcError, LeaseBaseError) as err:
                self.add_message(f'Failed {desc}: {err}')
                return None

        future = thunk()
        future.add_done_callback(on_future_done)

    def _quit_program(self):
        self._sit()

        if self._exit_check is not None:
            self._exit_check.request_exit()
        self._toggle_lease()

    def _toggle_estop(self):
        """toggle estop on/off. Initial state is ON"""
        if self._estop_client is not None and self._estop_endpoint is not None:
            if not self._estop_keepalive:
                self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
            else:
                self._try_grpc('stopping estop', self._estop_keepalive.stop)
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None

    def _toggle_lease(self):
        """toggle lease acquisition. Initial state is acquired"""
        if self._lease_client is not None:
            if self._lease_keepalive is None:
                self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                                       return_at_exit=True)
            else:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None

    def _start_robot_command(self, desc, command_proto, end_time_secs=None):

        def _start_command():
            self._robot_command_client.robot_command(command=command_proto,
                                                     end_time_secs=time.time() + 0.5)

        self._try_grpc(desc, _start_command)


    def _sit(self):
        self._start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())

    def _stand(self):
        self._start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())
        if self._opti_thred is not None:
            self.set_robot_origin_position()
        

    def _move_forward(self):
        self._velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

    def _move_backward(self):
        self._velocity_cmd_helper('move_backward', v_x=-VELOCITY_BASE_SPEED)

    def _strafe_left(self):
        self._velocity_cmd_helper('strafe_left', v_y=VELOCITY_BASE_SPEED)

    def _strafe_right(self):
        self._velocity_cmd_helper('strafe_right', v_y=-VELOCITY_BASE_SPEED)

    def _turn_left(self):
        self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

    def _turn_right(self):
        self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)

    def _stop(self):
        self._start_robot_command('stop', RobotCommandBuilder.stop_command())

    def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
        self._start_robot_command(
            desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION)

    def _stow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_stow_command())

    def _unstow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_ready_command())

    def _return_to_origin(self):
        self._start_robot_command(
            'fwd_and_rotate',
            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=0.0, goal_y=0.0, goal_heading=0.0, frame_name=ODOM_FRAME_NAME, params=None,
                body_height=0.0, locomotion_hint=spot_command_pb2.HINT_SPEED_SELECT_TROT),
            end_time_secs=time.time() + 20)


    def _toggle_power(self):
        power_state = self._power_state()
        if power_state is None:
            self.add_message('Could not toggle power because power state is unknown')
            return

        if power_state == robot_state_proto.PowerState.STATE_OFF:
            self._try_grpc_async('powering-on', self._request_power_on)
        else:
            self._try_grpc('powering-off', self._safe_power_off)

    def _request_power_on(self):
        request = PowerServiceProto.PowerCommandRequest.REQUEST_ON
        return self._power_client.power_command_async(request)

    def _safe_power_off(self):
        self._start_robot_command('safe_power_off', RobotCommandBuilder.safe_power_off_command())

    def _power_state(self):
        state = self.robot_state
        if not state:
            return None
        return state.power_state.motor_power_state

    def _lease_str(self, lease_keep_alive):
        if lease_keep_alive is None:
            alive = 'STOPPED'
            lease = 'RETURNED'
        else:
            try:
                _lease = lease_keep_alive.lease_wallet.get_lease()
                lease = f'{_lease.lease_proto.resource}:{_lease.lease_proto.sequence}'
            except bosdyn.client.lease.Error:
                lease = '...'
            if lease_keep_alive.is_alive():
                alive = 'RUNNING'
            else:
                alive = 'STOPPED'
        return f'Lease {lease} THREAD:{alive}'

    def _power_state_str(self):
        power_state = self._power_state()
        if power_state is None:
            return ''
        state_str = robot_state_proto.PowerState.MotorPowerState.Name(power_state)
        return f'Power: {state_str[6:]}'  # get rid of STATE_ prefix

    def _opti_state_str(self):
        state = self.opti_state
        if len(state)<1:
            return ''
        if len(self.traj)<1:
            flag = None
        else:
            flag = self.traj[-1]['done']
        pos = state['spot']['pos'] 
        rot = state['spot']['rot']
        return f'Spot State from opti : position {pos}\n\
            rotation is {rot}\n \
            Origin position {self.robot_pos_origin}\n \
            agent_flag {self.agent_flag}, Done is {flag}'  # get rid of STATE_ prefix

    def _rs_state_str(self):
        #state = self.realsense_state
        
        return f'Agent step number :  {self.agent_step_num}\n \
            agent_flag {self.agent_flag}, \
            '  # get rid of STATE_ prefix




    def _estop_str(self):
        if not self._estop_client:
            thread_status = 'NOT ESTOP'
        else:
            thread_status = 'RUNNING' if self._estop_keepalive else 'STOPPED'
        estop_status = '??'
        state = self.robot_state
        if state:
            for estop_state in state.estop_states:
                if estop_state.type == estop_state.TYPE_SOFTWARE:
                    estop_status = estop_state.State.Name(estop_state.state)[6:]  # s/STATE_//
                    break
        return f'Estop {estop_status} (thread: {thread_status})'

    def _time_sync_str(self):
        if not self._robot.time_sync:
            return 'Time sync: (none)'
        if self._robot.time_sync.stopped:
            status = 'STOPPED'
            exception = self._robot.time_sync.thread_exception
            if exception:
                status = f'{status} Exception: {exception}'
        else:
            status = 'RUNNING'
        try:
            skew = self._robot.time_sync.get_robot_clock_skew()
            if skew:
                skew_str = f'offset={duration_str(skew)}'
            else:
                skew_str = '(Skew undetermined)'
        except (TimeSyncError, RpcError) as err:
            skew_str = f'({err})'
        return f'Time sync: {status} {skew_str}'

    def _battery_str(self):
        if not self.robot_state:
            return ''
        battery_state = self.robot_state.battery_states[0]
        status = battery_state.Status.Name(battery_state.status)
        status = status[7:]  # get rid of STATUS_ prefix
        if battery_state.charge_percentage.value:
            bar_len = int(battery_state.charge_percentage.value) // 10
            bat_bar = f'|{"=" * bar_len}{" " * (10 - bar_len)}|'
        else:
            bat_bar = ''
        time_left = ''
        if battery_state.estimated_runtime:
            time_left = f'({secs_to_hms(battery_state.estimated_runtime.seconds)})'
        return f'Battery: {status}{bat_bar} {time_left}'

    def save_sample_pickle(self,sample, step=None, dir_path=None):
        if dir_path is None:
            dir_path = self.logs_path

        if step is None:
            step = self.agent_step_num
        os.makedirs(dir_path + '/samples/', exist_ok=True)
        filename = os.path.join(dir_path, f"samples/sample_step_{step}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(sample, f)

    def save_trajectory_pickle(self,trajectory=None, path=None):
        if path is None:
            path  = self.logs_path
        if trajectory is None:
            trajectory = self.traj
        
        filename = os.path.join(path, f"trajs.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(trajectory, f)
        
def _setup_logging(verbose):
    """Log to file at debug level, and log to console at INFO or DEBUG (if verbose).

    Returns the stream/console logger so that it can be removed when in curses mode.
    """
    LOGGER.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Save log messages to file wasd.log for later debugging.
    file_handler = logging.FileHandler('wasd.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(file_handler)

    # The stream handler is useful before and after the application is in curses-mode.
    if verbose:
        stream_level = logging.DEBUG
    else:
        stream_level = logging.INFO

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(log_formatter)
    LOGGER.addHandler(stream_handler)
    return stream_handler



def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--time-sync-interval-sec',
                        help='The interval (seconds) that time-sync estimate should be updated.',
                        type=float)
    
    options = parser.parse_args(['10.0.0.3'])

    stream_handler = _setup_logging(options.verbose)

    # Create robot object.
    sdk = create_standard_sdk('WASDClient')
    #robot = sdk.create_robot(options.hostname)
    robot = sdk.create_robot('10.0.0.30')

    try:
        robot.authenticate('rllab', 'robotlearninglab')
        bosdyn.client.util.authenticate(robot)
        robot.start_time_sync(options.time_sync_interval_sec)
    except RpcError as err:
        LOGGER.error('Failed to communicate with robot: %s', err)
        return False




    wasd_interface = WasdInterface(robot)

    try:
        wasd_interface.start()
    except (ResponseError, RpcError) as err:
        LOGGER.error('Failed to initialize robot communication: %s', err)
        return False

    LOGGER.removeHandler(stream_handler)  # Don't use stream handler in curses mode.


    try:
        try:
            # Prevent curses from introducing a 1-second delay for ESC key
            os.environ.setdefault('ESCDELAY', '0')
            # Run wasd interface in curses mode, then restore terminal config.
            curses.wrapper(wasd_interface.drive)
        finally:
            # Restore stream handler to show any exceptions or final messages.
            LOGGER.addHandler(stream_handler)

    except Exception as e:
        LOGGER.error('WASD has thrown an error: [%r] %s', e, e)
    finally:
        # Do any final cleanup steps.
        wasd_interface.shutdown()

    return True

 
if __name__ == '__main__':
    if not main():
        sys.exit(1)