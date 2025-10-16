from __future__ import annotations

import torch
import os


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg,RigidObjectCfg,RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import CameraCfg,Camera

from cfg.robotcfg import SPOT_CFG
from controller.spot_operational_space import OperationSpaceController
from omni.isaac.lab.markers import VisualizationMarkers,VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG


Debug_vis = False

@configclass
class SpotCurtainEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 5  #actions 10 times then obs 1
    episode_length_s = 6000.0 / 40.0
    observation_space = 114

    action_space = 9
    Debug_vis = False
    camera = True

    viewer = ViewerCfg(eye=(-5.89, 4.89, 6.36))

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        use_fabric= True
                                       )
    root = os.getcwd()
    # robot need to change
    robot_cfg: ArticulationCfg = SPOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.usd_path = root + '/asset/spot/spot_arm_fix_cam.usd'

    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/center_camera/center_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb","depth"], #["rgb"], #
        spawn= None
    )


    # table cfg
    curtain_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/curtain",
        spawn=sim_utils.UsdFileCfg(
            usd_path=root + f"/asset/objects/curtain.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 1.5, 0.2),
                                                  rot=(0.707, 0, 0, 0.707,)
                                                  ),

    )

    if Debug_vis:
        ee_maker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
        ee_maker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_maker_cfg.prim_path = "/Visuals/ee_current"

        goal_maker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.copy()
        goal_maker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_maker_cfg.prim_path = "/Visuals/ee_goal"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=4.0, replicate_physics=False)

# pre-physics step calls
#   |-- _pre_physics_step(action)
#   |-- _apply_action()
# post-physics step calls
#   |-- _get_dones()
#   |-- _get_rewards()
#   |-- _reset_idx(env_ids)
#   |-- _get_observations()

class SpotCurtainEnv( DirectRLEnv):
    cfg: SpotCurtainEnvCfg
    def __init__(self,
                 cfg: SpotCurtainEnvCfg,
                 render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)

        self.num_actions = cfg.action_space
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self.robot.num_joints), dtype=torch.float, device=self.sim.device
        )
        self.robot_dof_pos = torch.zeros(
            (self.num_envs, self.robot.num_joints), device=self.sim.device)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.sim.device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.flag = torch.zeros((self.num_envs), device=self.sim.device)

        self.ee_idx = self.robot.find_bodies("arm_ee")[0][0]
        self._ee_name = 'arm0_link_ee'
        self.controller.init_ctrl(self._ee_name,self.robot.body_names,self.robot.joint_names)


    def _setup_scene(self,) :
        self.robot = Articulation(self.cfg.robot_cfg)
        self.curtain = RigidObject(self.cfg.curtain_cfg)
        self._camera = Camera(self.cfg.camera_cfg)

        #for debug maker
        if self.cfg.Debug_vis:
            self.ee_marker = VisualizationMarkers(self.cfg.ee_maker_cfg)
            self.goal_marker = VisualizationMarkers(self.cfg.goal_maker_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground",
                           cfg=GroundPlaneCfg(color=(0.75, 0.75, 0.75)))
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["camera"] = self._camera

        self.controller = OperationSpaceController(num_robot=self.num_envs,
                                                   device=self.device,
                                                   )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        num_indices = len(env_ids)

        limit = self.robot.data.joint_limits[:, :, :]
        pos = torch.clamp(self.robot.data.default_joint_pos[env_ids], limit[:, :, 0], limit[:, :, 1])

        dof_pos = torch.zeros((num_indices, self.robot.num_joints), device=self.sim.device)
        dof_vel = torch.zeros((num_indices, self.robot.num_joints), device=self.sim.device)
        dof_pos[:, :] = pos
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, None, env_ids)



    def _pre_physics_step(self, actions):

        self.actions = actions.clone().to(self.sim.device)
        lin_vel = self.robot.data.root_lin_vel_b
        ang_vel = self.robot.data.root_ang_vel_b
        gravity_b = self.robot.data.projected_gravity_b
        current_joint_pos = self.robot.data.joint_pos
        current_joint_vel = self.robot.data.joint_vel
        body_state_w = self.robot.data.body_state_w
        arm_comd =None
        if self.actions[:,3:].any()!=0:
            arm_comd = self.actions[:,3:]

        # do not set the base_pose if arm related to body frame
        action,index,_ = self.controller.compute(lin_vel, ang_vel,  gravity_b,
                                                  current_joint_pos, current_joint_vel,
                                                  body_state_w,
                                                  self.actions[:,:3], #body command
                                                  arm_comd) #, success

        self.robot_dof_targets[:, index] = action


        #self.robot_dof_targets[:, self.arm_idxs] = arm_joint_act
        limit = self.robot.data.joint_limits[:, :, :]
        self.robot_dof_targets = torch.clamp(self.robot_dof_targets, limit[:, :, 0], limit[:, :, 1])



    def _apply_action(self):
        self.robot.set_joint_position_target(self.robot_dof_targets) # 10 times
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_idx, 0:7] #ee position
        # update marker positions
        if self.cfg.Debug_vis:
            self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

    def _get_image_obs(self):
        camera_data = {}
        process = False
        for data_type in self.cfg.camera_cfg.data_types:
            if data_type == "rgb":
                tem_data = self._camera.data.output[data_type].to(torch.uint8)  # / 255.0  # 【1，480，640，3】
                if process:
                    encode_feature = self.encoder.extract_dino_features(tem_data)
                    camera_data['dino_feature'] = encode_feature
            elif data_type == "depth":
                tem_data = self._camera.data.output[data_type]
                tem_data[tem_data == float("inf")] = 0
            else:
                tem_data = self._camera.data.output[data_type]

            camera_data[data_type] = tem_data

        return camera_data

    # collect observation info
    # need to debug
    def _get_observations(self, ):
        if self.cfg.camera:
            camera_data = self._get_image_obs()

        robot_position = self.robot.data.root_pos_w #[num_env,3]
        robot_rot=self.robot.data.root_quat_w #[num_env,4]

        robot_dof_pos = self.robot.data.joint_pos
        robot_dof_poses =  robot_dof_pos.reshape(self.num_envs, -1).to(dtype=torch.float)

        robot_dof_vel = self.robot.data.joint_vel
        robot_dof_vels = robot_dof_vel.reshape(self.num_envs, -1).to(dtype=torch.float)


#####must return the obs or there would be error
        observations = {
            'rgb':camera_data['rgb'],
            'depth':camera_data['depth'],
            'robot_position':robot_position,
            'robot_dof_pos':robot_dof_pos,
            'robot_dof_target':self.robot_dof_targets.clone(),
                            }
        return observations

    def _get_states(self) -> VecEnvObs | None:
        return None

    def pass_flag(self) -> None:
        curtain_position = self.curtain.data.root_pos_w[:, 0] + 0.3
        robot_current_position = self.robot.data.root_pos_w[:, 0]  # body_state_w[:, self.base_idx, 0]
        robot_origin_position = self.cfg.robot_cfg.init_state.pos[0]
        flag = (robot_origin_position - curtain_position) * (robot_current_position - curtain_position) < 0

        self.flag = flag
        if self.flag:
            print('Pass')
        return flag

    # know if it finish or not
    def _get_dones(self) -> None:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        resets = self.pass_flag()
        return resets, time_out

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        total_reward = self.flag
        return total_reward
