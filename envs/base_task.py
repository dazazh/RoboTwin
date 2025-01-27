import mplib.planner
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import mplib
import numpy as np
import gymnasium as gym
import pdb
import numpy as np
from PIL import Image, ImageColor
import toppra as ta
import open3d as o3d
import json
import transforms3d as t3d
from .utils import fps
from collections import OrderedDict
from .utils import *
import collections
from collections import deque
import cv2
import torch
from copy import deepcopy

class Base_task(gym.Env):

    def __init__(self):
        pass

    def _init(self, **kwags):
        '''
            Initialization
            - `self.PCD_INDEX`: The index of the file saved for the current scene.
            - `self.fcitx5-configtool`: Left gripper pose (close <=0, open >=0.4).
            - `self.ep_num`: Episode ID.
            - `self.task_name`: Task name.
            - `self.save_dir`: Save path.
            - `self.left_original_pose`: Left arm original pose.
            - `self.right_original_pose`: Right arm original pose.
            - `self.left_arm_joint_id`: [6,14,18,22,26,30].
            - `self.right_arm_joint_id`: [7,15,19,23,27,31].
            - `self.render_fre`: Render frequency.
        '''
        super().__init__()
        ta.setup_logging("CRITICAL") # hide logging
        np.random.seed(kwags.get('seed', 0))

        global left_pub_data
        global right_pub_data

        self.PCD_INDEX = 0
        self.task_name = kwags.get('task_name')
        self.save_dir = kwags.get('save_path', 'data') +'/' + kwags.get('task_name', 'Empty')
        self.ep_num = kwags.get('now_ep_num', 0)
        self.render_freq = kwags.get('render_freq', 10)
        self.data_type = kwags.get('data_type', None)
        self.save_type = kwags.get('save_type', None)
        self.pcd_crop = kwags.get('pcd_crop', False)
        self.pcd_down_sample_num = kwags.get('pcd_down_sample_num', 0)
        self.is_save = kwags.get('is_save', False)
        self.pcd_crop_bbox = kwags.get('bbox', [[-0.6, -0.35, 0.7401],[0.6, 0.35, 2]])
        self.dual_arm = kwags.get('dual_arm', True)
        self.table_static = kwags.get('table_static', True)
        self.head_camera_fovy = kwags.get('head_camera_fovy', 45)
        self.save_freq = kwags.get('save_freq', 15)
        self.file_path = []
        self.plan_success = True
        self.left_plan_success = True
        self.right_plan_success = True
        self.step_lim = None
        self.fix_gripper = False
        self.setup_scene()

        self.left_js = None
        self.right_js = None
        self.raw_head_pcl = None
        self.real_head_pcl = None
        self.real_head_pcl_color = None
        
        left_pub_data = [0,0,0,0,0,0,0]
        right_pub_data = [0,0,0,0,0,0,0]

        self.grasp_direction_dic = {
            'left':         [0,      0,   0,    -1],
            'front_left':   [-0.383, 0,   0,    -0.924],
            'front' :       [-0.707, 0,   0,    -0.707],
            'front_right':  [-0.924, 0,   0,    -0.383],
            'right':        [-1,     0,   0,    0],
            'top_down':     [-0.5,   0.5, -0.5, -0.5],
        }

        self.world_direction_dic = {
            'left':         [0.5,  0.5,  0.5,  0.5],
            'front_left':   [0.65334811, 0.27043713, 0.65334811, 0.27043713],
            'front' :       [0.707, 0,    0.707, 0],
            'front_right':  [0.65334811, -0.27043713,  0.65334811, -0.27043713],
            'right':        [0.5,    -0.5, 0.5,  0.5],
            'top_down':     [0,      0,   1,    0],
        }
        self.target_left_pose_front = [-0.19,-0.12,0.92,1,0,0,0]
        self.target_right_pose_front = [0.19,-0.12,0.92,-0.01,0.01,0.03,-1]
        self.handover_block_pose = [-0.054, -0.09,  0.82]
        self.actor_name_dic = {}
        self.actor_data_dic = {}
        self.used_contant = []
        self.left_prepare_grasp_data = None
        self.left_prepare_grasp_point_group = None
        self.right_prepare_grasp_data = None
        self.right_prepare_grasp_point_group = None

        self.pre_left_pose = None
        self.pre_right_pose = None

        self.pose_of_close_left_gripper = None
        self.pose_of_close_right_gripper = None
        
        self.now_left_pose = None
        self.now_right_pose = None

    def setup_scene(self,**kwargs):
        '''
        Set the scene
            - Set up the basic scene: light source, viewer.
        '''
        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config
        set_global_config(max_num_materials = 50000, max_num_textures = 50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)
        
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        # set simulation timestep
        self.scene.set_timestep(kwargs.get("timestep", 1 / 250))
        # add ground to scene
        self.scene.add_ground(kwargs.get("ground_height", 0))
        # set default physical material
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 0.9),
            kwargs.get("dynamic_friction", 0.9),
            kwargs.get("restitution", 0),
        )
        # give some white ambient light of moderate intensity
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        # default enable shadow unless specified otherwise
        shadow = kwargs.get("shadow", True)
        # default spotlight angle and intensity
        direction_lights = kwargs.get(
            "direction_lights", [[[0, 0.5, -1], [0.5, 0.5, 0.5]]]
        )
        for direction_light in direction_lights:
            self.scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        # default point lights position and intensity
        point_lights = kwargs.get(
            "point_lights",
            [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]]
        )
        for point_light in point_lights:
            self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)

        # initialize viewer with camera position and orientation
        if self.render_freq:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(
                x=kwargs.get("camera_xyz_x", 0.4),
                y=kwargs.get("camera_xyz_y", 0.22),
                z=kwargs.get("camera_xyz_z", 1.5),
            )
            self.viewer.set_camera_rpy(
                r=kwargs.get("camera_rpy_r", 0),
                p=kwargs.get("camera_rpy_p", -0.8),
                y=kwargs.get("camera_rpy_y", 2.45),
            )

    def create_table_and_wall(self):
        # creat wall
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9), 
            name='wall',
        )

        # creat table
        self.table = create_table(
            self.scene,
            sapien.Pose(p=[0, 0, 0.74]),
            length=1.2,
            width=0.7,
            height=0.74,
            thickness=0.05,
            is_static=self.table_static
        )

    def load_robot(self, **kwargs):
        """
            load aloha robot urdf file, set root pose and set joints
        """
        # load urdf file
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True

        self.robot = loader.load(
            kwargs.get("urdf_path", "./aloha_maniskill_sim/urdf/arx5_description_isaac.urdf")
        )

        # set root pose 
        self.robot.set_root_pose(
            sapien.Pose(
                kwargs.get("robot_origin_xyz", [0, -0.65, 0]),
                kwargs.get("robot_origin_quat", [1, 0, 0, 1]),
            )
        )

        # set joints
        self.active_joints = self.robot.get_active_joints()

        for joint in self.active_joints:
            joint.set_drive_property(
                stiffness=kwargs.get("joint_stiffness", 1000),
                damping=kwargs.get("joint_damping", 200),
            )

        self.all_joints = self.robot.get_joints()
        self.all_links = self.robot.get_links()

        self.left_endpose = self.robot.find_joint_by_name('fl_joint6')
        self.right_endpose = self.robot.find_joint_by_name('fr_joint6')

        self.left_gripper_val = 0.
        self.right_gripper_val = 0.
        self.left_original_pose = [-0.3,-0.32,0.935,1,0,0,1]
        self.right_original_pose = [0.3,-0.32,0.935,1,0,0,1]
        self.left_arm_joint_id = [6,14,18,22,26,30]
        self.right_arm_joint_id = [7,15,19,23,27,31]
        self.left_init_pose = sapien.Pose([-0.29881, -0.311307, 0.9375], [-5.33147e-06, -0.707107, -0.707107, -5.04532e-06])
        self.right_init_pose = sapien.Pose([0.30119, -0.311307, 0.9375], [-5.33147e-06, -0.707107, -0.707107, -5.04532e-06])

    def load_camera(self,camera_w,camera_h):
        '''
            Add cameras and set camera parameters
                - Including four cameras: left, right, front, head.
        '''

        near, far = 0.1, 100
        width, height = camera_w, camera_h

        # front camera
        front_cam_pos = np.array([0, -0.45, 0.85])
        front_cam_forward = np.array([0,1,-0.1]) / np.linalg.norm(np.array([0,1,-0.1]))
        front_cam_left = np.cross([0, 0, 1], front_cam_forward)
        front_cam_left = front_cam_left / np.linalg.norm(front_cam_left)
        front_up = np.cross(front_cam_forward, front_cam_left)
        front_mat44 = np.eye(4)
        front_mat44[:3, :3] = np.stack([front_cam_forward, front_cam_left, front_up], axis=1)
        front_mat44[:3, 3] = front_cam_pos
        
        # head camera
        head_cam_pos = np.array([-0.032, -0.45, 1.35])
        head_cam_forward = np.array([0,0.1,-0.55]) 
        head_cam_left = np.cross([0, 0, 1], head_cam_forward)
        head_cam_left = head_cam_left / np.linalg.norm(head_cam_left)
        head_up = np.cross(head_cam_forward, head_cam_left)
        head_mat44 = np.eye(4)
        head_mat44[:3, :3] = np.stack([head_cam_forward, head_cam_left, head_up], axis=1)
        head_mat44[:3, 3] = head_cam_pos

        # observer camera
        observer_cam_pos = np.array([0.4, 0.22, 1.42])
        observer_cam_forward = np.array([-1,-1,-1])
        observer_cam_left = np.array([1,-1, 0])
        observer_up = np.cross(observer_cam_forward, observer_cam_left)
        observer_mat44 = np.eye(4)
        observer_mat44[:3, :3] = np.stack([observer_cam_forward, observer_cam_left, observer_up], axis=1)
        observer_mat44[:3, 3] = observer_cam_pos
        
        self.left_camera = self.scene.add_camera(
            name="left_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(37),
            near=near,
            far=far,
        )

        self.right_camera = self.scene.add_camera(
            name="right_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(37),
            near=near,
            far=far,
        )

        self.front_camera = self.scene.add_camera(
            name="front_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(37),
            near=near,
            far=far,
        )
        
        self.head_camera = self.scene.add_camera(
            name="head_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(self.head_camera_fovy),
            near=near,
            far=far,
        )

        self.observer_camera = self.scene.add_camera(
            name = "observer_camera",
            width=width,
            height=height,
            fovy=np.deg2rad(93),
            near=near,
            far=far,
        )

        self.front_camera.entity.set_pose(sapien.Pose(front_mat44))
        self.head_camera.entity.set_pose(sapien.Pose(head_mat44))
        self.observer_camera.entity.set_pose(sapien.Pose(observer_mat44))
        self.left_camera.entity.set_pose(self.all_links[46].get_pose())
        self.right_camera.entity.set_pose(self.all_links[49].get_pose())

        self.scene.step()  # run a physical step
        self.scene.update_render()  # sync pose from SAPIEN to renderer

    def setup_planner(self, **kwargs):
        """
            Create an mplib planner using the default robot.
            See planner.py for more details on the arguments.
        """
        self.left_planner = mplib.Planner(
            urdf=kwargs.get("urdf_path", "./aloha_maniskill_sim/urdf/arx5_description_isaac.urdf"),
            srdf=kwargs.get("srdf_path", "./aloha_maniskill_sim/srdf/arx5_description_isaac.srdf"),
            move_group=kwargs.get("move_group", "fl_link6"),
        )
        self.right_planner = mplib.Planner(
            urdf=kwargs.get("urdf_path", "./aloha_maniskill_sim/urdf/arx5_description_isaac.urdf"),
            srdf=kwargs.get("srdf_path", "./aloha_maniskill_sim/srdf/arx5_description_isaac.srdf"),
            move_group=kwargs.get("move_group", "fr_link6"),
        )

        robot_pose_in_world = [0,-0.65,0,1,0,0,1] 
        self.left_planner.set_base_pose(robot_pose_in_world)
        self.right_planner.set_base_pose(robot_pose_in_world)
    

    def _update_render(self):
        """
            Update rendering to refresh the camera's RGBD information 
            (rendering must be updated even when disabled, otherwise data cannot be collected).
        """
        self.left_camera.entity.set_pose(self.all_links[46].get_pose())
        self.right_camera.entity.set_pose(self.all_links[49].get_pose())
        self.scene.update_render()
        self.scene.update_render()

    def left_follow_path(self, result, save_freq=-1): # For left arm
        save_freq = self.save_freq if save_freq == -1 else save_freq
        n_step = result["position"].shape[0]

        if n_step > 2000:
            self.plan_success = False
            return

        if save_freq != None:
            self._take_picture()

        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            for j in range(len(self.left_arm_joint_id)):
                n_j = self.left_arm_joint_id[j]
                self.active_joints[n_j].set_drive_target(result["position"][i][j])
                self.active_joints[n_j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            self.scene.step()
            if i%5 == 0:
                self._update_render()
                if self.render_freq and i % self.render_freq == 0:
                    self.viewer.render()
            
            if save_freq != None and i % save_freq == 0:
                self._take_picture()

        if save_freq != None:
            self._take_picture()
    
    def right_follow_path(self, result, save_freq=-1): # For right arm
        save_freq = self.save_freq if save_freq == -1 else save_freq
        n_step = result["position"].shape[0]

        if n_step > 2000:
            self.plan_success = False
            return
        
        if save_freq != None:
            self._take_picture()

        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            for j in range(len(self.right_arm_joint_id)):
                n_j = self.right_arm_joint_id[j]
                self.active_joints[n_j].set_drive_target(result["position"][i][j])
                self.active_joints[n_j].set_drive_velocity_target(
                    result["velocity"][i][j]
                )
            
            self.scene.step()
            if i % 5 == 0:
                self._update_render()
                if self.render_freq and i % self.render_freq == 0:
                    self.viewer.render()

            if save_freq != None and i % save_freq == 0:
                self._take_picture()

        if save_freq != None:
            self._take_picture()

    
    def together_follow_path(self, left_result,right_result, save_freq=-1):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        left_n_step = left_result["position"].shape[0]
        right_n_step = right_result["position"].shape[0]
        n_step = max(left_n_step, right_n_step)

        if n_step > 2000:
            self.plan_success = False
            return

        if save_freq != None:
            self._take_picture()

        now_left_id = 0
        now_right_id = 0
        i = 0

        while now_left_id < left_n_step or now_right_id < right_n_step:
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            # set the joint positions and velocities for move group joints only.
            # The others are not the responsibility of the planner
            if now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
                for j in range(len(self.left_arm_joint_id)):
                    left_j = self.left_arm_joint_id[j]
                    self.active_joints[left_j].set_drive_target(left_result["position"][now_left_id][j])
                    self.active_joints[left_j].set_drive_velocity_target(left_result["velocity"][now_left_id][j])
                now_left_id +=1
                
            if now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
                for j in range(len(self.right_arm_joint_id)):
                    right_j = self.right_arm_joint_id[j]
                    self.active_joints[right_j].set_drive_target(right_result["position"][now_right_id][j])
                    self.active_joints[right_j].set_drive_velocity_target(right_result["velocity"][now_right_id][j])
                now_right_id +=1

            self.scene.step()
            if i % 5==0:
                self._update_render()
                if self.render_freq and i % self.render_freq == 0:
                    self.viewer.render()

            if save_freq != None and i % save_freq == 0:
                self._take_picture()
            i+=1

        if save_freq != None:
            self._take_picture()
            

    def set_gripper(self, left_pos = 0.045, right_pos = 0.045, set_tag = 'together', save_freq=-1):
        '''
            Set gripper posture
            - `left_pos`: Left gripper pose
            - `right_pos`: Right gripper pose
            - `set_tag`: "left" to set the left gripper, "right" to set the right gripper, "together" to set both grippers simultaneously.
        '''
        save_freq = self.save_freq if save_freq == -1 else save_freq
        if save_freq != None:
            self._take_picture()
        
        left_gripper_step = 0
        right_gripper_step = 0
        real_left_gripper_step = 0
        real_right_gripper_step = 0

        if set_tag == 'left' or set_tag == 'together':
            left_gripper_step = (left_pos - self.left_gripper_val) / 400
            real_left_gripper_step = (left_pos - self.active_joints[34].get_drive_target()[0]) / 200

        if set_tag == 'right' or set_tag == 'together':
            right_gripper_step = (right_pos - self.right_gripper_val) / 400
            real_right_gripper_step = (right_pos - self.active_joints[36].get_drive_target()[0]) / 200
        
        for i in range(400):
            self.left_gripper_val +=  left_gripper_step
            self.right_gripper_val +=  right_gripper_step
            if i < 200:
                real_left_gripper_val = self.active_joints[34].get_drive_target()[0] + real_left_gripper_step
                real_right_gripper_val = self.active_joints[36].get_drive_target()[0] + real_right_gripper_step

            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            
            if set_tag == 'left' or set_tag == 'together':
                for joint in self.active_joints[34:36]:
                    joint.set_drive_target(real_left_gripper_val)
                    joint.set_drive_velocity_target(0.05)

            if set_tag == 'right' or set_tag == 'together':
                for joint in self.active_joints[36:38]:
                    joint.set_drive_target(real_right_gripper_val)
                    joint.set_drive_velocity_target(0.05)

            self.scene.step()
            if i % 5==0:
                self._update_render()
                if self.render_freq and i % self.render_freq == 0:
                    self.viewer.render()

            if save_freq != None and i % save_freq == 0:
                self._take_picture()

        if save_freq != None:
            self._take_picture()
        
        if set_tag == 'left' or set_tag == 'together':
            self.left_gripper_val = left_pos
        if set_tag == 'right' or set_tag == 'together':
            self.right_gripper_val = right_pos

    def gripper_move_back(self, endpose_tag: str,save_freq=-1):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        if endpose_tag == 'left':
            pose_of_close_gripper = self.pose_of_close_left_gripper
            now_pose = self.now_left_pose
            pre_pose = self.pre_left_pose
            move_function = self.left_move_to_pose_with_screw
        elif endpose_tag == 'right':
            pose_of_close_gripper = self.pose_of_close_right_gripper
            now_pose = self.now_right_pose
            pre_pose = self.pre_right_pose
            move_function = self.right_move_to_pose_with_screw
        else:
            return
        
        if pose_of_close_gripper is not None and pre_pose is not None and now_pose is not None:
            las_pose_matrix = np.eye(4)
            las_pose_matrix[:3,3] = pose_of_close_gripper[:3]
            las_pose_matrix[:3,:3] = t3d.quaternions.quat2mat(pose_of_close_gripper[3:])

            now_pose_matrix = np.eye(4)
            now_pose_matrix[:3,3] = now_pose[:3]
            now_pose_matrix[:3,:3] = t3d.quaternions.quat2mat(now_pose[3:])

            trans_matrix = np.dot(now_pose_matrix, np.linalg.inv(las_pose_matrix))
            
            las_pre_pose_matrix = np.eye(4)
            las_pre_pose_matrix[:3,3] = pre_pose[:3]
            las_pre_pose_matrix[:3,:3] = t3d.quaternions.quat2mat(pre_pose[3:])
            
            nxt_pose_matrix = np.dot(trans_matrix, las_pre_pose_matrix)

            xyz_dis = np.ones(4)
            xyz_dis[:3] = np.array(pre_pose[:3]) - np.array(pose_of_close_gripper[:3])
            xyz_dis[:3] = trans_matrix[:3,:3] @ xyz_dis[:3]
            nxt_pose_matrix[:3,3] = now_pose[:3] + xyz_dis[:3]
            nxt_pose = list(nxt_pose_matrix[:3,3]) + list(t3d.quaternions.mat2quat(nxt_pose_matrix[:3,:3]))
            if self.is_plan_success(endpose_tag, nxt_pose) == False:
                return
            move_function(nxt_pose, save_freq=save_freq)

    def open_left_gripper(self, save_freq=-1, pos = 0.045):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        self.set_gripper(left_pos = pos, set_tag='left', save_freq=save_freq)
        if self.left_prepare_grasp_point_group is not None:
            self.left_prepare_grasp_data['contact_points_mask'][self.left_prepare_grasp_point_group] = True
            self.left_prepare_grasp_data = None
            self.left_prepare_grasp_point_group = None
        self.gripper_move_back('left',save_freq=save_freq)

    def close_left_gripper(self, save_freq=-1, pos = 0):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        self.set_gripper(left_pos = pos, set_tag='left',save_freq=save_freq)
        if self.left_prepare_grasp_point_group is not None:
            self.left_prepare_grasp_data['contact_points_mask'][self.left_prepare_grasp_point_group] = False
        self.pose_of_close_left_gripper = self.now_left_pose

    def open_right_gripper(self, save_freq=-1,pos = 0.045):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        self.set_gripper(right_pos=pos, set_tag='right', save_freq=save_freq)
        if self.right_prepare_grasp_point_group is not None:
            self.right_prepare_grasp_data['contact_points_mask'][self.right_prepare_grasp_point_group] = True
            self.right_prepare_grasp_data = None
            self.right_prepare_grasp_point_group = None
        self.gripper_move_back('right',save_freq=save_freq)

    def close_right_gripper(self, save_freq=-1,pos = 0):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        self.set_gripper(right_pos=pos, set_tag='right', save_freq=save_freq)
        if self.right_prepare_grasp_point_group is not None:
            self.right_prepare_grasp_data['contact_points_mask'][self.right_prepare_grasp_point_group] = False
        self.pose_of_close_right_gripper = self.now_right_pose

    def together_open_gripper(self, save_freq=-1, left_pos = 0.045, right_pos = 0.045):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        self.set_gripper(left_pos=left_pos, right_pos=right_pos, set_tag='together', save_freq=save_freq)
        self.gripper_move_back('left',save_freq=save_freq)
        self.gripper_move_back('right',save_freq=save_freq)

    def together_close_gripper(self, save_freq=-1,left_pos = 0, right_pos = 0):
        save_freq = self.save_freq if save_freq == -1 else save_freq
        self.set_gripper(left_pos=left_pos, right_pos=right_pos, set_tag='together', save_freq=save_freq)
        self.pose_of_close_left_gripper = self.now_left_pose
        self.pose_of_close_right_gripper = self.now_right_pose
        
    def move_to_pose_with_RRTConnect(
        self, pose, use_point_cloud=False, use_attach=False,freq =10
    ):
        """
        Plan and follow a path to a pose using RRTConnect

        Args:
            pose: [x, y, z, qx, qy, qz, qw]
            use_point_cloud (optional): if to take the point cloud into consideration
                for collision checking.
            use_attach (optional): if to take the attach into consideration
                for collision checking.
        """
        # result is a dictionary with keys 'status', 'time', 'position', 'velocity',
        # 'acceleration', 'duration'
        a = self.robot.get_qpos()
        qpos = np.array([a[6],a[14],a[18],a[22],a[26],a[30]])
        
        result = self.left_planner.plan_qpos_to_pose(
            pose,
            current_qpos = self.robot.get_qpos(),
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
            # mask = [0],
            planner_name="RRTConnect",
        )
        # plan_qpos_to_pose ankor end
        if result["status"] != "Success":
            # print(result["status"])
            return -1
        # do nothing if the planning fails; follow the path if the planning succeeds
        self.left_follow_path(result,freq)
        return 0

    def left_move_to_pose_with_screw(self, pose, use_point_cloud=False, use_attach=False,save_freq=-1):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        save_freq = self.save_freq if save_freq == -1 else save_freq
        if type(pose) != list or len(pose) != 7:
            print("left arm pose error!")
            return
        if self.is_left_gripper_open():
            self.pre_left_pose = self.now_left_pose
        self.now_left_pose = pose
        joint_pose = self.robot.get_qpos()
        qpos=[]
        for i in range(6):
            qpos.append(joint_pose[self.left_arm_joint_id[i]])

        result = self.left_planner.plan_screw(
            target_pose=pose,
            qpos=qpos,
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
        )
        
        if result["status"] == "Success":
            self.left_follow_path(result,save_freq=save_freq)
            return 0
        else:
            print("\n left arm palnning failed!")
            self.left_plan_success = False
            self.plan_success = False

    def right_move_to_pose_with_screw(self, pose, use_point_cloud=False, use_attach=False,save_freq=-1):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        save_freq = self.save_freq if save_freq == -1 else save_freq
        if type(pose) != list or len(pose) != 7:
            print("right arm pose error!")
            return
        
        if self.is_right_gripper_open():
            self.pre_right_pose = self.now_right_pose
        self.now_right_pose = pose
        joint_pose = self.robot.get_qpos()
        qpos=[]
        for i in range(6):
            qpos.append(joint_pose[self.right_arm_joint_id[i]])
        result = self.right_planner.plan_screw(
            target_pose=pose,
            qpos=qpos,
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
        )
        
        if result["status"] == "Success":
            self.right_follow_path(result,save_freq=save_freq)
            return 0
        else:
            print("\n right arm palnning failed!")
            self.right_plan_success = False
            self.plan_success = False
        

    def together_move_to_pose_with_screw(self, left_target_pose,right_target_pose, use_point_cloud=False, use_attach=False,save_freq=-1):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        save_freq = self.save_freq if save_freq == -1 else save_freq
        if type(left_target_pose) != list or len(left_target_pose) != 7:
            print("left arm pose error!")
            return
        if type(right_target_pose) != list or len(right_target_pose) != 7:
            print("right arm pose error!")
            return
        if self.is_left_gripper_open():
            self.pre_left_pose = self.now_left_pose
        if self.is_right_gripper_open():
            self.now_left_pose = left_target_pose
        self.pre_right_pose = self.now_right_pose
        self.now_right_pose = right_target_pose
        joint_pose = self.robot.get_qpos()
        left_qpos=[]
        right_qpos=[]
        for i in range(6):
            left_qpos.append(joint_pose[self.left_arm_joint_id[i]])
            right_qpos.append(joint_pose[self.right_arm_joint_id[i]])

        left_result = self.left_planner.plan_screw(
            target_pose=left_target_pose,
            qpos=left_qpos,
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
        )

        right_result = self.right_planner.plan_screw(
            target_pose=right_target_pose,
            qpos=right_qpos,
            time_step=1 / 250,
            use_point_cloud=use_point_cloud,
            use_attach=use_attach,
        )

        if left_result["status"] == "Success" and right_result["status"] == "Success":
            self.together_follow_path(left_result,right_result,save_freq=save_freq)
            return 0
        else:
            if left_result["status"] != "Success":
                print("\n left arm palnning failed!")
                self.left_plan_success = False
            if right_result["status"] != "Success":
                print("\n right arm palnning failed!")
                self.right_plan_success = False
            self.plan_success = False

    # Get Camera RGBA
    def _get_camera_rgba(self, camera):
        rgba = camera.get_picture("Color")
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        return rgba_img
    
    # Get Camera Segmentation
    def _get_camera_segmentation(self, camera,level = "mesh"):
        # visual_id is the unique id of each visual shape
        seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
        colormap = sorted(set(ImageColor.colormap.values()))
        color_palette = np.array(
            [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
        )
        if level == "mesh":
            label0_image = seg_labels[..., 0].astype(np.uint8) # mesh-level
        elif level == "actor":
            label0_image = seg_labels[..., 1].astype(np.uint8) # actor-level
        return color_palette[label0_image]
    
    # Get Camera Depth
    def _get_camera_depth(self, camera):
        position = camera.get_picture("Position")
        depth = -position[..., 2]
        depth_image = (depth * 1000.0).astype(np.float64)
        return depth_image
    
    # Get Camera PointCloud
    def _get_camera_pcd(self, camera, point_num = 0):
        rgba = camera.get_picture_cuda("Color").torch() # [H, W, 4]
        position = camera.get_picture_cuda("Position").torch()
        model_matrix = camera.get_model_matrix()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

        # Extract valid three-dimensional points and corresponding color data.
        valid_mask = position[..., 3] < 1
        points_opengl = position[..., :3][valid_mask]
        points_color = rgba[valid_mask][:,:3]
        # Transform into the world coordinate system.
        points_world = torch.bmm(points_opengl.view(1, -1, 3), model_matrix[:3, :3].transpose(0,1).view(-1, 3, 3)).squeeze(1) + model_matrix[:3, 3]

        # Format color data.
        points_color = torch.clamp(points_color, 0, 1)

        points_world = points_world.squeeze(0)
        
        # If crop is needed
        if self.pcd_crop:
            min_bound = torch.tensor(self.pcd_crop_bbox[0], dtype=torch.float32).to(device)
            max_bound = torch.tensor(self.pcd_crop_bbox[1], dtype=torch.float32).to(device)
            inside_bounds_mask = (points_world.squeeze(0) >= min_bound).all(dim=1) & (points_world.squeeze(0)  <= max_bound).all(dim=1)
            points_world = points_world[inside_bounds_mask]
            points_color = points_color[inside_bounds_mask]
        
        # Convert the tensor back to a NumPy array for use with Open3D.
        points_world_np = points_world.cpu().numpy()
        points_color_np = points_color.cpu().numpy()

        if point_num > 0:
            points_world_np,index = fps(points_world_np,point_num)
            index = index.detach().cpu().numpy()[0]
            points_color_np = points_color_np[index,:]

        return np.hstack((points_world_np, points_color_np))

    def arr2pcd(self,point_arr,colors_arr = None):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_arr)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_arr)
        return point_cloud
        
    def get_left_arm_jointState(self) -> list:
        jointState_list = []
        for id in self.left_arm_joint_id:
            jointState_list.append(self.active_joints[id].get_drive_target()[0].astype(float))
        jointState_list.append(self.left_gripper_val)
        return jointState_list

    def get_right_arm_jointState(self) -> list:
        jointState_list = []
        for id in self.right_arm_joint_id:
            jointState_list.append(self.active_joints[id].get_drive_target()[0].astype(float))
        jointState_list.append(self.right_gripper_val)
        return jointState_list
    
    def endpose_transform(self, joint, gripper_val):
        rpy = joint.global_pose.get_rpy()
        roll, pitch, yaw = rpy
        x,y,z = joint.global_pose.p
        endpose = {
            "gripper": float(gripper_val),
            "pitch" : float(pitch),
            "roll" : float(roll),
            "x": float(x),
            "y": float(y),
            "yaw" : float(yaw),
            "z": float(z),
        }
        return endpose
    
    def get_camera_config(self,camera: sapien.render.RenderCameraComponent):
        config = {
                    "D" : [ 0, 0, 0, 0, 0 ],
                    "K" : camera.get_intrinsic_matrix().ravel().tolist(),
                    "P" : np.vstack((camera.get_intrinsic_matrix(), [0,0,0])).ravel().tolist(),
                    "R" : [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ],
                    "binning_x" : 0,
                    "binning_y" : 0,
                    "distortion_model" : "plumb_bob",
                    "height" : camera.get_height(),
                    "parent_frame" : 
                    {
                        "pitch" : 0,
                        "roll" : 1.57,
                        "x" : 0,
                        "y" : 0,
                        "yaw" : 1.57,
                        "z" : 0
                    },
                    "roi" : 
                    {
                        "do_rectify" : 0,
                        "height" : 0,
                        "width" : 0,
                        "x_offset" : 0,
                        "y_offset" : 0
                    },
                    "width" : camera.get_width()
                }
        return config
    
    def is_left_gripper_open(self):
        return self.active_joints[34].get_drive_target()[0] > 0.04
    def is_right_gripper_open(self):
        return self.active_joints[36].get_drive_target()[0] > 0.04
    def is_left_gripper_open_half(self):
        return self.active_joints[34].get_drive_target()[0] > 0.02
    def is_right_gripper_open_half(self):
        return self.active_joints[36].get_drive_target()[0] > 0.02
    def is_left_gripper_close(self):
        return self.active_joints[34].get_drive_target()[0] < 0.01
    def is_right_gripper_close(self):
        return self.active_joints[34].get_drive_target()[0] < 0.01
    
    def get_left_endpose_pose(self):
        return self.left_endpose.global_pose
    def get_right_endpose_pose(self):
        return self.right_endpose.global_pose

    def _take_picture(self): # Save data
        if not self.is_save:
            return

        print('saving: episode = ', self.ep_num, ' index = ',self.PCD_INDEX, end='\r')
        self._update_render()
        self.left_camera.take_picture()
        self.right_camera.take_picture()
        self.head_camera.take_picture()
        self.observer_camera.take_picture()
        self.front_camera.take_picture()
        
        if self.PCD_INDEX==0:
            self.file_path ={
                "observer_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/observer/",

                "l_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/left/",
                "l_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/left/",
                "l_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/left/",

                "f_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/front/",
                "f_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/front/",
                "f_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/front/",

                "r_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/right/",
                "r_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/right/",
                "r_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/right/",

                "t_color" : f"{self.save_dir}/episode{self.ep_num}/camera/color/head/",
                "t_depth" : f"{self.save_dir}/episode{self.ep_num}/camera/depth/head/",
                "t_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/head/",

                "f_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/front/mesh/",
                "l_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/left/mesh/",
                "r_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/right/mesh/",
                "t_seg_mesh" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/top/mesh/",

                "f_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/front/actor/",
                "l_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/left/actor/",
                "r_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/right/actor/",
                "t_seg_actor" : f"{self.save_dir}/episode{self.ep_num}/camera/segmentation/head/actor/",

                "f_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/front/",
                "t_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/head/",
                "l_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/left/",
                "r_camera" : f"{self.save_dir}/episode{self.ep_num}/camera/model_camera/right/",

                "ml_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/masterLeft/",
                "mr_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/masterRight/",
                "pl_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/puppetLeft/",
                "pr_ep" : f"{self.save_dir}/episode{self.ep_num}/arm/endPose/puppetRight/",
                "pl_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/puppetLeft/",
                "pr_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/puppetRight/",
                "ml_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/masterLeft/",
                "mr_joint" : f"{self.save_dir}/episode{self.ep_num}/arm/jointState/masterRight/",
                "pkl" : f"{self.save_dir}_pkl/episode{self.ep_num}/",
                "conbine_pcd" : f"{self.save_dir}/episode{self.ep_num}/camera/pointCloud/conbine/",
            }

            for directory in self.file_path.values():
                if os.path.exists(directory):
                    file_list = os.listdir(directory)
                    for file in file_list:
                        os.remove(directory + file)

        pkl_dic = {
            "observation":{
                "head_camera":{},   # rbg , mesh_seg , actior_seg , depth , intrinsic_cv , extrinsic_cv , cam2world_gl(model_matrix)
                "left_camera":{},
                "right_camera":{},
                "front_camera":{}
            },
            "pointcloud":[],   # conbinet pcd
            "joint_action":[],
            "endpose":[]
        }
        
        head_camera_intrinsic_cv = self.head_camera.get_intrinsic_matrix()
        head_camera_extrinsic_cv = self.head_camera.get_extrinsic_matrix()
        head_camera_model_matrix = self.head_camera.get_model_matrix()

        pkl_dic["observation"]["head_camera"] = {
            "intrinsic_cv" : head_camera_intrinsic_cv,
            "extrinsic_cv" : head_camera_extrinsic_cv,
            "cam2world_gl" : head_camera_model_matrix
        }

        front_camera_intrinsic_cv = self.front_camera.get_intrinsic_matrix()
        front_camera_extrinsic_cv = self.front_camera.get_extrinsic_matrix()
        front_camera_model_matrix = self.front_camera.get_model_matrix()

        pkl_dic["observation"]["front_camera"] = {
            "intrinsic_cv" : front_camera_intrinsic_cv,
            "extrinsic_cv" : front_camera_extrinsic_cv,
            "cam2world_gl" : front_camera_model_matrix
        }

        left_camera_intrinsic_cv = self.left_camera.get_intrinsic_matrix()
        left_camera_extrinsic_cv = self.left_camera.get_extrinsic_matrix()
        left_camera_model_matrix = self.left_camera.get_model_matrix()

        pkl_dic["observation"]["left_camera"] = {
            "intrinsic_cv" : left_camera_intrinsic_cv,
            "extrinsic_cv" : left_camera_extrinsic_cv,
            "cam2world_gl" : left_camera_model_matrix
        }

        right_camera_intrinsic_cv = self.right_camera.get_intrinsic_matrix()
        right_camera_extrinsic_cv = self.right_camera.get_extrinsic_matrix()
        right_camera_model_matrix = self.right_camera.get_model_matrix()

        pkl_dic["observation"]["right_camera"] = {
            "intrinsic_cv" : right_camera_intrinsic_cv,
            "extrinsic_cv" : right_camera_extrinsic_cv,
            "cam2world_gl" : right_camera_model_matrix
        }

        # # ---------------------------------------------------------------------------- #
        # # RGBA
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('rgb', False):
            front_rgba = self._get_camera_rgba(self.front_camera)
            head_rgba = self._get_camera_rgba(self.head_camera)
            left_rgba = self._get_camera_rgba(self.left_camera)
            right_rgba = self._get_camera_rgba(self.right_camera)

            if self.save_type.get('raw_data', True):
                if self.data_type.get('observer', False):
                    observer_rgba = self._get_camera_rgba(self.observer_camera)
                    save_img(self.file_path["observer_color"]+f"{self.PCD_INDEX}.png",observer_rgba)
                save_img(self.file_path["t_color"]+f"{self.PCD_INDEX}.png",head_rgba)
                save_img(self.file_path["f_color"]+f"{self.PCD_INDEX}.png",front_rgba)
                save_img(self.file_path["l_color"]+f"{self.PCD_INDEX}.png",left_rgba)
                save_img(self.file_path["r_color"]+f"{self.PCD_INDEX}.png",right_rgba)

            if self.save_type.get('pkl' , True):
                pkl_dic["observation"]["head_camera"]["rgb"] = head_rgba[:,:,:3]
                pkl_dic["observation"]["front_camera"]["rgb"] = front_rgba[:,:,:3]
                pkl_dic["observation"]["left_camera"]["rgb"] = left_rgba[:,:,:3]
                pkl_dic["observation"]["right_camera"]["rgb"] = right_rgba[:,:,:3]
        # # ---------------------------------------------------------------------------- #
        # # mesh_segmentation
        # # ---------------------------------------------------------------------------- # 
        if self.data_type.get('mesh_segmentation', False):
            head_seg = self._get_camera_segmentation(self.head_camera,level="mesh")
            left_seg = self._get_camera_segmentation(self.left_camera,level="mesh")
            right_seg = self._get_camera_segmentation(self.right_camera,level="mesh")
            front_seg = self._get_camera_segmentation(self.front_camera,level="mesh")

            if self.save_type.get('raw_data', True):
                save_img(self.file_path["t_seg_mesh"]+f"{self.PCD_INDEX}.png", head_seg)
                save_img(self.file_path["l_seg_mesh"]+f"{self.PCD_INDEX}.png", left_seg)
                save_img(self.file_path["r_seg_mesh"]+f"{self.PCD_INDEX}.png", right_seg)
                save_img(self.file_path["f_seg_mesh"]+f"{self.PCD_INDEX}.png", front_seg)

            if self.save_type.get('pkl' , True):
                pkl_dic["observation"]["head_camera"]["mesh_segmentation"] = head_seg
                pkl_dic["observation"]["front_camera"]["mesh_segmentation"] = front_seg
                pkl_dic["observation"]["left_camera"]["mesh_segmentation"] = left_seg
                pkl_dic["observation"]["right_camera"]["mesh_segmentation"] = right_seg
        # # ---------------------------------------------------------------------------- #
        # # actor_segmentation
        # # --------------------------------------------------------------------------- # 
        if self.data_type.get('actor_segmentation', False):
            head_seg = self._get_camera_segmentation(self.head_camera,level="actor")
            left_seg = self._get_camera_segmentation(self.left_camera,level="actor")
            right_seg = self._get_camera_segmentation(self.right_camera,level="actor")
            front_seg = self._get_camera_segmentation(self.front_camera,level="actor")

            if self.save_type.get('raw_data', True):
                save_img(self.file_path["t_seg_actor"]+f"{self.PCD_INDEX}.png", head_seg)
                save_img(self.file_path["l_seg_actor"]+f"{self.PCD_INDEX}.png", left_seg)
                save_img(self.file_path["r_seg_actor"]+f"{self.PCD_INDEX}.png", right_seg)
                save_img(self.file_path["f_seg_actor"]+f"{self.PCD_INDEX}.png", front_seg)
            if self.save_type.get('pkl' , True):
                pkl_dic["observation"]["head_camera"]["actor_segmentation"] = head_seg
                pkl_dic["observation"]["left_camera"]["actor_segmentation"] = left_seg
                pkl_dic["observation"]["right_camera"]["actor_segmentation"] = right_seg
                pkl_dic["observation"]["front_camera"]["actor_segmentation"] = front_seg
        # # ---------------------------------------------------------------------------- #
        # # DEPTH
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('depth', False):
            front_depth = self._get_camera_depth(self.front_camera)
            head_depth = self._get_camera_depth(self.head_camera)
            left_depth = self._get_camera_depth(self.left_camera)
            right_depth = self._get_camera_depth(self.right_camera)
            
            if self.save_type.get('raw_data', True):
                save_img(self.file_path["t_depth"]+f"{self.PCD_INDEX}.png", head_depth.astype(np.uint16))
                save_img(self.file_path["f_depth"]+f"{self.PCD_INDEX}.png", front_depth.astype(np.uint16))
                save_img(self.file_path["l_depth"]+f"{self.PCD_INDEX}.png", left_depth.astype(np.uint16))
                save_img(self.file_path["r_depth"]+f"{self.PCD_INDEX}.png", right_depth.astype(np.uint16))
            if self.save_type.get('pkl' , True):
                pkl_dic["observation"]["head_camera"]["depth"] = head_depth
                pkl_dic["observation"]["front_camera"]["depth"] = front_depth
                pkl_dic["observation"]["left_camera"]["depth"] = left_depth
                pkl_dic["observation"]["right_camera"]["depth"] = right_depth
        # # ---------------------------------------------------------------------------- #
        # # endpose JSON
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('endpose', False):
            left_endpose = self.endpose_transform(self.all_joints[42], self.left_gripper_val)
            right_endpose = self.endpose_transform(self.all_joints[43], self.right_gripper_val)

            if self.save_type.get('raw_data', True):
                save_json(self.file_path["ml_ep"]+f"{self.PCD_INDEX}.json", left_endpose)
                save_json(self.file_path["pl_ep"]+f"{self.PCD_INDEX}.json", left_endpose)
                save_json(self.file_path["mr_ep"]+f"{self.PCD_INDEX}.json", right_endpose)
                save_json(self.file_path["pr_ep"]+f"{self.PCD_INDEX}.json", right_endpose)

            if self.save_type.get('pkl' , True):
                if self.dual_arm:
                    pkl_dic["endpose"] = np.array([left_endpose["x"],left_endpose["y"],left_endpose["z"],left_endpose["roll"],
                                                left_endpose["pitch"],left_endpose["yaw"],left_endpose["gripper"],
                                                right_endpose["x"],right_endpose["y"],right_endpose["z"],right_endpose["roll"],
                                                right_endpose["pitch"],right_endpose["yaw"],right_endpose["gripper"],])
                else:
                    pkl_dic["endpose"] = np.array([right_endpose["x"],right_endpose["y"],right_endpose["z"],right_endpose["roll"],
                                                    right_endpose["pitch"],right_endpose["yaw"],right_endpose["gripper"],])
        # # ---------------------------------------------------------------------------- #
        # # JointState JSON
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('qpos', False):
            left_jointstate = {
                "effort" : [ 0, 0, 0, 0, 0, 0, 0 ],
                "position" : self.get_left_arm_jointState(),
                "velocity" : [ 0, 0, 0, 0, 0, 0, 0 ]
            }
            right_jointstate = {
                "effort" : [ 0, 0, 0, 0, 0, 0, 0 ],
                "position" : self.get_right_arm_jointState(),
                "velocity" : [ 0, 0, 0, 0, 0, 0, 0 ]
            }

            if self.save_type.get('raw_data', True):
                save_json(self.file_path["ml_joint"]+f"{self.PCD_INDEX}.json", left_jointstate)
                save_json(self.file_path["pl_joint"]+f"{self.PCD_INDEX}.json", left_jointstate)
                save_json(self.file_path["mr_joint"]+f"{self.PCD_INDEX}.json", right_jointstate)
                save_json(self.file_path["pr_joint"]+f"{self.PCD_INDEX}.json", right_jointstate)

            if self.save_type.get('pkl' , True):
                if self.dual_arm:
                    pkl_dic["joint_action"] = np.array(left_jointstate["position"]+right_jointstate["position"])
                else:
                    pkl_dic["joint_action"] = np.array(right_jointstate["position"])
        # # ---------------------------------------------------------------------------- #
        # # PointCloud
        # # ---------------------------------------------------------------------------- #
        if self.data_type.get('pointcloud', False):
            head_pcd = self._get_camera_pcd(self.head_camera, point_num=0)
            front_pcd = self._get_camera_pcd(self.front_camera, point_num=0)
            left_pcd = self._get_camera_pcd(self.left_camera, point_num=0)
            right_pcd = self._get_camera_pcd(self.right_camera, point_num=0) 

            # Merge pointcloud
            if self.data_type.get("conbine", False):
                conbine_pcd = np.vstack((head_pcd , left_pcd , right_pcd, front_pcd))
            else:
                conbine_pcd = head_pcd
            
            pcd_array,index = conbine_pcd[:,:3], np.array(range(len(conbine_pcd)))
            if self.pcd_down_sample_num > 0:
                pcd_array,index = fps(conbine_pcd[:,:3],self.pcd_down_sample_num)
                index = index.detach().cpu().numpy()[0]

            if self.save_type.get('raw_data', True):
                ensure_dir(self.file_path["t_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["t_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(head_pcd[:,:3], head_pcd[:,3:])) 
                ensure_dir(self.file_path["l_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["l_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(left_pcd[:,:3], left_pcd[:,3:]))
                ensure_dir(self.file_path["r_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["r_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(right_pcd[:,:3], right_pcd[:,3:]))
                ensure_dir(self.file_path["f_pcd"] + f"{self.PCD_INDEX}.pcd")
                o3d.io.write_point_cloud(self.file_path["f_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(front_pcd[:,:3], front_pcd[:,3:]))
                if self.data_type.get("conbine", False):
                    ensure_dir(self.file_path["conbine_pcd"] + f"{self.PCD_INDEX}.pcd")
                    o3d.io.write_point_cloud(self.file_path["conbine_pcd"] + f"{self.PCD_INDEX}.pcd", self.arr2pcd(pcd_array, conbine_pcd[index,3:]))

            if self.save_type.get('pkl' , True):
                pkl_dic["pointcloud"] = conbine_pcd[index]
        #===========================================================#
        if self.save_type.get('pkl' , True):
            save_pkl(self.file_path["pkl"]+f"{self.PCD_INDEX}.pkl", pkl_dic)

        self.PCD_INDEX +=1
    
    def get_obs(self):
        self.scene.step()
        self._update_render()
        self._update_render()
        obs = collections.OrderedDict()
        
        left_endpose = self.endpose_transform(self.all_joints[42], self.left_gripper_val)
        right_endpose = self.endpose_transform(self.all_joints[43], self.right_gripper_val)

        right_jointState = self.get_right_arm_jointState()
        right_jointState_array = np.array(right_jointState)

        left_jointState = self.get_left_arm_jointState()
        left_jointState_array = np.array(left_jointState)

        self.left_camera.take_picture()
        self.right_camera.take_picture()
        self.head_camera.take_picture()
        self.front_camera.take_picture()

        head_pcd = self._get_camera_pcd(self.head_camera, point_num=0)
        left_pcd = self._get_camera_pcd(self.left_camera, point_num=0)
        right_pcd = self._get_camera_pcd(self.right_camera, point_num=0)
        front_pcd = self._get_camera_pcd(self.front_camera, point_num=0)
        head_rgba = self._get_camera_rgba(self.head_camera)
        left_rgba = self._get_camera_rgba(self.left_camera)
        right_rgba = self._get_camera_rgba(self.right_camera)
        front_rgba = self._get_camera_rgba(self.front_camera)
        head_depth = self._get_camera_depth(self.head_camera)
        left_depth = self._get_camera_depth(self.left_camera)
        right_depth = self._get_camera_depth(self.right_camera)
        front_depth = self._get_camera_depth(self.front_camera)

        # Merge PointCloud
        if self.data_type.get("conbine", False):
            conbine_pcd = np.vstack((head_pcd , left_pcd , right_pcd, front_pcd))
        else:
            conbine_pcd = head_pcd
        pcd_array, index = fps(conbine_pcd[:,:3],self.pcd_down_sample_num)

        obs = {
            "observation":{
                "head_camera":{},   # rbg , mesh_seg , actior_seg , depth , intrinsic_cv , extrinsic_cv , cam2world_gl(model_matrix)
                "left_camera":{},
                "right_camera":{},
                "front_camera":{}
            },
            "pointcloud":[],   # conbinet pcd
            "joint_action":[],
            "endpose":[]
        }
        
        head_camera_intrinsic_cv = self.head_camera.get_intrinsic_matrix()
        head_camera_extrinsic_cv = self.head_camera.get_extrinsic_matrix()
        head_camera_model_matrix = self.head_camera.get_model_matrix()

        obs["observation"]["head_camera"] = {
            "intrinsic_cv" : head_camera_intrinsic_cv,
            "extrinsic_cv" : head_camera_extrinsic_cv,
            "cam2world_gl" : head_camera_model_matrix
        }

        front_camera_intrinsic_cv = self.front_camera.get_intrinsic_matrix()
        front_camera_extrinsic_cv = self.front_camera.get_extrinsic_matrix()
        front_camera_model_matrix = self.front_camera.get_model_matrix()

        obs["observation"]["front_camera"] = {
            "intrinsic_cv" : front_camera_intrinsic_cv,
            "extrinsic_cv" : front_camera_extrinsic_cv,
            "cam2world_gl" : front_camera_model_matrix
        }

        left_camera_intrinsic_cv = self.left_camera.get_intrinsic_matrix()
        left_camera_extrinsic_cv = self.left_camera.get_extrinsic_matrix()
        left_camera_model_matrix = self.left_camera.get_model_matrix()

        obs["observation"]["left_camera"] = {
            "intrinsic_cv" : left_camera_intrinsic_cv,
            "extrinsic_cv" : left_camera_extrinsic_cv,
            "cam2world_gl" : left_camera_model_matrix
        }

        right_camera_intrinsic_cv = self.right_camera.get_intrinsic_matrix()
        right_camera_extrinsic_cv = self.right_camera.get_extrinsic_matrix()
        right_camera_model_matrix = self.right_camera.get_model_matrix()

        obs["observation"]["right_camera"] = {
            "intrinsic_cv" : right_camera_intrinsic_cv,
            "extrinsic_cv" : right_camera_extrinsic_cv,
            "cam2world_gl" : right_camera_model_matrix
        }

        obs["observation"]["head_camera"]["rgb"] = head_rgba[:,:,:3]
        obs["observation"]["front_camera"]["rgb"] = front_rgba[:,:,:3]
        obs["observation"]["left_camera"]["rgb"] = left_rgba[:,:,:3]
        obs["observation"]["right_camera"]["rgb"] = right_rgba[:,:,:3]

        obs["observation"]["head_camera"]["depth"] = head_depth
        obs["observation"]["front_camera"]["depth"] = front_depth
        obs["observation"]["left_camera"]["depth"] = left_depth
        obs["observation"]["right_camera"]["depth"] = right_depth

        obs["pointcloud"] = conbine_pcd[index.detach().cpu().numpy()[0]]
        obs["endpose"] = np.array([left_endpose["x"],left_endpose["y"],left_endpose["z"],left_endpose["roll"],
                                    left_endpose["pitch"],left_endpose["yaw"],left_endpose["gripper"],
                                    right_endpose["x"],right_endpose["y"],right_endpose["z"],right_endpose["roll"],
                                    right_endpose["pitch"],right_endpose["yaw"],right_endpose["gripper"],])
        obs["joint_action"] = np.hstack((left_jointState_array, right_jointState_array))

        return obs
        
    def get_cam_obs(self, observation: dict) -> dict:
        head_cam = np.moveaxis(observation['observation']['head_camera']['rgb'], -1, 0) / 255
        front_cam = np.moveaxis(observation['observation']['front_camera']['rgb'], -1, 0) / 255
        left_cam = np.moveaxis(observation['observation']['left_camera']['rgb'], -1, 0) / 255
        right_cam = np.moveaxis(observation['observation']['right_camera']['rgb'], -1, 0) / 255
        return dict(
            head_cam = head_cam,
            front_cam = front_cam,
            left_cam = left_cam,
            right_cam = right_cam
        )

    def apply_dp(self, model, video_log=False, save_dir='default'):
        cnt = 0
        self.test_num += 1

        if video_log:
            import subprocess
            from pathlib import Path
            save_dir = Path('video') / save_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg = subprocess.Popen([
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pixel_format', 'rgb24',
                '-video_size', '320x240',
                '-framerate', '10',
                '-i', '-',
                '-pix_fmt', 'yuv420p',
                '-vcodec', 'libx264',
                '-crf', '23',
                f'{save_dir}/{self.test_num}.mp4'
            ], stdin=subprocess.PIPE)

        success_flag = False
        self._update_render()
        if self.render_freq:
            self.viewer.render()
        
        self.actor_pose = True

        observation = self.get_obs()
        obs = self.get_cam_obs(observation)

        obs['agent_pos'] = observation['joint_action']
        model.update_obs(obs)

        while cnt < self.step_lim:
            if video_log:
                ffmpeg.stdin.write(observation['head_camera'].tobytes())
            
            actions = model.get_action()
            obs = model.get_last_obs()
            left_arm_actions , left_gripper , left_current_qpos, left_path = [], [], [], []
            right_arm_actions , right_gripper , right_current_qpos, right_path = [], [], [], []
            if self.dual_arm:
                left_arm_actions,left_gripper = actions[:, :6],actions[:, 6]
                right_arm_actions,right_gripper = actions[:, 7:13],actions[:, 13]
                left_current_qpos, right_current_qpos = obs['agent_pos'][:6], obs['agent_pos'][7:13]
            else:
                right_arm_actions,right_gripper = actions[:, :6],actions[:, 6]
                right_current_qpos = obs['agent_pos'][:6]
            
            if self.dual_arm:
                left_path = np.vstack((left_current_qpos, left_arm_actions))
            right_path = np.vstack((right_current_qpos, right_arm_actions))


            topp_left_flag, topp_right_flag = True, True
            try:
                times, left_pos, left_vel, acc, duration = self.left_planner.TOPP(left_path, 1/250, verbose=True)
                left_result = dict()
                left_result['position'], left_result['velocity'] = left_pos, left_vel
                left_n_step = left_result["position"].shape[0]
                left_gripper = np.linspace(left_gripper[0], left_gripper[-1], left_n_step)
            except:
                topp_left_flag = False
                left_n_step = 1
            
            if left_n_step == 0 or (not self.dual_arm):
                topp_left_flag = False
                left_n_step = 1

            try:
                times, right_pos, right_vel, acc, duration = self.right_planner.TOPP(right_path, 1/250, verbose=True)            
                right_result = dict()
                right_result['position'], right_result['velocity'] = right_pos, right_vel
                right_n_step = right_result["position"].shape[0]
                right_gripper = np.linspace(right_gripper[0], right_gripper[-1], right_n_step)
            except:
                topp_right_flag = False
                right_n_step = 1
            
            if right_n_step == 0:
                topp_right_flag = False
                right_n_step = 1
            
            cnt += actions.shape[0]
            
            n_step = max(left_n_step, right_n_step)

            obs_update_freq = n_step // actions.shape[0]

            now_left_id = 0 if topp_left_flag else 1e9
            now_right_id = 0 if topp_right_flag else 1e9
            i = 0
            
            while now_left_id < left_n_step or now_right_id < right_n_step:
                qf = self.robot.compute_passive_force(
                    gravity=True, coriolis_and_centrifugal=True
                )
                self.robot.set_qf(qf)
                if topp_left_flag and now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
                    for j in range(len(self.left_arm_joint_id)):
                        left_j = self.left_arm_joint_id[j]
                        self.active_joints[left_j].set_drive_target(left_result["position"][now_left_id][j])
                        self.active_joints[left_j].set_drive_velocity_target(left_result["velocity"][now_left_id][j])
                    if not self.fix_gripper:
                        for joint in self.active_joints[34:36]:
                            # joint.set_drive_target(left_result["position"][i][6])
                            joint.set_drive_target(left_gripper[now_left_id])
                            joint.set_drive_velocity_target(0.05)
                            self.left_gripper_val = left_gripper[now_left_id]

                    now_left_id +=1
                    
                if topp_right_flag and now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
                    for j in range(len(self.right_arm_joint_id)):
                        right_j = self.right_arm_joint_id[j]
                        self.active_joints[right_j].set_drive_target(right_result["position"][now_right_id][j])
                        self.active_joints[right_j].set_drive_velocity_target(right_result["velocity"][now_right_id][j])
                    if not self.fix_gripper:
                        for joint in self.active_joints[36:38]:
                            # joint.set_drive_target(right_result["position"][i][6])
                            joint.set_drive_target(right_gripper[now_right_id])
                            joint.set_drive_velocity_target(0.05)
                            self.right_gripper_val = right_gripper[now_right_id]

                    now_right_id +=1
                
                self.scene.step()
                self._update_render()

                if i != 0 and i % obs_update_freq == 0:
                    observation = self.get_obs()
                    obs = self.get_cam_obs(observation)
                    obs['agent_pos'] = observation['joint_action']
                    
                    model.update_obs(obs)
                    self._take_picture()

                if i % 5==0:
                    self._update_render()
                    if self.render_freq and i % self.render_freq == 0:
                        self.viewer.render()
                
                i+=1
                if self.check_success():
                    success_flag = True
                    break

                if self.actor_pose == False:
                    break
            
            self. _update_render()

            if self.render_freq:
                self.viewer.render()
            
            self._take_picture()

            print(f'step: {cnt} / {self.step_lim}', end='\r')

            if success_flag:
                print("\nsuccess!")
                self.suc +=1
                if video_log:
                    ffmpeg.stdin.close()
                    ffmpeg.wait()
                    del ffmpeg

                return
            
            if self.actor_pose == False:
                break
            continue
        print("\nfail!")
        if video_log:
            ffmpeg.stdin.close()
            ffmpeg.wait()
            del ffmpeg

    def apply_dp3(self, model):
        cnt = 0
        self.test_num += 1

        success_flag = False
        self._update_render()
        if self.render_freq:
            self.viewer.render()
        
        self.actor_pose = True
        
        while cnt < self.step_lim:
            observation = self.get_obs()  
            obs = dict()
            obs['point_cloud'] = observation['pointcloud']
            if self.dual_arm:
                obs['agent_pos'] = observation['joint_action']
                assert obs['agent_pos'].shape[0] == 14, 'agent_pose shape, error'
            else:
                obs['agent_pos'] = observation['joint_action']
                assert obs['agent_pos'].shape[0] == 7, 'agent_pose shape, error'
            
            actions = model.get_action(obs)
            left_arm_actions , left_gripper , left_current_qpos, left_path = [], [], [], []
            right_arm_actions , right_gripper , right_current_qpos, right_path = [], [], [], []
            if self.dual_arm:
                left_arm_actions,left_gripper = actions[:, :6],actions[:, 6]
                right_arm_actions,right_gripper = actions[:, 7:13],actions[:, 13]
                left_current_qpos, right_current_qpos = obs['agent_pos'][:6], obs['agent_pos'][7:13]
            else:
                right_arm_actions,right_gripper = actions[:, :6],actions[:, 6]
                right_current_qpos = obs['agent_pos'][:6]
            
            if self.dual_arm:
                left_path = np.vstack((left_current_qpos, left_arm_actions))
            right_path = np.vstack((right_current_qpos, right_arm_actions))

            topp_left_flag, topp_right_flag = True, True
            try:
                times, left_pos, left_vel, acc, duration = self.left_planner.TOPP(left_path, 1/250, verbose=True)
                left_result = dict()
                left_result['position'], left_result['velocity'] = left_pos, left_vel
                left_n_step = left_result["position"].shape[0]
                left_gripper = np.linspace(left_gripper[0], left_gripper[-1], left_n_step)
            except:
                topp_left_flag = False
                left_n_step = 1
            
            if left_n_step == 0 or (not self.dual_arm):
                topp_left_flag = False
                left_n_step = 1

            try:
                times, right_pos, right_vel, acc, duration = self.right_planner.TOPP(right_path, 1/250, verbose=True)            
                right_result = dict()
                right_result['position'], right_result['velocity'] = right_pos, right_vel
                right_n_step = right_result["position"].shape[0]
                right_gripper = np.linspace(right_gripper[0], right_gripper[-1], right_n_step)
            except:
                topp_right_flag = False
                right_n_step = 1
            
            if right_n_step == 0:
                topp_right_flag = False
                right_n_step = 1
            
            cnt += actions.shape[0]
            
            n_step = max(left_n_step, right_n_step)

            obs_update_freq = n_step // actions.shape[0]

            now_left_id = 0 if topp_left_flag else 1e9
            now_right_id = 0 if topp_right_flag else 1e9
            i = 0
            
            while now_left_id < left_n_step or now_right_id < right_n_step:
                qf = self.robot.compute_passive_force(
                    gravity=True, coriolis_and_centrifugal=True
                )
                self.robot.set_qf(qf)
                if topp_left_flag and now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
                    for j in range(len(self.left_arm_joint_id)):
                        left_j = self.left_arm_joint_id[j]
                        self.active_joints[left_j].set_drive_target(left_result["position"][now_left_id][j])
                        self.active_joints[left_j].set_drive_velocity_target(left_result["velocity"][now_left_id][j])
                    if not self.fix_gripper:
                        for joint in self.active_joints[34:36]:
                            joint.set_drive_target(left_gripper[now_left_id])
                            joint.set_drive_velocity_target(0.05)
                            self.left_gripper_val = left_gripper[now_left_id]

                    now_left_id +=1
                    
                if topp_right_flag and now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
                    for j in range(len(self.right_arm_joint_id)):
                        right_j = self.right_arm_joint_id[j]
                        self.active_joints[right_j].set_drive_target(right_result["position"][now_right_id][j])
                        self.active_joints[right_j].set_drive_velocity_target(right_result["velocity"][now_right_id][j])
                    if not self.fix_gripper:
                        for joint in self.active_joints[36:38]:
                            joint.set_drive_target(right_gripper[now_right_id])
                            joint.set_drive_velocity_target(0.05)
                            self.right_gripper_val = right_gripper[now_right_id]

                    now_right_id +=1
                
                self.scene.step()
                self._update_render()

                if i != 0 and i % obs_update_freq == 0:
                    observation = self.get_obs()
                    obs=dict()
                    obs['point_cloud'] = observation['pointcloud']
                    if self.dual_arm:
                        obs['agent_pos'] = observation['joint_action']
                        assert obs['agent_pos'].shape[0] == 14, 'agent_pose shape, error'
                    else:
                        obs['agent_pos'] = observation['joint_action']
                        assert obs['agent_pos'].shape[0] == 7, 'agent_pose shape, error'
                    
                    model.update_obs(obs)
                    self._take_picture()

                if i % 5==0:
                    self._update_render()
                    if self.render_freq and i % self.render_freq == 0:
                        self.viewer.render()
                
                i+=1
                if self.check_success():
                    success_flag = True
                    break

                if self.actor_pose == False:
                    break
            
            self. _update_render()
            if self.render_freq:
                self.viewer.render()
            
            self._take_picture()

            print(f'step: {cnt} / {self.step_lim}', end='\r')

            if success_flag:
                print("\nsuccess!")
                self.suc +=1
                return
            
            if self.actor_pose == False:
                break
            
        print("\nfail!")

################################################# Generate Data API #################################################
    def load_actors(self, scene_info_file):
        """
            Scene generation function 
            - scene_info_file: Path to the JSON file containing scene information
        """
        try:
            with open(scene_info_file, 'r') as f:
                scene_info = json.load(f)
        except:
            print("Load Scene Info Error!")
            return
        actor_dis_mat = scene_info['actor_dis_lim_mat']
        actor_info_list = scene_info['actor_info_list']
        try:
            actor_pos_link = scene_info['actor_position_link']
        except:
            actor_pos_link = False
        try:    
            model_id_link = scene_info['model_id_link']
        except:
            model_id_link = False
        position_id = -1
        if actor_pos_link:
            position_id = np.random.randint(0, scene_info['actor_position_num'])
        model_id = -1
        if model_id_link:
            model_id = np.random.choice(self.id_list)
        actor_num = len(actor_info_list)
        actor_position = []

        def create_rand_pose(actor_info, now_actor_id):
            rotate_tag = actor_info['rotate_tag']
            ylim_prop = actor_info['ylim_prop']

            now_position_id = position_id if position_id != -1 else np.random.randint(0, len(actor_info["x_limit"]))
            x_lim = actor_info["x_limit"][now_position_id]
            y_lim = actor_info["y_limit"][min(now_position_id, len(actor_info["y_limit"])-1)]
            z_lim = actor_info["z_limit"][min(now_position_id, len(actor_info["z_limit"])-1)]
            qpose = actor_info['qpose'][min(now_position_id, len(actor_info["qpose"])-1)]
            rotate_limit = actor_info['rotate_limit'][min(now_position_id, len(actor_info["rotate_limit"])-1)] if rotate_tag else [0,0,0]
            pose = rand_pose(
                xlim=x_lim,
                ylim=y_lim,
                zlim=z_lim,
                ylim_prop=ylim_prop,
                rotate_rand=rotate_tag,
                rotate_lim=rotate_limit,
                qpos=qpose
            )

            recreate = True
            while recreate:
                recreate = False
                for i in range(now_actor_id):
                    las_actor_position = actor_position[i]
                    try:
                        dis = np.max(actor_dis_mat[i][now_actor_id], actor_dis_mat[now_actor_id][i])
                        if np.sum(pow(pose.p[:2] - las_actor_position,2)) < dis**2:
                            recreate = True
                    except:
                        continue
                if recreate:
                    pose = rand_pose(
                        xlim=x_lim,
                        ylim=y_lim,
                        zlim=z_lim,
                        ylim_prop=ylim_prop,
                        rotate_rand=rotate_tag,
                        rotate_limit=rotate_limit,
                        qpos=qpose
                    )
            actor_position.append(pose.p[:2])
            return pose

        for i in range(actor_num):
            actor_info = actor_info_list[i]
            actor_name = actor_info['actor_name']
            actor_type = actor_info['actor_type']
            is_static = actor_info['is_static_or_fix_root_link']
            try:
                convex = actor_info['convex']
            except:
                convex = False
            try:
                actor_file = actor_info['actor_file']
            except:
                actor_file = ''
            z_val_protect = actor_info['z_val_protect']
            actor_data_from = actor_info['actor_data_from']
            rand_model_id = actor_info['rand_model_id']

            setattr(self, actor_name, None)
            setattr(self, actor_name + "_data", actor_data_from)
            now_actor = getattr(self, actor_name)
            now_actor_data = getattr(self, actor_name + "_data")

            if actor_data_from != 'file':
                get_actor_data_func = getattr(self, actor_data_from)

            actor_pose = create_rand_pose(actor_info, i)
            
            if actor_type == 'box':
                now_actor = create_box(
                    self.scene,
                    pose=actor_pose,
                    half_size=actor_info["half_size"],
                    color=actor_info["color"],
                    is_static=is_static,
                    name=actor_name
                )
                now_actor_data = get_actor_data_func(actor_info["half_size"])
            elif actor_type == 'urdf':
                now_actor, now_actor_data = create_urdf_obj(
                    self.scene,
                    pose=actor_pose,
                    modelname=actor_file,
                    fix_root_link=is_static,
                )
                active_joints = now_actor.get_active_joints()
                for joint in active_joints:
                    joint.set_drive_property(stiffness=20, damping=5, force_limit=1000, mode="force")

            elif actor_type == 'obj' or actor_type == 'glb':
                now_model_id = None
                if rand_model_id:
                    now_model_id = model_id if model_id != -1 else np.random.choice(self.id_list)
                now_actor, now_actor_data = create_actor(
                    self.scene,
                    pose=actor_pose,
                    modelname=actor_file,
                    convex=convex,
                    is_static=is_static,
                    model_id=now_model_id,
                    z_val_protect=z_val_protect
                )
            else:
                print("Actor info type Error!")
                continue

            if is_static == False and actor_type != 'urdf':
                now_actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = actor_info["mass"]

            self.actor_name_dic[actor_name] = now_actor
            self.actor_data_dic[actor_name+'_data'] = now_actor_data

        # rand create actor, and no need to record specific information
        try:
            if scene_info["rand_create_actor"] == True:
                rand_actor_num = scene_info["rand_actor_num"]
                rand_actor_tol = len(scene_info["rand_choice_actor_list"])
                rand_actor_id_list = np.random.choice(rand_actor_tol, size = rand_actor_num, replace=True)
                rand_choice_actor_list = scene_info["rand_choice_actor_list"]
                for i,id in enumerate(rand_actor_id_list):
                    actor_info = rand_choice_actor_list[id]
                    actor_name = actor_info['actor_name']
                    actor_type = actor_info['actor_type']
                    is_static = actor_info['is_static_or_fix_root_link']
                    try:
                        convex = actor_info['convex']
                    except:
                        convex = False
                    try:
                        actor_file = actor_info['actor_file']
                    except:
                        actor_file = ''
                    z_val_protect = actor_info['z_val_protect']
                    actor_data_from = actor_info['actor_data_from']
                    rand_model_id = actor_info['rand_model_id']

                    if actor_data_from != 'file':
                        get_actor_data_func = getattr(self, actor_data_from)

                    actor_pose = create_rand_pose(actor_info, i + actor_num)

                    if actor_type == 'box':
                        now_actor = create_box(
                            self.scene,
                            pose=actor_pose,
                            half_size=actor_info["half_size"],
                            color=actor_info["color"],
                            is_static=is_static,
                            name=actor_name
                        )
                        now_actor_data = get_actor_data_func(actor_info["half_size"])
                    elif actor_type == 'urdf':
                        now_actor, now_actor_data = create_urdf_obj(
                            self.scene,
                            pose=actor_pose,
                            modelname=actor_file,
                            fix_root_link=is_static,
                        )
                        active_joints = now_actor.get_active_joints()
                        for joint in active_joints:
                            joint.set_drive_property(stiffness=20, damping=5, force_limit=1000, mode="force")

                    elif actor_type == 'obj' or actor_type == 'glb':
                        now_actor, now_actor_data = create_actor(
                            self.scene,
                            pose=actor_pose,
                            modelname=actor_file,
                            convex=convex,
                            is_static=is_static,
                            model_id=None if not rand_model_id else np.random.choice(self.id_list),
                            z_val_protect=z_val_protect
                        )
                    if is_static == False and actor_type != 'urdf':
                        now_actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = actor_info["mass"]
        except:         
            pass

        # add other data to 'actor_name_dic' and 'actor_data_dic'
        try:
            for other_actor_data in scene_info['other_actor_data']:
                self.actor_name_dic[other_actor_data] = getattr(self, other_actor_data)
                self.actor_data_dic[other_actor_data] = getattr(self, other_actor_data)
        except:
            pass

        # A short delay to make the actor stop shaking
        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def get_target_pose_from_goal_point_and_direction(self, actor, actor_data = None, endpose = None, target_pose = None, target_grasp_qpose = None):
        actor_matrix = actor.get_pose().to_transformation_matrix()
        local_target_matrix = np.asarray(actor_data['target_pose'][0])
        local_target_matrix[:3,3] *= actor_data['scale']
        res_matrix = np.eye(4)
        res_matrix[:3,3] = (actor_matrix  @ local_target_matrix)[:3,3] - endpose.global_pose.p
        res_matrix[:3,3] = np.linalg.inv(t3d.quaternions.quat2mat(endpose.global_pose.q) @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])) @ res_matrix[:3,3]
        res_pose = list(target_pose - t3d.quaternions.quat2mat(target_grasp_qpose) @ res_matrix[:3,3]) + target_grasp_qpose
        return res_pose

    def get_grasp_pose_w_labeled_direction(self, actor, actor_data, pre_dis = 0., id = 0):
        """
            Obtain the grasp pose through the marked grasp point.
            - actor: The instance of the object to be grasped.
            - actor_data: The annotation data corresponding to the instance of the object to be grasped.
            - pre_dis: The distance in front of the grasp point.
            - id: The index of the grasp point.
        """
        actor_matrix = actor.get_pose().to_transformation_matrix()
        local_contact_matrix = np.asarray(actor_data['contact_points_pose'][id])
        local_contact_matrix[:3,3] *= actor_data['scale']
        global_contact_pose_matrix = actor_matrix  @ local_contact_matrix @ np.array([[0, 0, 1, 0],
                                                                                      [-1,0, 0, 0],
                                                                                      [0, -1,0, 0],
                                                                                      [0, 0, 0, 1]])
        global_contact_pose_matrix_q = global_contact_pose_matrix[:3,:3]
        global_grasp_pose_p = global_contact_pose_matrix[:3,3] + global_contact_pose_matrix_q @ np.array([-0.12-pre_dis,0,0]).T
        global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
        res_pose = list(global_grasp_pose_p)+list(global_grasp_pose_q)
        return res_pose
    
    def get_grasp_pose_from_goal_point_and_direction(self, actor, actor_data,  endpose_tag: str, actor_functional_point_id = 0, target_point = None,
                                                     target_approach_direction = [0,0,1,0], actor_target_orientation = None, pre_dis = 0.):
        """
            Obtain the grasp pose through the given target point and contact direction.
            - actor: The instance of the object to be grasped.
            - actor_data: The annotation data corresponding to the instance of the object to be grasped.
            - endpose_tag: Left and right gripper marks, with values "left" or "right".
            - actor_functional_point_id: The index of the functional point to which the object to be grasped needs to be aligned.
            - target_point: The target point coordinates for aligning the functional points of the object to be grasped.
            - target_approach_direction: The direction of the grasped object's contact target point, 
                                         represented as a quaternion in the world coordinate system.
            - actor_target_orientation: The final target orientation of the object, 
                                        represented as a direction vector in the world coordinate system.
            - pre_dis: The distance in front of the grasp point.
        """
        target_approach_direction_mat = t3d.quaternions.quat2mat(target_approach_direction)
        actor_matrix = actor.get_pose().to_transformation_matrix()
        target_point_copy = deepcopy(target_point[:3])
        target_point_copy -= target_approach_direction_mat @ np.array([0,0,pre_dis])

        try:
            actor_orientation_point = np.array(actor_data['orientation_point'])[:3,3]
        except:
            actor_orientation_point = [0,0,0]

        if actor_target_orientation is not None:
            actor_target_orientation = actor_target_orientation / np.linalg.norm(actor_target_orientation)
        
        adjunction_matrix_list = [
            # 90 degree
            t3d.euler.euler2mat(0,0,0),
            t3d.euler.euler2mat(0,0,np.pi/2),
            t3d.euler.euler2mat(0,0,-np.pi/2),
            t3d.euler.euler2mat(0,0,np.pi),
            # 45 degree
            t3d.euler.euler2mat(0,0,np.pi/4),
            t3d.euler.euler2mat(0,0,np.pi*3/4),
            t3d.euler.euler2mat(0,0,-np.pi*3/4),
            t3d.euler.euler2mat(0,0,-np.pi/4),
        ]

        end_effector_pose = self.left_endpose if endpose_tag == 'left' else self.right_endpose
        res_pose = None
        res_eval= -1e10
        for adjunction_matrix in adjunction_matrix_list:
            local_target_matrix = np.asarray(actor_data['functional_matrix'][actor_functional_point_id])
            local_target_matrix[:3,3] *= actor_data['scale']
            fuctional_matrix = actor_matrix[:3,:3] @ np.asarray(actor_data['functional_matrix'][actor_functional_point_id])[:3,:3]
            fuctional_matrix = fuctional_matrix @ adjunction_matrix
            trans_matrix = target_approach_direction_mat @ np.linalg.inv(fuctional_matrix)
            end_effector_pose_matrix = t3d.quaternions.quat2mat(end_effector_pose.global_pose.q) @ np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            target_grasp_matrix = trans_matrix @ end_effector_pose_matrix

            # Use actor target orientation to filter
            if actor_target_orientation is not None:
                now_actor_orientation_point = trans_matrix @ actor_matrix[:3,:3] @ np.array(actor_orientation_point)
                now_actor_orientation_point = now_actor_orientation_point / np.linalg.norm(now_actor_orientation_point)
                produt = np.dot(now_actor_orientation_point, actor_target_orientation)
                # The difference from the target orientation is too large
                if produt < 0.8:
                    continue
            
            res_matrix = np.eye(4)
            res_matrix[:3,3] = (actor_matrix  @ local_target_matrix)[:3,3] - end_effector_pose.global_pose.p
            res_matrix[:3,3] = np.linalg.inv(end_effector_pose_matrix) @ res_matrix[:3,3]
            target_grasp_qpose = t3d.quaternions.mat2quat(target_grasp_matrix)
            # priget_grasp_pose_w_labeled_directionnt(target_grasp_matrix @ res_matrix[:3,3])
            now_pose = (target_point_copy - target_grasp_matrix @ res_matrix[:3,3]).tolist() + target_grasp_qpose.tolist()
            now_pose_eval = self.evaluate_grasp_pose(endpose_tag, now_pose, actor, is_grasp_actor=False, target_point=target_point[:3])
            if actor_target_orientation is not None and produt > res_eval or now_pose_eval > res_eval:
                res_pose = now_pose
                res_eval = now_pose_eval if actor_target_orientation is None else produt
        return res_pose
    
    # Get the pose coordinates of the actor's target point in the world coordinate system.
    # Return value: [x, y, z]
    def get_actor_goal_pose(self,actor,actor_data, id = 0):
        if type(actor) == list:
            return actor
        actor_matrix = actor.get_pose().to_transformation_matrix()
        local_target_matrix = np.asarray(actor_data['target_pose'][id])
        local_target_matrix[:3,3] *= actor_data['scale']
        return (actor_matrix @ local_target_matrix)[:3,3]

    # Get the actor's functional point and axis corresponding to the index in the world coordinate system.
    # Return value: [x, y, z, quaternion].
    def get_actor_functional_pose(self, actor, actor_data, actor_functional_point_id = 0):
        if type(actor) == list:
            return actor
        actor_matrix = actor.get_pose().to_transformation_matrix()
        # if "model_type" in actor_data.keys() and actor_data["model_type"] == "urdf": actor_matrix[:3,:3] = self.URDF_MATRIX
        local_functional_matrix = np.asarray(actor_data['functional_matrix'][actor_functional_point_id])
        local_functional_matrix[:3,3] *= actor_data['scale']
        res_matrix = actor_matrix @ local_functional_matrix
        return res_matrix[:3,3].tolist() + t3d.quaternions.mat2quat(res_matrix[:3,:3]).tolist()

    # Get the actor's grasp point and axis corresponding to the index in the world coordinate system.
    # Return value: [x, y, z, quaternion]
    def get_actor_contact_point_position(self, actor, actor_data, actor_contact_id = 0):
        if type(actor) == list:
            return actor
        actor_matrix = actor.get_pose().to_transformation_matrix()
        # if "model_type" in actor_data.keys() and actor_data["model_type"] == "urdf": actor_matrix[:3,:3] = self.URDF_MATRIX
        local_contact_matrix = np.asarray(actor_data['contact_points_pose'][actor_contact_id])
        local_contact_matrix[:3,3] *= actor_data['scale']
        res_matrix = actor_matrix @ local_contact_matrix
        return res_matrix[:3,3].tolist() + t3d.quaternions.mat2quat(res_matrix[:3,:3]).tolist()

    def get_grasp_pose_to_grasp_object(self, endpose_tag: str, actor, actor_data, pre_dis = 0):
        """
            Grasp the target object and obtain the appropriate grasp pose for the left and right arms when grasping the target actor.
            - endpose_tag: Left and right gripper marks, with values "left" or "right".
            - actor: The instance of the object to be grasped.
            - actor_data: The annotation data corresponding to the instance of the object to be grasped.
            - pre_dis: The distance in front of the grasp point.
        """
        endpose = self.left_endpose if endpose_tag == 'left' else self.right_endpose
        contact_points = actor_data['contact_points_pose']

        grasp_pose_eval = -1e9
        res_grasp_pose = None
        id = None

        mask = [True] * len(actor_data["contact_points_pose"])

        for i in range(len(actor_data["contact_points_group"])):
            if actor_data["contact_points_mask"][i] == False:
                for id in actor_data["contact_points_group"][i]:
                    mask[id] = False
        
        for i in range(len(contact_points)):
            if mask[i] == False:
                continue
            grasp_pose = self.get_grasp_pose_w_labeled_direction(actor, actor_data, pre_dis, i)
            now_grasp_pose_eval = self.evaluate_grasp_pose(endpose_tag, grasp_pose, actor, is_grasp_actor = True)
            if now_grasp_pose_eval > grasp_pose_eval:
                grasp_pose_eval = now_grasp_pose_eval
                res_grasp_pose = grasp_pose
                id = i
        
        for i, contact_points in enumerate(actor_data["contact_points_group"]):
            if id in contact_points:
                if endpose_tag == 'left':
                    self.left_prepare_grasp_data = actor_data
                    self.left_prepare_grasp_point_group = i
                else:
                    self.right_prepare_grasp_data = actor_data
                    self.right_prepare_grasp_point_group = i
                break
    
        return res_grasp_pose

    # Use the planner to test whether the target pose of the left and right robotic arms is reachable.
    def is_plan_success(self, endpose_tag: str, grasp_pose: list):
        planner = self.left_planner if endpose_tag == 'left' else self.right_planner
        arm_joint_id = self.left_arm_joint_id if endpose_tag == 'left' else self.right_arm_joint_id
        joint_pose = self.robot.get_qpos()
        qpos=[]
        for i in range(6):
            qpos.append(joint_pose[arm_joint_id[i]])
        
        las_robot_qpose = planner.robot.get_qpos()
        result = planner.plan_screw(
            target_pose=grasp_pose,
            qpos=qpos,
            time_step=1 / 250,
            use_point_cloud=False,
            use_attach=False,
        )
        planner.robot.set_qpos(las_robot_qpose,full=True)
        return result["status"] == "Success" and result["position"].shape[0] <= 2000
    
    def evaluate_grasp_pose(self, endpose_tag: str, grasp_pose: list, actor, is_grasp_actor = True, target_point = None):
        """
            Evaluate whether the grasp poses of the left and right arms are appropriate.
            - endpose_tag: Left and right gripper marks, with values "left" or "right".
            - grasp_pose: The target pose of the gripper that needs to be evaluated.
            - actor: The instance of the object to be grasped.
            - is_grasp_actor: Whether it is the pose being evaluated during the grasping of the actor, 
                              this parameter affects the evaluation criteria.
            - target_point: The the target position the object should reach after being grasped.
        """
        # Use the planner to test the grasp pose
        is_plan_suc = self.is_plan_success(endpose_tag=endpose_tag, grasp_pose=grasp_pose)
        if not is_plan_suc:
            return -1e10
        
        endpose = self.left_endpose if endpose_tag == 'left' else self.right_endpose
        base_xy = np.array([-0.3,-0.417]) if endpose_tag == 'left' else np.array([0.3,-0.417])
        actor_xy = np.array([actor.get_pose().p[0], actor.get_pose().p[1]]) if is_grasp_actor else np.array(target_point[:2])
        angle = np.arctan2(actor_xy[1]-base_xy[1], actor_xy[0]-base_xy[0])
        res = 0
        # The angles of the three rotation axes relative to the reference position should not exceed [-pi/2, pi/2]
        trans_matrix = t3d.quaternions.quat2mat(grasp_pose[3:]) @ np.linalg.inv(np.array([[0,-1,0],[1,0,0],[0,0,1]]))
        delta_euler = np.array(t3d.euler.mat2euler(trans_matrix))
        # print(delta_euler)
        if np.any(delta_euler > np.pi/2 + 0.1):
            res += 1e9
        base_pose = self.left_original_pose if endpose_tag == 'left' else self.right_original_pose
        distance = np.sqrt(np.sum((np.array(base_pose[:3]) - np.array(grasp_pose)[:3]) ** 2))
        if np.fabs(delta_euler[0]) > 1.2 and distance > 0.4:
            res += 1e9
        
        # Restrict the range of x, y, and z coordinates
        if endpose_tag == 'left':
            grasp_limit = [[-0.4,0.1],[-0.3,0.3],[0.85, 1.2]]
        elif endpose_tag == 'right':
            grasp_limit = [[-0.1, 0.4],[-0.3,0.3],[0.85, 1.2]]

        if np.any([grasp_pose[i] < grasp_limit[i][0] or grasp_pose[i] > grasp_limit[i][1] for i in range(3)]):
            res += 1e8

        # Calculate the evaluation value of the grasp pose
        res += np.sqrt(np.sum((endpose.global_pose.p - np.array(grasp_pose)[:3]) ** 2)) / 0.7
        trans_now_pose_matrix = t3d.quaternions.quat2mat(grasp_pose[3:]) @ np.linalg.inv(endpose.global_pose.to_transformation_matrix()[:3,:3])
        theta_xy = np.mod(np.abs(t3d.euler.mat2euler(trans_now_pose_matrix)[:2]), np.pi)
        theta_z = delta_euler[-1] + np.pi/2 - np.mod(angle + np.pi, np.pi)
        
        res += 2 * np.sum(theta_xy/np.pi) + 2 * np.abs(theta_z)/np.pi
        return -res

    def get_avoid_collision_pose(self, avoid_collision_arm_tag: str):
        """
            Obtain the poses of the left and right arms that can avoid arm collisions.
            - avoid_collision_arm_tag: The gripper mark that needs to avoid arm collisions, with values "left" or "right".
        """
        if avoid_collision_arm_tag == 'right':
            endpose = self.right_endpose.global_pose
            target_pose = [0.28, -0.07, endpose.p[2]] + t3d.quaternions.qmult(endpose.q, [0,1,0,0]).tolist()
        else:
            endpose = self.left_endpose.global_pose
            target_pose = [-0.28, -0.07, endpose.p[2]] + t3d.quaternions.qmult(endpose.q, [0,1,0,0]).tolist()

        return target_pose
    
    #  Get the contact points, target points, and functional points of the object.
    def get_actor_points_discription(self):
        res_dic = {}
        for actor_data_str in self.actor_data_dic.keys():
            res_dic[actor_data_str] = {}
        for actor_data_str in self.actor_data_dic.keys():
            try:
                res_dic[actor_data_str]['contact_points'] = self.actor_data_dic[actor_data_str]['contact_points_discription']
            except:
                res_dic[actor_data_str]['contact_points'] = ""

            try:
                res_dic[actor_data_str]['target_point'] = self.actor_data_dic[actor_data_str]['target_point_discription']
            except:
                res_dic[actor_data_str]['target_point'] = ""
            
            try:
                res_dic[actor_data_str]['functional_point'] = self.actor_data_dic[actor_data_str]['functional_point_discription']
            except:
                res_dic[actor_data_str]['functional_point'] = ""

            try:
                res_dic[actor_data_str]['actor_orientation'] = self.actor_data_dic[actor_data_str]['orientation_point_discription']
            except:
                res_dic[actor_data_str]['actor_orientation'] = ""

        return res_dic    

################################################# Generate Data API #################################################

    def play_once(self):
        pass
    
    def check_success(self):
        pass

    def pre_move(self):
        pass

    # ================= For Your Policy Deployment =================
    def apply_policy_demo(self, model):
        step_cnt = 0
        self.test_num += 1
        success_flag = False
        self._update_render()
        if self.render_freq:
            self.viewer.render()
        self.actor_pose = True
        
        while step_cnt < self.step_lim: # If it is not successful within the specified number of steps, it is judged as a failure.
            obs = self.get_obs() # get observation
            
            actions = model.get_action(obs) # TODO, get actions according to your policy and current obs

            left_arm_actions , left_gripper , left_current_qpos, left_path = [], [], [], []
            right_arm_actions , right_gripper , right_current_qpos, right_path = [], [], [], []

            left_arm_actions, left_gripper = actions[:, :6], actions[:, 6] # 0-5 left joint action, 6 left gripper action
            right_arm_actions, right_gripper = actions[:, 7:13], actions[:, 13] # 7-12 right joint action, 13 right gripper action
            left_current_qpos, right_current_qpos = obs['joint_action'][:6], obs['joint_action'][7:13]  # current joint and gripper action
            

            left_path = np.vstack((left_current_qpos, left_arm_actions))
            right_path = np.vstack((right_current_qpos, right_arm_actions))
            topp_left_flag, topp_right_flag = True, True
            try:
                times, left_pos, left_vel, acc, duration = self.left_planner.TOPP(left_path, 1/250, verbose=True)
                left_result = dict()
                left_result['position'], left_result['velocity'] = left_pos, left_vel
                left_n_step = left_result["position"].shape[0]
                left_gripper = np.linspace(left_gripper[0], left_gripper[-1], left_n_step)
            except:
                topp_left_flag = False
                left_n_step = 1

            try:
                times, right_pos, right_vel, acc, duration = self.right_planner.TOPP(right_path, 1/250, verbose=True)            
                right_result = dict()
                right_result['position'], right_result['velocity'] = right_pos, right_vel
                right_n_step = right_result["position"].shape[0]
                right_gripper = np.linspace(right_gripper[0], right_gripper[-1], right_n_step)
            except:
                topp_right_flag = False
                right_n_step = 1
            
            if right_n_step == 0:
                topp_right_flag = False
                right_n_step = 1
            
            step_cnt += actions.shape[0]
            
            n_step = max(left_n_step, right_n_step)

            obs_update_freq = n_step // actions.shape[0]

            now_left_id = 0 if topp_left_flag else 1e9
            now_right_id = 0 if topp_right_flag else 1e9
            i = 0
            
            while now_left_id < left_n_step or now_right_id < right_n_step:
                qf = self.robot.compute_passive_force(
                    gravity=True, coriolis_and_centrifugal=True
                )
                self.robot.set_qf(qf)
                if topp_left_flag and now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
                    for j in range(len(self.left_arm_joint_id)):
                        left_j = self.left_arm_joint_id[j]
                        self.active_joints[left_j].set_drive_target(left_result["position"][now_left_id][j])
                        self.active_joints[left_j].set_drive_velocity_target(left_result["velocity"][now_left_id][j])
                    if not self.fix_gripper:
                        for joint in self.active_joints[34:36]:
                            joint.set_drive_target(left_gripper[now_left_id])
                            joint.set_drive_velocity_target(0.05)
                            self.left_gripper_val = left_gripper[now_left_id]

                    now_left_id +=1
                    
                if topp_right_flag and now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
                    for j in range(len(self.right_arm_joint_id)):
                        right_j = self.right_arm_joint_id[j]
                        self.active_joints[right_j].set_drive_target(right_result["position"][now_right_id][j])
                        self.active_joints[right_j].set_drive_velocity_target(right_result["velocity"][now_right_id][j])
                    if not self.fix_gripper:
                        for joint in self.active_joints[36:38]:
                            joint.set_drive_target(right_gripper[now_right_id])
                            joint.set_drive_velocity_target(0.05)
                            self.right_gripper_val = right_gripper[now_right_id]

                    now_right_id +=1
                
                self.scene.step()
                self._update_render()

                if i % 5==0:
                    self._update_render()
                    if self.render_freq and i % self.render_freq == 0:
                        self.viewer.render()
                
                i += 1
                if self.check_success():
                    success_flag = True
                    break

                if self.actor_pose == False:
                    break
            
            self. _update_render()
            if self.render_freq:
                self.viewer.render()
            
            print(f'step: {step_cnt} / {self.step_lim}', end='\r')

            if success_flag:
                print("\nsuccess!")
                self.suc +=1
                return
            
            if self.actor_pose == False:
                break
            
        print("\nfail!")