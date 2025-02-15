from .base_task import Base_task
from .utils import *
import sapien
from math import pi

class tube_grasp_WBCD(Base_task):
    def setup_demo(self, **kwags):
        super()._init(**kwags, table_static=True)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        self.load_actors()

        self.left_tube_mid_position = [-0.3,-0.32,0.935]
        self.right_tube_mid_position = [0.3,-0.32,0.935]
        self.step_lim = 500
        # print(f"./task_config/scene_info/{self.task_name}.json")
        # print(self.actor_name_dic)
        # print(self.actor_data_dic)

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        self.test_tube,self.test_tube_data = rand_create_glb(
            self.scene,
            xlim=[-0.25,-0.15],
            ylim=[0,0],
            zlim=[0.76],
            modelname="045_test_tube",
            convex=True,
            rotate_rand=False,
            qpos=[1,0,0,0],
            scale=(1,1,1),
        )

        self.test_tube_rack, self.test_tube_rack_data = rand_create_glb(
            self.scene,
            xlim=[0],
            ylim=[0.25],
            zlim=[0.80],
            modelname="048_rackA",
            rotate_rand=False,
            convex=False,
            qpos=[0.5,0.5,-0.5,-0.5],
            is_static=True,
        )

        self.plastic_container, self.plastic_container_data = rand_create_glb(
            self.scene,
            xlim=[-0.2],
            ylim=[0.0],
            zlim=[0.74],
            modelname="047_plastic_container",
            rotate_rand=False,
            convex=False,
            qpos=[0.7071,0.7071,0,0],
            is_static=True,
            scale=(0.2,0.1,0.15),
        )
        
        self.test_tube.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.1
    def play_once(self):
        test_tube_pose = self.test_tube.get_pose().p

        if test_tube_pose[0] > 0:
            arm_tag = "right"
            move_function = self.right_move_to_pose_with_screw
            close_gripper_function = self.close_right_gripper
            open_gripper_function = self.open_right_gripper
            # tube_mid_position_data = self.actor_data_dic['right_tube_mid_position']
        else:
            arm_tag = "left"
            move_function = self.left_move_to_pose_with_screw
            close_gripper_function = self.close_left_gripper
            open_gripper_function = self.open_left_gripper
            # tube_mid_position_data = self.actor_data_dic['left_tube_mid_position']
        
        # Get the grasp pose for the beaker
        pre_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor=self.test_tube, actor_data=self.test_tube_data, grasp_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), pre_dis=0.1, id=0)
        target_grasp_pose = self.get_grasp_pose_w_labeled_direction(actor=self.test_tube, actor_data=self.test_tube_data, grasp_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), pre_dis=0.022, id=0)
        
        # Move to the pre-grasp pose
        move_function(pre_grasp_pose)
        
        # Move to the target grasp pose
        move_function(target_grasp_pose)
        
        # Close the gripper to grasp the beaker
        # Adjust the position to ensure a gentle grasp
        close_gripper_function(pos=-0.05)
        
        # Lift the beaker slightly
        lift_pose = pre_grasp_pose.copy()
        lift_pose[2] += 0.04  # Lift the beaker by 10 cm
        move_function(lift_pose)

        # mid_position = self.left_original_pose
        # print(self.left_original_pose)
        # mid_position[2] += 0.1
        mid_position = lift_pose[:3] + [1,0,0,1]
        mid_position[2] += 0.01
        mid_position[1] -= 0.2
        move_function(mid_position)

        tube_above_rack_pose = self.get_grasp_pose_w_labeled_direction(actor=self.test_tube_rack, actor_data=self.test_tube_rack_data, grasp_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), pre_dis=0.035, id=0)
        tube_above_rack_pose[2]+=0.12
        tube_above_rack_pose[0]+=0.015
        move_function(tube_above_rack_pose)

        tube_in_rack_pose = tube_above_rack_pose
        tube_in_rack_pose[2]-=0.09
        move_function(tube_in_rack_pose)

        open_gripper_function()


    
    # Check success
    def check_success(self):
    
        target_point = self.get_actor_contact_position(self.test_tube_rack, self.test_tube_rack_data, id = 1)
        place_point = self.get_actor_goal_pose(self.test_tube, self.test_tube_data)[:3]
        eps = 0.05
        return abs(place_point[0] - target_point[0])<eps  and  abs(place_point[1] - target_point[1])<eps and abs(place_point[2] - target_point[2])<eps
        # target_pose_p = np.array([0,-0.08])
        # target_pose_q = np.array([0.5,0.5,-0.5,-0.5])
        # eps = np.array([0.05,0.02,0.05,0.05,0.05,0.05])
        # return np.all(abs(beaker_pose_p[:2] - target_pose_p) < eps[:2]) and np.all(abs(beaker_pose_q - target_pose_q) < eps[-4:] ) and self.is_left_gripper_open() and self.is_right_gripper_open()