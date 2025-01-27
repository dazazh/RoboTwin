from .base_task import Base_task
from .utils import *
import sapien

class tube_grasp(Base_task):
    def setup_demo(self, **kwargs):
        super()._init(**kwargs, table_static=True)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera(kwargs.get('camera_w', 640), kwargs.get('camera_h', 480))
        self.pre_move()
        self.left_tube_mid_position = [-0.3,-0.32,0.935]
        self.right_tube_mid_position = [0.3,-0.32,0.935]
        self.load_actors(f"./task_config/scene_info/{self.task_name}.json")
        # print(f"./task_config/scene_info/{self.task_name}.json")
        # print(self.actor_name_dic)
        # print(self.actor_data_dic)

    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq
        
    def play_once(self):
        # Retrieve the beaker and coaster objects

        test_tube = self.actor_name_dic['test_tube']
        test_tube_rack = self.actor_name_dic['test_tube_rack']
        
        # Retrieve the beaker and coaster data
        test_tube_data = self.actor_data_dic['test_tube_data']
        test_tube_rack_data = self.actor_data_dic['test_tube_rack_data']
        
        # Get the current pose of the beaker
        test_tube_pose = self.get_actor_functional_pose(test_tube, test_tube_data)
        # Determine which arm to use based on the beaker's x-coordinate
        # while 1:
        #     self.close_left_gripper()
        if test_tube_pose[0] > 0:
            arm_tag = "right"
            move_function = self.right_move_to_pose_with_screw
            close_gripper_function = self.close_right_gripper
            open_gripper_function = self.open_right_gripper
            tube_mid_position_data = self.actor_data_dic['right_tube_mid_position']
        else:
            arm_tag = "left"
            move_function = self.left_move_to_pose_with_screw
            close_gripper_function = self.close_left_gripper
            open_gripper_function = self.open_left_gripper
            tube_mid_position_data = self.actor_data_dic['left_tube_mid_position']
        
        # Get the grasp pose for the beaker
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=test_tube, actor_data=test_tube_data, pre_dis=0.2)
        target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=test_tube, actor_data=test_tube_data, pre_dis=0.02)
        
        # Move to the pre-grasp pose
        move_function(pre_grasp_pose)
        
        # Move to the target grasp pose
        move_function(target_grasp_pose)
        
        # Close the gripper to grasp the beaker
        # Adjust the position to ensure a gentle grasp
        close_gripper_function(pos=-0.01)
        
        # Lift the beaker slightly
        lift_pose = pre_grasp_pose.copy()
        lift_pose[2] += 0.04  # Lift the beaker by 10 cm
        move_function(lift_pose)

        # mid_position = self.left_original_pose
        # print(self.left_original_pose)
        # mid_position[2] += 0.1
        mid_position = lift_pose[:3] + [1,0,0,1]
        mid_position[2] += 0.01
        mid_position[1] -= 0.02
        move_function(mid_position)
        
        tube_above_rack_point = self.get_actor_functional_pose(actor=test_tube_rack,actor_data=test_tube_rack_data,actor_functional_point_id=8)
        tube_above_rack_point[2] += 0.01
        # print(tube_above_rack_point)
        tube_above_rack_pose = self.get_grasp_pose_from_goal_point_and_direction(actor=test_tube, actor_data=test_tube_data, endpose_tag=arm_tag, actor_functional_point_id=1, target_point=tube_above_rack_point, target_approach_direction=self.world_direction_dic['top_down'], actor_target_orientation=[0, 1, 0], pre_dis=0)
        move_function(tube_above_rack_pose)

        # tube_put_rack_pose = self.get_grasp_pose_from_goal_point_and_direction(actor=test_tube, actor_data=test_tube_data, endpose_tag=arm_tag, actor_functional_point_id=1, target_point=tube_above_rack_point, target_approach_direction=self.world_direction_dic['top_down'], actor_target_orientation=[0, 1, 0], pre_dis=-0.02)
        
        # tube_put_rack_pose = tube_above_rack_pose
        # tube_put_rack_pose[3] -= 0.04
        # move_function(tube_put_rack_pose)

        tube_in_rack_point = self.get_actor_functional_pose(test_tube_rack, test_tube_rack_data, actor_functional_point_id=9)[:3]
        tube_in_rack_point[2] += 0.04
        tube_in_rack_pose = self.get_grasp_pose_from_goal_point_and_direction(actor=test_tube, actor_data=test_tube_data, endpose_tag=arm_tag, actor_functional_point_id=1, target_point=tube_in_rack_point, target_approach_direction=self.world_direction_dic['top_down'], actor_target_orientation=[0, 1, 0], pre_dis=0.0)
        move_function(tube_in_rack_pose)
        self.open_left_gripper()
        move_function(self.left_original_pose)

        # while True:
        #     self.open_left_gripper()
        # # Get the target pose for placing the beaker on the coaster
        # coaster_pose = self.get_actor_goal_pose(coaster, coaster_data, id=0)
        # target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(
        #     actor=beaker, actor_data=beaker_data, endpose_tag=arm_tag, actor_functional_point_id=1,
        #     target_point=coaster_pose, target_approach_direction=self.world_direction_dic['top_down'],
        #     actor_target_orientation=[0, 1, 0], pre_dis=0.09
        # )
        
        # # Move to the pre-place pose
        # move_function(target_place_pose)
        
        # # Move to the target place pose
        # target_place_pose[2] -= 0.09  # Adjust the height to place the beaker on the coaster
        # move_function(target_place_pose)
        
        # # Open the gripper to place the beaker
        # open_gripper_function()
        
        # # Lift the arm slightly after placing the beaker
        # lift_pose = target_place_pose.copy()
        # lift_pose[2] += 0.1  # Lift the arm by 10 cm
        # move_function(lift_pose)
    
    # Check success
    def check_success(self):
        test_tube = self.actor_name_dic['test_tube']
        test_tube_data = self.actor_data_dic['test_tube_data']

        test_tube_rack = self.actor_name_dic['test_tube_rack']
        test_tube_rack_data = self.actor_data_dic['test_tube_rack_data']
        
        place_point = self.get_actor_functional_pose(test_tube, test_tube_data, actor_functional_point_id=1)[:3]
        target_point = self.get_actor_functional_pose(test_tube_rack, test_tube_rack_data, actor_functional_point_id = 9)[:3]
        
        eps = 0.05
        return abs(place_point[0] - target_point[0])<eps  and  abs(place_point[1] - target_point[1])<eps and abs(place_point[2] - target_point[2])<eps
        # target_pose_p = np.array([0,-0.08])
        # target_pose_q = np.array([0.5,0.5,-0.5,-0.5])
        # eps = np.array([0.05,0.02,0.05,0.05,0.05,0.05])
        # return np.all(abs(beaker_pose_p[:2] - target_pose_p) < eps[:2]) and np.all(abs(beaker_pose_q - target_pose_q) < eps[-4:] ) and self.is_left_gripper_open() and self.is_right_gripper_open()