
from .base_task import Base_task
from .beaker_grasp import beaker_grasp
from .utils import *
import sapien

class gpt_beaker_grasp(beaker_grasp):
    def play_once(self):
        # Retrieve the beaker and coaster objects
        beaker = self.actor_name_dic['beaker']
        coaster = self.actor_name_dic['coaster']
        
        # Retrieve the beaker and coaster data
        beaker_data = self.actor_data_dic['beaker_data']
        coaster_data = self.actor_data_dic['coaster_data']
        
        # Get the current pose of the beaker
        beaker_pose = self.get_actor_functional_pose(beaker, beaker_data)
        
        # Determine which arm to use based on the beaker's x-coordinate
        if beaker_pose[0] > 0:
            arm_tag = "right"
            move_function = self.right_move_to_pose_with_screw
            close_gripper_function = self.close_right_gripper
            open_gripper_function = self.open_right_gripper
        else:
            arm_tag = "left"
            move_function = self.left_move_to_pose_with_screw
            close_gripper_function = self.close_left_gripper
            open_gripper_function = self.open_left_gripper
        
        # Get the grasp pose for the beaker
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=beaker, actor_data=beaker_data, pre_dis=0.09)
        target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=beaker, actor_data=beaker_data, pre_dis=0)
        
        # Move to the pre-grasp pose
        move_function(pre_grasp_pose)
        
        # Move to the target grasp pose
        move_function(target_grasp_pose)
        
        # Close the gripper to grasp the beaker
        close_gripper_function(pos=0.01)  # Adjust the position to ensure a gentle grasp
        
        # Lift the beaker slightly
        lift_pose = pre_grasp_pose.copy()
        lift_pose[2] += 0.1  # Lift the beaker by 10 cm
        move_function(lift_pose)
        
        # Get the target pose for placing the beaker on the coaster
        coaster_pose = self.get_actor_goal_pose(coaster, coaster_data, id=0)
        target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(
            actor=beaker, actor_data=beaker_data, endpose_tag=arm_tag, actor_functional_point_id=1,
            target_point=coaster_pose, target_approach_direction=self.world_direction_dic['top_down'],
            actor_target_orientation=[0, 1, 0], pre_dis=0.09
        )
        
        # Move to the pre-place pose
        move_function(target_place_pose)
        
        # Move to the target place pose
        target_place_pose[2] -= 0.09  # Adjust the height to place the beaker on the coaster
        move_function(target_place_pose)
        
        # Open the gripper to place the beaker
        open_gripper_function()
        
        # Lift the arm slightly after placing the beaker
        lift_pose = target_place_pose.copy()
        lift_pose[2] += 0.1  # Lift the arm by 10 cm
        move_function(lift_pose)
