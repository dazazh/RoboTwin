
from .base_task import Base_task
from .empty_cup_place import empty_cup_place
from .utils import *
import sapien

class gpt_empty_cup_place(empty_cup_place):
    def play_once(self):
        # Retrieve the actor objects and data
        cup = self.actor_name_dic['cup']
        coaster = self.actor_name_dic['coaster']
        cup_data = self.actor_data_dic['cup_data']
        coaster_data = self.actor_data_dic['coaster_data']

        # Get the cup's pose
        cup_pose = self.get_actor_functional_pose(cup, cup_data)

        # Determine which arm to use based on the cup's x-coordinate
        if cup_pose[0] > 0:
            arm_tag = "right"
            move_function = self.right_move_to_pose_with_screw
            close_gripper_function = self.close_right_gripper
            open_gripper_function = self.open_right_gripper
        else:
            arm_tag = "left"
            move_function = self.left_move_to_pose_with_screw
            close_gripper_function = self.close_left_gripper
            open_gripper_function = self.open_left_gripper

        # Get the pre-grasp and grasp poses for the cup
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=cup, actor_data=cup_data, pre_dis=0.05)
        grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=cup, actor_data=cup_data, pre_dis=0)

        # Move to the pre-grasp position
        move_function(pre_grasp_pose)

        # Move to the grasp position
        move_function(grasp_pose)

        # Close the gripper to pick up the cup
        close_gripper_function(pos=-0.01)  # Tighten the gripper

        # Lift the cup by moving back to the pre-grasp position
        move_function(pre_grasp_pose)

        # Get the coaster's target pose
        coaster_target_pose = self.get_actor_goal_pose(coaster, coaster_data, id=0)

        # Get the pre-place and place poses for the coaster
        pre_place_pose = self.get_grasp_pose_from_goal_point_and_direction(actor=cup, actor_data=cup_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=coaster_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0.05)
        place_pose = self.get_grasp_pose_from_goal_point_and_direction(actor=cup, actor_data=cup_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=coaster_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)

        # Move to the pre-place position
        move_function(pre_place_pose)

        # Move to the place position
        move_function(place_pose)

        # Open the gripper to place the cup
        open_gripper_function()

        # Move back to avoid collisions
        move_function(pre_place_pose)
