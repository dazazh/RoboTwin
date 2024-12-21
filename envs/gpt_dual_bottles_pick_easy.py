
from .base_task import Base_task
from .dual_bottles_pick_easy import dual_bottles_pick_easy
from .utils import *
import sapien

class gpt_dual_bottles_pick_easy(dual_bottles_pick_easy):
    def play_once(self):
        # Retrieve the actors and their data
        red_bottle = self.actor_name_dic['red_bottle']
        green_bottle = self.actor_name_dic['green_bottle']
        left_target_position = self.actor_name_dic['left_bottle_target_position']
        right_target_position = self.actor_name_dic['right_bottle_target_position']

        red_bottle_data = self.actor_data_dic['red_bottle_data']
        green_bottle_data = self.actor_data_dic['green_bottle_data']
        left_target_position_data = self.actor_data_dic['left_bottle_target_position']
        right_target_position_data = self.actor_data_dic['right_bottle_target_position']

        # Get the target positions for the bottles
        left_target_pose = self.get_actor_goal_pose(left_target_position, left_target_position_data, id=0)
        right_target_pose = self.get_actor_goal_pose(right_target_position, right_target_position_data, id=0)

        # Get the grasp poses for the bottles
        red_pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=red_bottle, actor_data=red_bottle_data, pre_dis=0.09)
        red_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=red_bottle, actor_data=red_bottle_data, pre_dis=0)

        green_pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=green_bottle, actor_data=green_bottle_data, pre_dis=0.09)
        green_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=green_bottle, actor_data=green_bottle_data, pre_dis=0)

        # Move both arms to the pre-grasp positions
        self.together_move_to_pose_with_screw(red_pre_grasp_pose, green_pre_grasp_pose)

        # Move both arms to the grasp positions
        self.together_move_to_pose_with_screw(red_grasp_pose, green_grasp_pose)

        # Close both grippers to grasp the bottles
        self.together_close_gripper()

        # Get the target poses for the bottles at the target positions
        red_target_pose = self.get_grasp_pose_from_goal_point_and_direction(red_bottle, red_bottle_data, endpose_tag="left", actor_functional_point_id=0, target_point=left_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)
        green_target_pose = self.get_grasp_pose_from_goal_point_and_direction(green_bottle, green_bottle_data, endpose_tag="right", actor_functional_point_id=0, target_point=right_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)

        # Move both arms to the target positions
        self.together_move_to_pose_with_screw(red_target_pose, green_target_pose)

        # Note: The bottles are not put down at the end, as per the task description.

# Example usage:
# instance = gpt_dual_bottles_pick_easy()
# instance.play_once()
