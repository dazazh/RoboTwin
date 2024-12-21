
from .base_task import Base_task
from .diverse_bottles_pick import diverse_bottles_pick
from .utils import *
import sapien

class gpt_diverse_bottles_pick(diverse_bottles_pick):
    def play_once(self):
        # Retrieve the actor objects and their data
        bottle1 = self.actor_name_dic['bottle1']
        bottle2 = self.actor_name_dic['bottle2']
        bottle1_data = self.actor_data_dic['bottle1_data']
        bottle2_data = self.actor_data_dic['bottle2_data']
        left_target_position = self.actor_name_dic['left_bottle_target_position']
        right_target_position = self.actor_name_dic['right_bottle_target_position']
        left_target_position_data = self.actor_data_dic['left_bottle_target_position']
        right_target_position_data = self.actor_data_dic['right_bottle_target_position']

        # Get the target positions for the bottles
        left_target_pose = self.get_actor_goal_pose(left_target_position, left_target_position_data, id=0)
        right_target_pose = self.get_actor_goal_pose(right_target_position, right_target_position_data, id=0)

        # Get the grasp poses for the bottles
        left_pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=bottle1, actor_data=bottle1_data, pre_dis=0.09)
        left_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=bottle1, actor_data=bottle1_data, pre_dis=0)
        right_pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=bottle2, actor_data=bottle2_data, pre_dis=0.09)
        right_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=bottle2, actor_data=bottle2_data, pre_dis=0)

        # Move to the pre-grasp poses
        self.together_move_to_pose_with_screw(left_pre_grasp_pose, right_pre_grasp_pose)

        # Move to the grasp poses
        self.together_move_to_pose_with_screw(left_grasp_pose, right_grasp_pose)

        # Close the grippers to grasp the bottles
        self.together_close_gripper()

        # Move the bottles to the target positions
        left_target_grasp_pose = self.get_grasp_pose_from_goal_point_and_direction(bottle1, bottle1_data, endpose_tag="left", actor_functional_point_id=0, target_point=left_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)
        right_target_grasp_pose = self.get_grasp_pose_from_goal_point_and_direction(bottle2, bottle2_data, endpose_tag="right", actor_functional_point_id=0, target_point=right_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)

        # Move to the target positions
        self.together_move_to_pose_with_screw(left_target_grasp_pose, right_target_grasp_pose)

        # Note: The bottles are not put down, so the grippers remain closed
