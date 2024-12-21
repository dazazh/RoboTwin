
from .base_task import Base_task
from .dual_bottles_pick_hard import dual_bottles_pick_hard
from .utils import *
import sapien

class gpt_dual_bottles_pick_hard(dual_bottles_pick_hard):
    def play_once(self):
        # Retrieve the actor objects and their corresponding data
        red_bottle = self.actor_name_dic['red_bottle']
        green_bottle = self.actor_name_dic['green_bottle']
        left_target_position = self.actor_name_dic['left_bottle_target_position']
        right_target_position = self.actor_name_dic['right_bottle_target_position']

        red_bottle_data = self.actor_data_dic['red_bottle_data']
        green_bottle_data = self.actor_data_dic['green_bottle_data']
        left_target_position_data = self.actor_data_dic['left_bottle_target_position']
        right_target_position_data = self.actor_data_dic['right_bottle_target_position']

        # Get the target poses for the bottles
        left_target_pose = self.get_actor_goal_pose(left_target_position, left_target_position_data, id=0)
        right_target_pose = self.get_actor_goal_pose(right_target_position, right_target_position_data, id=0)

        # Get the functional poses of the bottles
        red_bottle_functional_pose = self.get_actor_functional_pose(red_bottle, red_bottle_data)
        green_bottle_functional_pose = self.get_actor_functional_pose(green_bottle, green_bottle_data)

        # Get the grasp poses for the bottles
        red_bottle_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=red_bottle, actor_data=red_bottle_data, pre_dis=0.09)
        green_bottle_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=green_bottle, actor_data=green_bottle_data, pre_dis=0.09)

        # Move both arms to the pre-grasp poses
        self.together_move_to_pose_with_screw(red_bottle_grasp_pose, green_bottle_grasp_pose)

        # Move both arms to the grasp poses
        red_bottle_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=red_bottle, actor_data=red_bottle_data, pre_dis=0)
        green_bottle_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=green_bottle, actor_data=green_bottle_data, pre_dis=0)
        self.together_move_to_pose_with_screw(red_bottle_grasp_pose, green_bottle_grasp_pose)

        # Close both grippers to grasp the bottles
        self.together_close_gripper()

        # Lift the bottles upward
        red_bottle_lift_pose = red_bottle_grasp_pose.copy()
        green_bottle_lift_pose = green_bottle_grasp_pose.copy()
        red_bottle_lift_pose[2] += 0.1  # Lift the red bottle 10 cm upward
        green_bottle_lift_pose[2] += 0.1  # Lift the green bottle 10 cm upward
        self.together_move_to_pose_with_screw(red_bottle_lift_pose, green_bottle_lift_pose)

        # Move the bottles to the target positions
        # left_target_grasp_pose = self.get_grasp_pose_from_goal_point_and_direction(red_bottle, red_bottle_data, endpose_tag="left", actor_functional_point_id=0, target_point=left_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0.09)
        # right_target_grasp_pose = self.get_grasp_pose_from_goal_point_and_direction(green_bottle, green_bottle_data, endpose_tag="right", actor_functional_point_id=0, target_point=right_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0.09)
        # self.together_move_to_pose_with_screw(left_target_grasp_pose, right_target_grasp_pose)

        # Move the bottles to the final target positions
        left_target_grasp_pose = self.get_grasp_pose_from_goal_point_and_direction(red_bottle, red_bottle_data, endpose_tag="left", actor_functional_point_id=0, target_point=left_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)
        right_target_grasp_pose = self.get_grasp_pose_from_goal_point_and_direction(green_bottle, green_bottle_data, endpose_tag="right", actor_functional_point_id=0, target_point=right_target_pose, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)
        self.together_move_to_pose_with_screw(left_target_grasp_pose, right_target_grasp_pose)

        # No need to open the grippers or put down the bottles
