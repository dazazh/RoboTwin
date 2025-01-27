
from .base_task import Base_task
from .mug_hanging import mug_hanging
from .utils import *
import sapien

class gpt_mug_hanging(mug_hanging):
    def play_once(self):
        # Retrieve the actor objects
        mug = self.actor_name_dic['mug']
        rack = self.actor_name_dic['rack']
        middle_pose_of_left_arm = self.actor_name_dic['middle_pose_of_left_arm']

        # Retrieve the actor_data objects
        mug_data = self.actor_data_dic['mug_data']
        rack_data = self.actor_data_dic['rack_data']
        middle_pose_of_left_arm_data = self.actor_data_dic['middle_pose_of_left_arm']

        # Step 1: Move the left arm to grasp the mug
        pre_grasp_pose_left = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=mug, actor_data=mug_data, pre_dis=0.09)
        target_grasp_pose_left = self.get_grasp_pose_to_grasp_object(endpose_tag="left", actor=mug, actor_data=mug_data, pre_dis=0)

        self.left_move_to_pose_with_screw(pre_grasp_pose_left)
        self.left_move_to_pose_with_screw(target_grasp_pose_left)
        self.close_left_gripper(pos=-0.01)  # Tighten the gripper to pick up the mug

        # Step 2: Move the mug to the middle position
        middle_pos = self.get_actor_goal_pose(middle_pose_of_left_arm, middle_pose_of_left_arm_data)
        target_pose_middle = self.get_grasp_pose_from_goal_point_and_direction(mug, mug_data, endpose_tag="left", actor_functional_point_id=1, target_point=middle_pos, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0.09)
        target_pose_middle_final = self.get_grasp_pose_from_goal_point_and_direction(mug, mug_data, endpose_tag="left", actor_functional_point_id=1, target_point=middle_pos, target_approach_direction=self.world_direction_dic['top_down'], pre_dis=0)

        self.left_move_to_pose_with_screw(target_pose_middle)
        self.left_move_to_pose_with_screw(target_pose_middle_final)
        self.open_left_gripper()  # Release the mug at the middle position

        # Step 3: Avoid collision with the left arm
        left_avoid_collision_pose = self.get_avoid_collision_pose(avoid_collision_arm_tag='left')
        self.left_move_to_pose_with_screw(left_avoid_collision_pose)

        # Step 4: Move the right arm to grasp the mug
        pre_grasp_pose_right = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=mug, actor_data=mug_data, pre_dis=0.09)
        target_grasp_pose_right = self.get_grasp_pose_to_grasp_object(endpose_tag="right", actor=mug, actor_data=mug_data, pre_dis=0)

        self.right_move_to_pose_with_screw(pre_grasp_pose_right)
        self.right_move_to_pose_with_screw(target_grasp_pose_right)
        self.close_right_gripper(pos=-0.01)  # Tighten the gripper to pick up the mug

        # Step 5: Move the mug to the rack
        rack_functional_pose = self.get_actor_functional_pose(rack, rack_data, actor_functional_point_id=0)
        rack_point = rack_functional_pose[:3]
        rack_approach_direction = rack_functional_pose[3:]

        pre_rack_pose = self.get_grasp_pose_from_goal_point_and_direction(mug, mug_data, endpose_tag="right", actor_functional_point_id=0, target_point=rack_point, target_approach_direction=rack_approach_direction, pre_dis=0.09)
        target_rack_pose = self.get_grasp_pose_from_goal_point_and_direction(mug, mug_data, endpose_tag="right", actor_functional_point_id=0, target_point=rack_point, target_approach_direction=rack_approach_direction, pre_dis=0)

        self.right_move_to_pose_with_screw(pre_rack_pose)
        self.right_move_to_pose_with_screw(target_rack_pose)
        self.open_right_gripper()  # Release the mug on the rack