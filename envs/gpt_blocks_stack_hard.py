
from .base_task import Base_task
from .blocks_stack_hard import blocks_stack_hard
from .utils import *
import sapien

class gpt_blocks_stack_hard(blocks_stack_hard):
    def play_once(self):
        # Retrieve actor objects
        block1 = self.actor_name_dic['block1']
        block2 = self.actor_name_dic['block2']
        block3 = self.actor_name_dic['block3']
        block1_target_pose = self.actor_name_dic['block1_target_pose']

        # Retrieve actor data objects
        block1_data = self.actor_data_dic['block1_data']
        block2_data = self.actor_data_dic['block2_data']
        block3_data = self.actor_data_dic['block3_data']
        block1_target_pose_data = self.actor_data_dic['block1_target_pose']

        # Define pre-dis for grasping and placing
        pre_dis = 0.08

        # Function to grasp and place a block
        def grasp_and_place(block, block_data, target_pose, target_pose_data, pre_dis):
            # Determine which arm to use based on the block's x coordinate
            block_pose = self.get_actor_goal_pose(block, block_data)
            if block_pose[0] > 0:
                arm_tag = "right"
                move_function = self.right_move_to_pose_with_screw
                close_gripper_function = self.close_right_gripper
                open_gripper_function = self.open_right_gripper
            else:
                arm_tag = "left"
                move_function = self.left_move_to_pose_with_screw
                close_gripper_function = self.close_left_gripper
                open_gripper_function = self.open_left_gripper

            # Get the grasp pose
            pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=block, actor_data=block_data, pre_dis=pre_dis)
            target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=block, actor_data=block_data, pre_dis=0)

            # Move to the pre-grasp pose
            move_function(pre_grasp_pose)

            # Move to the grasp pose
            move_function(target_grasp_pose)

            # Close the gripper to grasp the block
            close_gripper_function()

            # Lift the block up
            move_function(pre_grasp_pose)

            # Get the target pose for placing the block
            target_point = self.get_actor_goal_pose(target_pose, target_pose_data)
            target_approach_direction = self.world_direction_dic['top_down']
            pre_place_pose = self.get_grasp_pose_from_goal_point_and_direction(block, block_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=target_point, target_approach_direction=target_approach_direction, pre_dis=pre_dis)
            target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(block, block_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=target_point, target_approach_direction=target_approach_direction, pre_dis=0)

            # Move to the pre-place pose
            move_function(pre_place_pose)

            # Move to the place pose
            move_function(target_place_pose)

            # Open the gripper to place the block
            open_gripper_function()

            # Lift the arm up
            move_function(pre_place_pose)

        # Grasp and place block1
        grasp_and_place(block1, block1_data, block1_target_pose, block1_target_pose_data, pre_dis)

        # Avoid collision if necessary
        if self.get_actor_goal_pose(block1, block1_data)[0] > 0:
            avoid_collision_pose = self.get_avoid_collision_pose(avoid_collision_arm_tag='left')
            self.left_move_to_pose_with_screw(avoid_collision_pose)
        else:
            avoid_collision_pose = self.get_avoid_collision_pose(avoid_collision_arm_tag='right')
            self.right_move_to_pose_with_screw(avoid_collision_pose)

        # Grasp and place block2 on top of block1
        grasp_and_place(block2, block2_data, block1, block1_data, pre_dis)

        # Avoid collision if necessary
        if self.get_actor_goal_pose(block2, block2_data)[0] > 0:
            avoid_collision_pose = self.get_avoid_collision_pose(avoid_collision_arm_tag='left')
            self.left_move_to_pose_with_screw(avoid_collision_pose)
        else:
            avoid_collision_pose = self.get_avoid_collision_pose(avoid_collision_arm_tag='right')
            self.right_move_to_pose_with_screw(avoid_collision_pose)

        # Grasp and place block3 on top of block2
        grasp_and_place(block3, block3_data, block2, block2_data, pre_dis)

        # Avoid collision if necessary
        if self.get_actor_goal_pose(block3, block3_data)[0] > 0:
            avoid_collision_pose = self.get_avoid_collision_pose(avoid_collision_arm_tag='left')
            self.left_move_to_pose_with_screw(avoid_collision_pose)
        else:
            avoid_collision_pose = self.get_avoid_collision_pose(avoid_collision_arm_tag='right')
            self.right_move_to_pose_with_screw(avoid_collision_pose)
