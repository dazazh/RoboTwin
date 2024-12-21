
from .base_task import Base_task
from .blocks_stack_easy import blocks_stack_easy
from .utils import *
import sapien

class gpt_blocks_stack_easy(blocks_stack_easy):
    def play_once(self):
        # Retrieve actor objects and data
        block1 = self.actor_name_dic['block1']
        block2 = self.actor_name_dic['block2']
        block1_target_pose = self.actor_name_dic['block1_target_pose']
        
        block1_data = self.actor_data_dic['block1_data']
        block2_data = self.actor_data_dic['block2_data']
        block1_target_pose_data = self.actor_data_dic['block1_target_pose']
        
        # Determine which arm to use for block1
        block1_pose = self.get_actor_goal_pose(block1, block1_data, 0)
        if block1_pose[0] > 0:
            arm_tag_block1 = "right"
            arm_tag_block2 = "left"
        else:
            arm_tag_block1 = "left"
            arm_tag_block2 = "right"
        
        # Pick up block1 and move it to the target position
        self.pick_and_place_block(block1, block1_data, block1_target_pose, block1_target_pose_data, arm_tag_block1)
        
        # Determine which arm to use for block2
        block2_pose = self.get_actor_goal_pose(block2, block2_data, 0)
        if block2_pose[0] > 0:
            arm_tag_block2 = "right"
        else:
            arm_tag_block2 = "left"
        
        # Pick up block2 and place it on block1
        self.pick_and_place_block(block2, block2_data, block1, block1_data, arm_tag_block2)
        
    def pick_and_place_block(self, block, block_data, target_block, target_block_data, arm_tag):
        # Get the pre-grasp and target grasp poses
        pre_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=block, actor_data=block_data, pre_dis=0.09)
        target_grasp_pose = self.get_grasp_pose_to_grasp_object(endpose_tag=arm_tag, actor=block, actor_data=block_data, pre_dis=0)
        
        # Move to the pre-grasp pose
        self.move_arm_to_pose(arm_tag, pre_grasp_pose)
        
        # Move to the target grasp pose
        self.move_arm_to_pose(arm_tag, target_grasp_pose)
        
        # Close the gripper to grasp the block
        self.close_gripper(arm_tag)
        
        # Lift the block up
        self.move_arm_to_pose(arm_tag, pre_grasp_pose)
        
        # Get the target pose for placing the block
        target_pose = self.get_actor_goal_pose(target_block, target_block_data, 0)
        target_approach_direction = self.world_direction_dic['top_down']
        pre_place_pose = self.get_grasp_pose_from_goal_point_and_direction(block, block_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=target_pose, target_approach_direction=target_approach_direction, pre_dis=0.09)
        target_place_pose = self.get_grasp_pose_from_goal_point_and_direction(block, block_data, endpose_tag=arm_tag, actor_functional_point_id=0, target_point=target_pose, target_approach_direction=target_approach_direction, pre_dis=0)
        
        # Move to the pre-place pose
        self.move_arm_to_pose(arm_tag, pre_place_pose)
        
        # Move to the target place pose
        self.move_arm_to_pose(arm_tag, target_place_pose)
        
        # Open the gripper to place the block
        self.open_gripper(arm_tag)
        
        # Move the arm away to avoid collision
        avoid_collision_pose = self.get_avoid_collision_pose(arm_tag)
        self.move_arm_to_pose(arm_tag, avoid_collision_pose)
        
    def move_arm_to_pose(self, arm_tag, pose):
        if arm_tag == "left":
            self.left_move_to_pose_with_screw(pose)
        elif arm_tag == "right":
            self.right_move_to_pose_with_screw(pose)
    
    def close_gripper(self, arm_tag):
        if arm_tag == "left":
            self.close_left_gripper()
        elif arm_tag == "right":
            self.close_right_gripper()
    
    def open_gripper(self, arm_tag):
        if arm_tag == "left":
            self.open_left_gripper()
        elif arm_tag == "right":
            self.open_right_gripper()
